import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from models import SASRecModel
from trainers import ELCRecTrainer
from trainers import DualExpertTrainer 
from utils import get_user_seqs, check_path, set_seed
from datasets import RecWithContrastiveLearningDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from utils import EarlyStopping, get_user_seqs, get_item2attribute_json, check_path, set_seed
from torch.optim.lr_scheduler import CosineAnnealingLR

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data/", type=str)
    parser.add_argument("--output_dir", default="output_yelp/", type=str)
    parser.add_argument("--data_name", default="Sports_and_Outdoors", type=str)
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--model_idx", default=0, type=int, help="model idenfier 10, 20, 30...")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
    parser.add_argument("--noise_ratio", default=0.0, type=float, help="percentage of negative interactions in a sequence - robustness analysis")
    parser.add_argument("--training_data_ratio", default=1.0, type=float,help="percentage of training samples used for training - robustness analysis")
    parser.add_argument("--augment_type", default="random", type=str, help="default data augmentation types. Chosen from: mask, crop, reorder, substitute, insert, random, combinatorial_enumerate (for multi-view).")
    parser.add_argument("--tao", type=float, default=0.2, help="crop ratio for crop operator")
    parser.add_argument("--gamma", type=float, default=0.7, help="mask ratio for mask operator")
    parser.add_argument("--beta", type=float, default=0.2, help="reorder ratio for reorder operator")
    parser.add_argument("--temperature", default=1.0, type=float, help="softmax temperature (default:  1.0) - not studied.")
    parser.add_argument("--n_views", default=2, type=int, metavar="N", help="Number of augmented data for each sequence - not studied.")
    parser.add_argument("--contrast_type", default="Hybrid", type=str, help="Ways to contrastive of. Support InstanceCL and ShortInterestCL, IntentCL, and Hybrid types.")
    parser.add_argument("--num_intent_clusters", default="256", type=str, help="Number of cluster of intents. Activated only when using IntentCL or Hybrid types.")
    parser.add_argument("--seq_representation_type", default="mean", type=str, help="operate of item representation overtime. Support types: mean, concatenate")
    parser.add_argument("--seq_representation_instancecl_type", default="concatenate", type=str, help="operate of item representation overtime. Support types: mean, concatenate")
    parser.add_argument("--warm_up_epoches", type=float, default=0, help="number of epochs to start IntentCL.")
    parser.add_argument("--de_noise", action="store_true", help="whether to de-false negative pairs during learning.")
    parser.add_argument("--model_name", default="ELCRec", type=str)
    parser.add_argument("--hidden_size", type=int, default=128, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.2, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.35, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--max_seq_length", default=50, type=int)
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=400, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--cf_weight", type=float, default=0.1, help="weight of contrastive learning task")
    parser.add_argument("--rec_weight", type=float, default=1.0, help="weight of contrastive learning task")
    parser.add_argument("--intent_cf_weight", type=float, default=0.3, help="weight of contrastive learning task")
    parser.add_argument("--deno_weight", type=float, default=0.0001)
    parser.add_argument("--expl_weight", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--shorten_seq_to", type=int, default=None)
    parser.add_argument("--prototype", type=str, default="concat", help="fusion method for intent, 'shift' or 'concat'")
    parser.add_argument("--monitor_clustering", action="store_true", help="Enable clustering visualization during training.")
    parser.add_argument("--monitor_freq", type=int, default=50, help="Frequency (in epochs) to perform clustering visualization.")
    parser.add_argument("--trade_off", type=float, default=1.0, help="trade_off for clustering loss (alpha in paper)"
    parser.add_argument("--enable_diffusion_aug", action="store_true", help="Enable fixed-depth diffusion augmentation.")
    parser.add_argument("--diffusion_aug_rate", type=float, default=0.5, help="Weight for the recommendation loss on diffused sequences.")
    parser.add_argument("--diffusion_fixed_depth", type=int, default=5, help="Fixed diffusion depth T for augmentation.")
    parser.add_argument("--diffusion_t_max", type=int, default=50, help="Maximum steps in the diffusion noise schedule.")
    parser.add_argument("--dual_expert", action="store_true", help="Enable dual expert system.")
    parser.add_argument("--enable_adaptive_diffusion", action="store_true", help="Enable dual expert system.")


    args = parser.parse_args()



    set_seed(args.seed)
    check_path(args.output_dir)
    if args.shorten_seq_to is not None and args.shorten_seq_to > 0:
        
        args.max_seq_length = args.shorten_seq_to + 3 
        print(f"--- [INFO] Dynamically adjusting max_seq_length to: {args.max_seq_length} ---")
    
    args_str = f"{args.model_name}-{args.data_name}-{args.model_idx}-jianduan555"
    args.log_file = os.path.join(args.output_dir, args_str + ".txt")
    checkpoint_prefix = os.path.join(args.output_dir, args_str)

    if args.data_name in ["Toys_and_Games", "Sports_and_Outdoors"]:
        args.num_intent_clusters = "512"
    elif args.data_name in ["Yelp", "Beauty", "ml-1m"]:
        args.num_intent_clusters = "256"

    if args.data_name in ["Toys_and_Games", "Sports_and_Outdoors"]:
        args.trade_off = 1
    elif args.data_name in ["Yelp"]:
        args.trade_off = 0.1
    elif args.data_name in ["Beauty"]:
        args.trade_off = 10
        
    if args.data_name in ["Beauty"]:
        args.prototype = "shift"
    else:
        args.prototype = "concat"
        
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    print("Using Cuda:", torch.cuda.is_available())
    args.data_file = args.data_dir + args.data_name + ".txt"

    user_seq, max_item, valid_rating_matrix, test_rating_matrix = get_user_seqs(args.data_file)

    args.item_size = max_item + 2
    args.mask_id = max_item + 1

    with open(args.log_file, "a") as f:
        f.write(str(args) + "\n")

    args.train_matrix = valid_rating_matrix

    checkpoint = args_str 
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    cluster_dataset = RecWithContrastiveLearningDataset(
        args, user_seq[: int(len(user_seq) * args.training_data_ratio)], data_type="train"
    )
    cluster_sampler = SequentialSampler(cluster_dataset)
    cluster_dataloader = DataLoader(cluster_dataset, sampler=cluster_sampler, batch_size=args.batch_size)

    train_dataset = RecWithContrastiveLearningDataset(
        args, user_seq[: int(len(user_seq) * args.training_data_ratio)], data_type="train"
    )
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = RecWithContrastiveLearningDataset(args, user_seq, data_type="valid")
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)
    
    test_dataset = RecWithContrastiveLearningDataset(args, user_seq, data_type="test")
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    if args.dual_expert:
        print("Initializing Dual Expert System...")
        levels_e1 = [4,5]  
        levels_e2 = [9,10] 
        model_e5 = SASRecModel(args=args, diffusion_levels= levels_e1)
        model_e10 = SASRecModel(args=args, diffusion_levels= levels_e2)

        early_stopper_e5 = EarlyStopping(
            checkpoint_prefix + "_e5.pt",
            patience=40, verbose=True
        )
        early_stopper_e10 = EarlyStopping(
            checkpoint_prefix + "_e10.pt",
            patience=40, verbose=True
        )
        trainer = DualExpertTrainer(
            model_e5, model_e10, 
            early_stopper_e5, early_stopper_e10,
            train_dataloader, cluster_dataloader, 
            eval_dataloader, test_dataloader, args
        )
    else:
        print("Initializing Single Expert System...")
        model = SASRecModel(args=args)
        trainer = ELCRecTrainer(model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args)

    if args.wandb:
        import wandb
        wandb.init(config=args,
                   project="rec",
                   dir="./wandb/",
                   name="best_cluster_cluster_{}-{}".format(args.num_intent_clusters, args.data_name),
                   job_type="training",
                   reinit=True)
        
    if args.do_eval:
        trainer.model_e5.load_state_dict(torch.load(checkpoint_prefix ))
        trainer.model_e10.load_state_dict(torch.load(checkpoint_prefix ))
        scores, result_info = trainer.iteration(0, trainer.test_dataloader, full_sort=True, train=False, expert_to_eval='fused')

    else:
        # early_stopping = EarlyStopping(args.checkpoint_path, patience=400, verbose=True)
        for epoch in tqdm(range(args.epochs)):
            trainer.train_epoch(epoch)
            all_stopped = trainer.valid_epoch(epoch)
            if all_stopped:
                break
        trainer.args.train_matrix = test_rating_matrix
        trainer.model_e5.load_state_dict(torch.load(checkpoint_prefix + "_e5.pt"))
        trainer.model_e10.load_state_dict(torch.load(checkpoint_prefix + "_e10.pt"))
        scores, result_info = trainer.iteration(0, trainer.test_dataloader, full_sort=True, train=False, expert_to_eval='fused')
   
    print(args_str)
    print(result_info)
    with open(args.log_file, "a") as f:
        f.write(args_str + '5'+"\n")
        f.write(result_info + '5'+"\n")

main()


import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from models import KMeans
from torch.optim import Adam
import torch.nn.functional as F
from modules import NCELoss, PCLoss
from utils import recall_at_k, ndcg_k, nCr,  check_path
import os
from diffusion_utils import get_noise_schedule, forward_diffusion
import time
class Trainer:
    def __init__(self, model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        self.model = model
        self.num_intent_clusters = [int(i) for i in self.args.num_intent_clusters.split(",")]
        
        self.clusters = []
        for num_intent_cluster in self.num_intent_clusters:
            if self.args.seq_representation_type == "mean":
                cluster = KMeans(
                    num_cluster=num_intent_cluster,
                    seed=self.args.seed,
                    hidden_size=self.args.hidden_size,
                    gpu_id=self.args.gpu_id,
                    device=self.device,
                )
                self.clusters.append(cluster)
            else:
                cluster = KMeans(
                    num_cluster=num_intent_cluster,
                    seed=self.args.seed,
                    hidden_size=self.args.hidden_size * self.args.max_seq_length,
                    gpu_id=self.args.gpu_id,
                    device=self.device,
                )
                self.clusters.append(cluster)

        self.total_augmentaion_pairs = nCr(self.args.n_views, 2)
        self.projection = nn.Sequential(
            nn.Linear(self.args.max_seq_length * self.args.hidden_size, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.args.hidden_size, bias=True),
        )
        if self.cuda_condition:
            self.model.cuda()
            self.projection.cuda()
            
        
        self.train_dataloader = train_dataloader
        self.cluster_dataloader = cluster_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)
        
        self.cf_criterion = NCELoss(self.args.temperature, self.device)
        self.pcl_criterion = PCLoss(self.args.temperature, self.device)


    def train(self, epoch):
        return self.iteration(epoch, self.train_dataloader, self.cluster_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort=full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort=full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError
    
    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": "{:.4f}".format(recall[0]),
            "NDCG@5": "{:.4f}".format(ndcg[0]),
            "HIT@20": "{:.4f}".format(recall[1]),
            "NDCG@20": "{:.4f}".format(ndcg[1]),
        }
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        return [recall[0], ndcg[0], recall[1], ndcg[1]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def cross_entropy(self, seq_out, pos_ids, neg_ids, model):

        pos_emb = model.item_embeddings(pos_ids)
        neg_emb = model.item_embeddings(neg_ids)

        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size) 
        pos_logits = torch.sum(pos * seq_emb, -1) 
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * model.args.max_seq_length).float() 
        loss = torch.sum(
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
            - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss
    
    def predict_sample(self, seq_out, test_neg_sample, model): 
        test_item_emb = model.item_embeddings(test_neg_sample) 
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)
        return test_logits

    def predict_full(self, seq_out, model):
        test_item_emb = model.item_embeddings.weight 
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


class ELCRecTrainer(Trainer):
    def __init__(self, model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args):
        super(ELCRecTrainer, self).__init__(
            model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args
        )
        if self.args.enable_diffusion_aug:
            self.alphas_cumprod = get_noise_schedule(self.args.diffusion_t_max, device=self.device)
    
    def _instance_cl_one_pair_contrastive_learning(self, inputs, intent_ids=None):
        cl_batch = torch.cat(inputs, dim=0)
        cl_batch = cl_batch.to(self.device)
        cl_sequence_output = self.model(cl_batch)

        bz = cl_sequence_output.shape[0]
        seq = cl_sequence_output.shape[1]
        clu_num = self.model.cluster_centers.shape[0]
        xx = (cl_sequence_output * cl_sequence_output).sum(-1).reshape(bz, seq, 1).repeat(1, 1, clu_num)
        cc = (self.model.cluster_centers * self.model.cluster_centers).sum(-1).reshape(1, 1, clu_num).repeat(bz, seq, 1)
        xc = torch.matmul(cl_sequence_output, self.model.cluster_centers.T)
        dis = xx + cc - 2 * xc
        index = torch.argmin(dis, dim=-1)
        shift = self.model.cluster_centers[index]

        if self.args.prototype == "shift":
            cl_sequence_output += shift

        elif self.args.prototype == "concat":
            cl_sequence_output = torch.concat([cl_sequence_output, shift], dim=-1)

        if self.args.seq_representation_instancecl_type == "mean":
            cl_sequence_output = torch.mean(cl_sequence_output, dim=1, keepdim=False)
        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
        batch_size = cl_batch.shape[0] // 2
        cl_output_slice = torch.split(cl_sequence_flatten, batch_size)

        if self.args.de_noise:
            cl_loss = self.cf_criterion(cl_output_slice[0], cl_output_slice[1], intent_ids=intent_ids)
        else:
            cl_loss = self.cf_criterion(cl_output_slice[0], cl_output_slice[1], intent_ids=None)
        return cl_loss

    @ staticmethod
    def distance(x, c):
        xx = (x * x).sum(-1).reshape(-1, 1).repeat(1, c.shape[0])
        cc = (c * c).sum(-1).reshape(1, -1).repeat(x.shape[0], 1)
        xx_cc = xx + cc
        xc = x @ c.T
        dis = xx_cc - 2 * xc
        return dis

    def _pcl_one_pair_contrastive_learning(self, inputs, intents, intent_ids):
        n_views, (bsz, seq_len) = len(inputs), inputs[0].shape
        cl_batch = torch.cat(inputs, dim=0)
        cl_batch = cl_batch.to(self.device)
        cl_sequence_output = self.model(cl_batch)

        if self.args.seq_representation_type == "mean":
            cl_sequence_output = torch.mean(cl_sequence_output, dim=1, keepdim=False)
        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
        cl_output_slice = torch.split(cl_sequence_flatten, bsz)

        if self.args.de_noise:
            cl_loss = self.pcl_criterion(cl_output_slice[0], cl_output_slice[1], intents=intents, intent_ids=intent_ids)
        else:
            cl_loss = self.pcl_criterion(cl_output_slice[0], cl_output_slice[1], intents=intents, intent_ids=None)
        return cl_loss
    

    def iteration(self, epoch, dataloader, cluster_dataloader=None, full_sort=True, train=True):

        if train:
            if self.args.contrast_type in ["IntentCL", "Hybrid"] and epoch >= self.args.warm_up_epoches:
                if epoch==0:
                    self.model.eval()
                    kmeans_training_data = []
                    rec_cf_data_iter = enumerate(cluster_dataloader)
                    for i, (rec_batch, _, _) in rec_cf_data_iter:
                        rec_batch = tuple(t.to(self.device) for t in rec_batch)
                        _, input_ids, target_pos, target_neg, _ = rec_batch
                        sequence_output = self.model(input_ids)
                        if self.args.seq_representation_type == "mean":
                            sequence_output = torch.mean(sequence_output, dim=1, keepdim=False)
                        sequence_output = sequence_output.view(sequence_output.shape[0], -1)
                        sequence_output = sequence_output.detach().cpu().numpy()
                        kmeans_training_data.append(sequence_output)
                    kmeans_training_data = np.concatenate(kmeans_training_data, axis=0)

                    for i, cluster in enumerate(self.clusters):

                        cluster.train(kmeans_training_data)
                        self.model.cluster_centers.data = cluster.centroids

                    import gc

                    gc.collect()

            self.model.train()
            rec_avg_loss = 0.0
            cl_sum_avg_loss = 0.0
            joint_avg_loss = 0.0

            rec_cf_data_iter = enumerate(dataloader)

            for i, (rec_batch, cl_batches, seq_class_label_batches) in rec_cf_data_iter:
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, input_ids, target_pos, target_neg, _ = rec_batch

                sequence_output = self.model(input_ids)
                rec_loss = self.cross_entropy(sequence_output, target_pos, target_neg)
                diffusion_rec_loss = 0.0
                if self.args.enable_diffusion_aug:
                    with torch.no_grad(): 
                        original_sequence_emb = self.model.add_position_embedding(input_ids).detach()

                    t = torch.full(
                        (input_ids.shape[0],), 
                        self.args.diffusion_fixed_depth - 1,
                        device=self.device, 
                        dtype=torch.long
                    )
                    
                    diffused_sequence_emb = forward_diffusion(original_sequence_emb, t, self.alphas_cumprod)
                    encoded_layers = self.model.item_encoder(diffused_sequence_emb, self.model.get_attention_mask(input_ids))
                    diffused_sequence_output = encoded_layers[-1]
                    diffusion_rec_loss = self.cross_entropy(diffused_sequence_output, target_pos, target_neg)

                total_rec_loss = rec_loss + self.args.diffusion_aug_rate * diffusion_rec_loss
                cl_losses = []
                sample_distance_losses = []
                for cl_batch in cl_batches:
                    if self.args.contrast_type == "InstanceCL":
                        cl_loss = self._instance_cl_one_pair_contrastive_learning(
                            cl_batch, intent_ids=seq_class_label_batches
                        )
                        cl_losses.append(self.args.cf_weight * cl_loss)
                    elif self.args.contrast_type == "IntentCL":
                        if epoch >= self.args.warm_up_epoches:
                            if self.args.seq_representation_type == "mean":
                                sequence_output = torch.mean(sequence_output, dim=1, keepdim=False)
                            sequence_output = sequence_output.view(sequence_output.shape[0], -1)
                            sequence_output = sequence_output.detach().cpu().numpy()

                            for cluster in self.model.cluster_centers:
                                seq2intents = []
                                intent_ids = []
                                intent_id, seq2intent = cluster.query(sequence_output)
                                seq2intents.append(seq2intent)
                                intent_ids.append(intent_id)
                            cl_loss = self._pcl_one_pair_contrastive_learning(
                                cl_batch, intents=seq2intents, intent_ids=intent_ids
                            )
                            cl_losses.append(self.args.intent_cf_weight * cl_loss)
                        else:
                            continue
                    elif self.args.contrast_type == "Hybrid":
                        if epoch < self.args.warm_up_epoches:
                            cl_loss1 = self._instance_cl_one_pair_contrastive_learning(
                                cl_batch, intent_ids=seq_class_label_batches
                            )
                            cl_losses.append(self.args.cf_weight * cl_loss1)

                        else:

                            cl_loss1 = self._instance_cl_one_pair_contrastive_learning(
                                cl_batch, intent_ids=seq_class_label_batches
                            )

                            cl_losses.append(self.args.cf_weight * cl_loss1)

                            if self.args.seq_representation_type == "mean":
                                sequence_output = torch.mean(sequence_output, dim=1, keepdim=False)
                            sequence_output = sequence_output.view(sequence_output.shape[0], -1)
                            sequence_output = F.normalize(sequence_output, p=2, dim=1)

                            seq2intents = []
                            intent_ids = []

                            self.model.cluster_centers.data = F.normalize(self.model.cluster_centers.data, p=2, dim=1)
                            center = self.model.cluster_centers

                            sample_center_distance = self.distance(sequence_output, center)
                            index = torch.argmin(sample_center_distance, dim=-1)
                            sample_distance_loss = sample_center_distance.mean()

                            center_center_distance = self.distance(center, center)
                            center_center_distance.flatten()[:-1].view(center.shape[0] - 1, center.shape[0] + 1)[:, 1:].flatten()
                            center_distance_loss = -center_center_distance.mean()

                            sample_distance_losses.append(self.args.trade_off*(sample_distance_loss+center_distance_loss))

                            seq2intent = self.model.cluster_centers[index]

                            intent_ids.append(index)
                            seq2intents.append(seq2intent)

                            cl_loss3 = self._pcl_one_pair_contrastive_learning(
                                cl_batch, intents=seq2intents, intent_ids=intent_ids
                            )

                            cl_losses.append(self.args.intent_cf_weight * cl_loss3)

                # joint_loss = self.args.rec_weight * rec_loss
                joint_loss = self.args.rec_weight * total_rec_loss
                for cl_loss in cl_losses:
                    joint_loss += cl_loss

                for dis_loss in sample_distance_losses:
                    joint_loss += dis_loss


                self.optim.zero_grad()
                joint_loss.backward(retain_graph=True)
                self.optim.step()

                rec_avg_loss += rec_loss.item()

                for i, cl_loss in enumerate(cl_losses):
                    cl_sum_avg_loss += cl_loss.item()


                joint_avg_loss += joint_loss.item()

            
            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": "{:.4f}".format(rec_avg_loss / len(dataloader)),
                "joint_avg_loss": "{:.4f}".format(joint_avg_loss / len(dataloader)),
            }

            with open(self.args.log_file, "a") as f:
                f.write(str(post_fix) + "\n")

            return rec_avg_loss / len(dataloader), joint_avg_loss / len(dataloader)

        else:
            rec_data_iter = enumerate(dataloader)
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:

                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    recommend_output = self.model(input_ids)

                    recommend_output = recommend_output[:, -1, :]

                    rating_pred = self.predict_full(recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0

                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                for i, batch in rec_data_iter:
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    recommend_output = self.model.finetune(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)

                return self.get_sample_scores(epoch, pred_list)




class DualExpertTrainer(ELCRecTrainer):
    def __init__(self, model_e5, model_e10, early_stopper_e5, early_stopper_e10, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args):

        super(DualExpertTrainer, self).__init__(
            model_e5, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args
        )
        
        print("Initializing Dual Expert Trainer...")
        self.model_e5 = model_e5
        self.model_e10 = model_e10
        self.early_stopper_e5 = early_stopper_e5
        self.early_stopper_e10 = early_stopper_e10
        self.e5_stopped = False
        self.e10_stopped = False

        if self.cuda_condition:
            self.model_e5.cuda()
            self.model_e10.cuda()

        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim_e5 = Adam(self.model_e5.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)
        self.optim_e10 = Adam(self.model_e10.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)
        self.diffusion_depth_e5 = 5
        self.diffusion_depth_e10 = 5
        self.alphas_cumprod = get_noise_schedule(self.args.diffusion_t_max, device=self.device)


    def _train_one_expert(self, model, optimizer, rec_batch, cl_batches, seq_class_label_batches, is_coarse_expert=False, fixed_diffusion_depth=None):

        *rec_data, sequence_len = rec_batch
        _, input_ids, target_pos, target_neg, _ = rec_data
        is_adaptive = hasattr(model, 'controller') and model.controller is not None
        sequence_output_for_cl = None
        total_rec_loss = 0.0
        probs = None

        if is_adaptive:
            raw_sequence_output, adaptive_sequence_output , probs= model(input_ids, sequence_lengths=sequence_len)
            raw_rec_loss = self.cross_entropy(raw_sequence_output, target_pos, target_neg, model)
            adaptive_rec_loss = self.cross_entropy(adaptive_sequence_output, target_pos, target_neg, model)
            total_rec_loss = raw_rec_loss + self.args.diffusion_aug_rate * adaptive_rec_loss
            sequence_output_for_cl = raw_sequence_output

            # L_deno
            reconstruction_loss = 0.0
            target_representation = raw_sequence_output.detach()
            reconstructed_representation = adaptive_sequence_output
            input_mask = (input_ids > 0).unsqueeze(-1).float()
            reconstruction_loss = F.mse_loss(
                reconstructed_representation * input_mask, 
                target_representation * input_mask,
                reduction='sum'
            ) / (input_mask.sum() + 1e-8)

        else:
            if fixed_diffusion_depth is None:
                raise ValueError("fixed_diffusion_depth must be provided for a non-adaptive expert.")
            sequence_output = model(input_ids)
            raw_rec_loss = self.cross_entropy(sequence_output, target_pos, target_neg, model)
            with torch.no_grad():
                original_sequence_emb = model.add_position_embedding(input_ids).detach()

            t = torch.full((input_ids.shape[0],), fixed_diffusion_depth - 1, device=self.device, dtype=torch.long)
            diffused_sequence_emb = forward_diffusion(original_sequence_emb, t, self.alphas_cumprod)
            
            encoded_layers = model.item_encoder(diffused_sequence_emb, model.get_attention_mask(input_ids))
            diffused_sequence_output = encoded_layers[-1]
            
            fixed_diffusion_loss = self.cross_entropy(diffused_sequence_output, target_pos, target_neg, model)
            total_rec_loss = raw_rec_loss + self.args.diffusion_aug_rate * fixed_diffusion_loss
            sequence_output_for_cl = sequence_output
            reconstruction_loss = 0.0 

        # L_expl
        exploration_loss = 0.0
        if probs is not None:
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            exploration_loss = -torch.mean(entropy)

        cl_losses = []
        sample_distance_losses = []

        if self.args.contrast_type == "Hybrid": 
            for cl_batch in cl_batches:
                cl_inputs_inst = torch.cat(cl_batch, dim=0) 
                cl_inputs_inst_lengths = torch.sum(cl_inputs_inst > 0, dim=1)

                _, cl_sequence_output_inst, _ = model(cl_inputs_inst, sequence_lengths=cl_inputs_inst_lengths)
                
                if self.args.seq_representation_instancecl_type == "mean":
                    cl_sequence_output_inst = torch.mean(cl_sequence_output_inst, dim=1, keepdim=False)
                
                cl_sequence_flatten_inst = cl_sequence_output_inst.view(cl_inputs_inst.shape[0], -1)
                batch_size_inst = cl_inputs_inst.shape[0] // 2
                cl_output_slice_inst = torch.split(cl_sequence_flatten_inst, batch_size_inst)
                intent_ids_inst = seq_class_label_batches if self.args.de_noise else None
                cl_loss1 = self.cf_criterion(cl_output_slice_inst[0], cl_output_slice_inst[1], intent_ids=intent_ids_inst)
                cl_losses.append(self.args.cf_weight * cl_loss1)
                if is_coarse_expert:
                    if self.args.seq_representation_type == "mean":
                        intent_query_output = torch.mean(sequence_output_for_cl.detach(), dim=1, keepdim=False)
                    else:
                        intent_query_output = sequence_output_for_cl.detach()

                    intent_query_output = intent_query_output.view(intent_query_output.shape[0], -1)
                    intent_query_output = F.normalize(intent_query_output, p=2, dim=1)

                    seq2intents = []
                    intent_ids_list = [] 
                    
                    model.cluster_centers.data = F.normalize(model.cluster_centers.data, p=2, dim=1)
                    center = model.cluster_centers

                    sample_center_distance = self.distance(intent_query_output, center)
                    index = torch.argmin(sample_center_distance, dim=-1)
                    sample_distance_loss = sample_center_distance.mean()
                    center_center_distance = self.distance(center, center)
                    center_distance_loss = -center_center_distance.mean()
                    sample_distance_losses.append(self.args.trade_off * (sample_distance_loss + center_distance_loss))
                    
                    seq2intent = model.cluster_centers[index]
                    intent_ids_list.append(index)
                    seq2intents.append(seq2intent)

                    cl_inputs_pcl = torch.cat(cl_batch, dim=0)
                    cl_inputs_pcl_lengths = torch.sum(cl_inputs_pcl > 0, dim=1)
                    
                    _, cl_sequence_output_pcl,_ = model(cl_inputs_pcl, sequence_lengths=cl_inputs_pcl_lengths)
                    
                    if self.args.seq_representation_type == "mean":
                        cl_sequence_output_pcl = torch.mean(cl_sequence_output_pcl, dim=1, keepdim=False)

                    cl_sequence_flatten_pcl = cl_sequence_output_pcl.view(cl_inputs_pcl.shape[0], -1)
                    cl_output_slice_pcl = torch.split(cl_sequence_flatten_pcl, batch_size_inst)
                    
                    cl_loss3 = self.pcl_criterion(cl_output_slice_pcl[0], cl_output_slice_pcl[1], intents=seq2intents, intent_ids=intent_ids_list)
                    cl_losses.append(self.args.intent_cf_weight * cl_loss3)

        joint_loss = self.args.rec_weight * total_rec_loss
        joint_loss += self.args.deno_weight * reconstruction_loss
        joint_loss += self.args.expl_weight * exploration_loss
        for cl_loss in cl_losses:
            joint_loss += cl_loss
        for dis_loss in sample_distance_losses:
            joint_loss += dis_loss
        
        optimizer.zero_grad()
        joint_loss.backward()
        optimizer.step()
        
        rec_loss_to_return = total_rec_loss.item() if isinstance(total_rec_loss, torch.Tensor) else total_rec_loss
        return rec_loss_to_return, joint_loss.item()
   
    def train_epoch(self, epoch):

        if self.cuda_condition:
            torch.cuda.synchronize()
        start_time = time.time()
        self.model_e5.train()
        self.model_e10.train()
        
        rec_avg_loss_e5, joint_avg_loss_e5 = 0.0, 0.0
        rec_avg_loss_e10, joint_avg_loss_e10 = 0.0, 0.0

        rec_cf_data_iter = enumerate(self.train_dataloader)
        
        for i, (rec_batch, cl_batches, seq_class_label_batches) in rec_cf_data_iter:
            rec_batch = tuple(t.to(self.device) for t in rec_batch)
            seq_class_label_batches = seq_class_label_batches.to(self.device)
            processed_cl_batches = [[tensor.to(self.device) for tensor in pair] for pair in cl_batches]

            if not self.e5_stopped:
                rec_loss, joint_loss = self._train_one_expert(
                    self.model_e5, self.optim_e5, 
                    rec_batch, processed_cl_batches, seq_class_label_batches,
                    is_coarse_expert=True,
                    fixed_diffusion_depth=None
                )
                rec_avg_loss_e5 += rec_loss
                joint_avg_loss_e5 += joint_loss
            
            if not self.e10_stopped:
                rec_loss, joint_loss = self._train_one_expert(
                    self.model_e10, self.optim_e10,
                    rec_batch, processed_cl_batches, seq_class_label_batches,
                    is_coarse_expert=True,
                    fixed_diffusion_depth=None
                )
                rec_avg_loss_e10 += rec_loss
                joint_avg_loss_e10 += joint_loss

        if self.cuda_condition:
            torch.cuda.synchronize()
        end_time = time.time()
        epoch_duration = end_time - start_time
        post_fix = { "epoch": epoch }
        if not self.e5_stopped:
            post_fix["rec_loss_e5"] = f"{rec_avg_loss_e5 / len(self.train_dataloader):.4f}"
        if not self.e10_stopped:
            post_fix["rec_loss_e10"] = f"{rec_avg_loss_e10 / len(self.train_dataloader):.4f}"
        post_fix["train_duration_sec"] = f"{epoch_duration:.2f}"
        
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
    
    def valid_epoch(self, epoch):

        self.model_e5.eval()
        self.model_e10.eval()

        if not self.e5_stopped:

            scores_e5, _ = self.iteration(epoch, self.eval_dataloader, full_sort=True, train=False, expert_to_eval='e5')
            self.early_stopper_e5(np.array(scores_e5[-1:]), self.model_e5)
            if self.early_stopper_e5.early_stop:
                self.e5_stopped = True

        if not self.e10_stopped:
            scores_e10, _ = self.iteration(epoch, self.eval_dataloader, full_sort=True, train=False, expert_to_eval='e10')
            self.early_stopper_e10(np.array(scores_e10[-1:]), self.model_e10)
            if self.early_stopper_e10.early_stop:
                self.e10_stopped = True

        fused_scores, _ = self.iteration(epoch, self.eval_dataloader, full_sort=True, train=False, expert_to_eval='fused')

        if self.e5_stopped and self.e10_stopped:
            return True 
        return False
    
    def iteration(self, epoch, dataloader, cluster_dataloader=None, full_sort=True, train=True, expert_to_eval = None):
        if train:
            raise NotImplementedError("Training logic is now in train_epoch()")

        else: 
            if expert_to_eval is None:
                raise ValueError("expert_to_eval must be specified in eval mode: 'e5', 'e10', or 'fused'.")
            self.model_e5.eval()
            self.model_e10.eval()
            
            pred_list = None
            answer_list = None
            
            rec_data_iter = dataloader
            for i, batch in enumerate(rec_data_iter):
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, target_pos, target_neg, answers,_ = batch
                
                with torch.no_grad():
                    rating_pred = None
                    if expert_to_eval == 'e5':
                        raw_output, _ ,_= self.model_e5(input_ids) 
                        output = raw_output[:, -1, :] 
                        rating_pred = self.predict_full(output, self.model_e5)
                    elif expert_to_eval == 'e10':
                        raw_output, _,_ = self.model_e10(input_ids)
                        output = raw_output[:, -1, :]
                        rating_pred = self.predict_full(output, self.model_e10)
                    elif expert_to_eval == 'fused':
                        raw_output_e5, _ ,_= self.model_e5(input_ids)
                        raw_output_e10, _ ,_= self.model_e10(input_ids)
                        
                        output_e5 = raw_output_e5[:, -1, :]
                        output_e10 = raw_output_e10[:, -1, :]
                        rating_pred_e5 = self.predict_full(output_e5, self.model_e5)
                        rating_pred_e10 = self.predict_full(output_e10, self.model_e10)
                        rating_pred = (rating_pred_e5 + rating_pred_e10) / 2.0
                
                rating_pred = rating_pred.cpu().data.numpy().copy()
                batch_user_index = user_ids.cpu().numpy()
                rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                
                ind = np.argpartition(rating_pred, -20)[:, -20:]
                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                if i == 0:
                    pred_list = batch_pred_list
                    answer_list = answers.cpu().data.numpy()
                else:
                    pred_list = np.append(pred_list, batch_pred_list, axis=0)
                    answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
            
            return self.get_full_sort_score(epoch, answer_list, pred_list)

    def save(self, file_name_prefix):

        path_e5 = file_name_prefix + "_e5.pt"
        path_e10 = file_name_prefix + "_e10.pt"
        
        torch.save(self.model_e5.cpu().state_dict(), path_e5)
        torch.save(self.model_e10.cpu().state_dict(), path_e10)
        
        self.model_e5.to(self.device)
        self.model_e10.to(self.device)

    def load(self, file_name_prefix):
        path_e5 = file_name_prefix + "_e5.pt"
        path_e10 = file_name_prefix + "_e10.pt"
        self.model_e5.load_state_dict(torch.load(path_e5, map_location=self.device))
        self.model_e10.load_state_dict(torch.load(path_e10, map_location=self.device))
        

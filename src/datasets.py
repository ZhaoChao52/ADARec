import copy
import torch
import random
from utils import neg_sample, nCr
from torch.utils.data import Dataset
from data_augmentation import Crop, Mask, Reorder, Random
from diffusion_utils import get_noise_schedule, forward_diffusion 

class RecWithContrastiveLearningDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type="train"):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

        if self.args.shorten_seq_to is not None:
            print(f"Shortening all user sequences to a max length of {self.args.shorten_seq_to}...")
            processed_user_seq = []

            MIN_SEQ_LEN_FOR_TRAIN = 4 
            MIN_SEQ_LEN_FOR_VALID = 3
            MIN_SEQ_LEN_FOR_TEST = 2
            for seq in self.user_seq:
                
                if data_type == "train":
                    if len(seq) < MIN_SEQ_LEN_FOR_TRAIN:
                        continue 
                    special_tokens = seq[-3:]
                    main_seq = seq[:-3]
                    main_seq = main_seq[-self.args.shorten_seq_to:]
                    processed_user_seq.append(main_seq + special_tokens)
                elif data_type == "valid":
                    if len(seq) < MIN_SEQ_LEN_FOR_VALID:
                        continue
                    special_tokens = seq[-2:]
                    main_seq = seq[:-2]
                    main_seq = main_seq[-self.args.shorten_seq_to:]
                    processed_user_seq.append(main_seq + special_tokens)
                else: # test
                    if len(seq) < MIN_SEQ_LEN_FOR_TEST:
                        continue
                    special_tokens = seq[-1:]
                    main_seq = seq[:-1]
                    main_seq = main_seq[-self.args.shorten_seq_to:]
                    processed_user_seq.append(main_seq + special_tokens)
            
            self.user_seq = processed_user_seq 
            print("Sequence shortening complete.")
        
        self.augmentations = {
            "crop": Crop(tao=args.tao),
            "mask": Mask(gamma=args.gamma),
            "reorder": Reorder(beta=args.beta),
            "random": Random(tao=args.tao, gamma=args.gamma, beta=args.beta),
        }
        if self.args.augment_type not in self.augmentations:
            raise ValueError(f"augmentation type: '{self.args.augment_type}' is invalided")
        print(f"Creating Contrastive Learning Dataset using '{self.args.augment_type}' data augmentation")
        self.base_transform = self.augmentations[self.args.augment_type]
        
        self.n_views = self.args.n_views
    
    def _get_all_augmentations(self, input_ids):
        
        views = []
        for _ in range(self.n_views): 
            augmented_input_ids = self.base_transform(input_ids)
            pad_len = self.max_len - len(augmented_input_ids)
            augmented_input_ids = [0] * pad_len + augmented_input_ids
            augmented_input_ids = augmented_input_ids[-self.max_len :]
            views.append(torch.tensor(augmented_input_ids, dtype=torch.long))
        
        if self.do_diffusion_aug:
            
            raw_input_ids = input_ids[:] 
            pad_len = self.max_len - len(raw_input_ids)
            raw_input_ids = [0] * pad_len + raw_input_ids
            raw_input_ids = raw_input_ids[-self.max_len :]
            views.append(torch.tensor(raw_input_ids, dtype=torch.long))

        return views
    
    def _one_pair_data_augmentation(self, input_ids):
        augmented_seqs = []
        for i in range(2):
            augmented_input_ids = self.base_transform(input_ids)

            pad_len = self.max_len - len(augmented_input_ids)
            augmented_input_ids = [0] * pad_len + augmented_input_ids
            augmented_input_ids = augmented_input_ids[-self.max_len :]

            assert len(augmented_input_ids) == self.max_len

            cur_tensors = torch.tensor(augmented_input_ids, dtype=torch.long)
            augmented_seqs.append(cur_tensors)
        return augmented_seqs

    def _process_sequence_label_signal(self, seq_label_signal):
        seq_class_label = torch.tensor(seq_label_signal, dtype=torch.long)
        return seq_class_label

    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, answer):
        copied_input_ids = copy.deepcopy(input_ids)
        target_neg = []
        seq_set = set(items)
        for _ in copied_input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(copied_input_ids)
        copied_input_ids = [0] * pad_len + copied_input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        copied_input_ids = copied_input_ids[-self.max_len :]
        target_pos = target_pos[-self.max_len :]
        target_neg = target_neg[-self.max_len :]

        assert len(copied_input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_rec_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(copied_input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_rec_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(copied_input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
            )

        return cur_rec_tensors

    def _add_noise_interactions(self, items):
        copied_sequence = copy.deepcopy(items)
        insert_nums = max(int(self.args.noise_ratio * len(copied_sequence)), 0)
        if insert_nums == 0:
            return copied_sequence
        insert_idx = random.choices([i for i in range(len(copied_sequence))], k=insert_nums)
        inserted_sequence = []
        for index, item in enumerate(copied_sequence):
            if index in insert_idx:
                item_id = random.randint(1, self.args.item_size - 2)
                while item_id in copied_sequence:
                    item_id = random.randint(1, self.args.item_size - 2)
                inserted_sequence += [item_id]
            inserted_sequence += [item]
        return inserted_sequence

    def __getitem__(self, index):
        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            seq_label_signal = items[-2]
            answer = [0]  
        elif self.data_type == "valid":
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        else:
            items_with_noise = self._add_noise_interactions(items)
            input_ids = items_with_noise[:-1]
            target_pos = items_with_noise[1:]
            answer = [items_with_noise[-1]]
        
        sequence_len = len(input_ids)
        if self.data_type == "train":
            cur_rec_tensors = self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)
            cur_rec_tensors = cur_rec_tensors + (torch.tensor(sequence_len, dtype=torch.long),)
            cf_tensors_list = []
            total_augmentaion_pairs = nCr(self.n_views, 2)

            for i in range(total_augmentaion_pairs):
                cf_tensors_list.append(self._one_pair_data_augmentation(input_ids))

            seq_class_label = self._process_sequence_label_signal(seq_label_signal)

            return (cur_rec_tensors, cf_tensors_list, seq_class_label)
        elif self.data_type == "valid":
            cur_rec_tensors = self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)
            cur_rec_tensors = cur_rec_tensors + (torch.tensor(sequence_len, dtype=torch.long),)
            return cur_rec_tensors
        else:
            cur_rec_tensors = self._data_sample_rec_task(user_id, items_with_noise, input_ids, target_pos, answer)
            cur_rec_tensors = cur_rec_tensors + (torch.tensor(sequence_len, dtype=torch.long),)
            return cur_rec_tensors

    def __len__(self):
        
        return len(self.user_seq)


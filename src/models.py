import torch
import faiss
import torch.nn as nn
from modules import Encoder, LayerNorm, AdaptiveDepthController
from diffusion_utils import get_noise_schedule, forward_diffusion
import torch.nn.functional as F
class KMeans(object):
    def __init__(self, num_cluster, seed, hidden_size, gpu_id=0, device="cpu"):
        """
        Args:
            k: number of clusters
        """
        self.seed = seed
        self.num_cluster = num_cluster
        self.max_points_per_centroid = 4096
        self.min_points_per_centroid = 0
        self.gpu_id = 0
        self.device = device
        self.first_batch = True
        self.hidden_size = hidden_size
        self.clus, self.index = self.__init_cluster(self.hidden_size)
        self.centroids = []

    def __init_cluster(
        self, hidden_size, verbose=False, niter=20, nredo=5, max_points_per_centroid=4096, min_points_per_centroid=0
    ):
        clus = faiss.Clustering(hidden_size, self.num_cluster)
        clus.verbose = verbose
        clus.niter = niter
        clus.nredo = nredo
        clus.seed = self.seed
        clus.max_points_per_centroid = max_points_per_centroid
        clus.min_points_per_centroid = min_points_per_centroid

        index = faiss.IndexFlatL2(hidden_size)
        return clus, index

    def train(self, x):
        if x.shape[0] > self.num_cluster:
            self.clus.train(x, self.index)
        centroids = faiss.vector_to_array(self.clus.centroids).reshape(self.num_cluster, self.hidden_size)
        centroids = torch.tensor(centroids, requires_grad=True).to(self.device)
        self.centroids = nn.functional.normalize(centroids, p=2, dim=1)

    def query(self, x):
        D, I = self.index.search(x, 1)  
        seq2cluster = [int(n[0]) for n in I]
        seq2cluster = torch.LongTensor(seq2cluster).to(self.device)
        return seq2cluster, self.centroids[seq2cluster]


class SASRecModel(nn.Module):
    def __init__(self, args, diffusion_levels=None):
        super(SASRecModel, self).__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)

        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args
        self.cluster_centers = nn.Parameter(torch.Tensor(int(args.num_intent_clusters), int(args.hidden_size)))

        self.criterion = nn.BCELoss(reduction="none")
        self.apply(self.init_weights)

        if args.enable_adaptive_diffusion:
            self.controller = AdaptiveDepthController(args, args.hidden_size, num_levels=2)
            self.diffusion_levels = torch.tensor(diffusion_levels, dtype=torch.long)
            self.alphas_cumprod = get_noise_schedule(args.diffusion_t_max, device=torch.device("cuda" if args.cuda_condition else "cpu"))


    def add_position_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb
    
    def get_attention_mask(self, input_ids): 
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
    
    def forward(self, input_ids, sequence_lengths=None):
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) 
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)


        raw_item_encoded_layers = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)
        raw_sequence_output = raw_item_encoded_layers[-1]

        if self.args.enable_adaptive_diffusion and self.training: 
            if sequence_lengths is None:
                raise ValueError("sequence_lengths must be provided for adaptive diffusion.")
            
            self.diffusion_levels = self.diffusion_levels.to(sequence_emb.device)
            self.alphas_cumprod = self.alphas_cumprod.to(sequence_emb.device)
            probs = self.controller(sequence_emb.detach(), sequence_lengths) 

            weighted_embs = []

            non_zero_depth_indices = torch.where(self.diffusion_levels > 0)[0]

            if len(non_zero_depth_indices) == 0:
                return raw_sequence_output, raw_sequence_output 
                
            probs_for_diffusion = probs[:, non_zero_depth_indices]
            
            probs_for_diffusion_normalized = probs_for_diffusion / (torch.sum(probs_for_diffusion, dim=1, keepdim=True) + 1e-8)
            for i, idx in enumerate(non_zero_depth_indices):
                depth = self.diffusion_levels[idx]
                t = torch.full((input_ids.shape[0],), depth - 1, device=input_ids.device, dtype=torch.long)
                diffused_emb = forward_diffusion(sequence_emb, t, self.alphas_cumprod)
                weight = probs_for_diffusion_normalized[:, i].unsqueeze(1).unsqueeze(2)
                weighted_embs.append(diffused_emb * weight)

            final_sequence_emb = sum(weighted_embs)
            
            adaptive_item_encoded_layers = self.item_encoder(final_sequence_emb, extended_attention_mask, output_all_encoded_layers=True)
            adaptive_sequence_output = adaptive_item_encoded_layers[-1]
            
            return raw_sequence_output, adaptive_sequence_output, probs

        else:
            return raw_sequence_output, raw_sequence_output, None

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


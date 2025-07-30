<div id="top" align="center">

<h2><a>De-collapsing User Intent: Adaptive Diffusion Augmentation with Mixture-of-Experts for Sequential Recommendation</a></h2>


<!-- [Author 1](https://...), [Author 2](https://...), [Author 3](https://...), ... -->

<p align="center">
  <a href="https://pytorch.org/" alt="PyTorch">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white" />
  </a>
</p>

</div>

<p align="justify">
Sequential recommendation (SR) aims to predict users' next action based on their historical behavior, and is widely adopted by a number of platforms. The performance of SR models fundamentally relies on rich interaction data. However, in real-world scenarios, many users only have  a few historical interactions, leading to the problem of data sparsity. Data sparsity not only leads to model overfitting on sparse sequences, but also hinders the model’s ability to capture the underlying hierarchy of user intents. This results in misinterpreting the user's true intents and recommending irrelevant items. Existing data augmentation methods attempt to mitigate overfitting by generating relevant and varied data. However, they overlook the problem of reconstructing the user's intent hierarchy, which is lost in sparse data. Consequently, the augmented data often fails to align with the user's true intents, potentially leading to misguided recommendations. To address this, we propose the Adaptive Diffusion Augmentation for Recommendation (ADARec) framework. Critically, instead of using a diffusion model as a black-box generator, we use its entire step-wise denoising trajectory to reconstruct a user's intent hierarchy from a single sparse sequence. To ensure both efficiency and effectiveness, our framework adaptively determines the required augmentation depth for each sequence and employs a specialized mixture-of-experts architecture to decouple coarse- and fine-grained intents. Experiments show ADARec outperforms state-of-the-art methods by 2–10\% on standard benchmarks and 3-17\% on sparse sequences, demonstrating its ability to reconstruct hierarchical intent representations from sparse data.
</p>

<div align="center">
  <img src="https://i.postimg.cc/tRhsr10S/adarec-framework.png" alt="ADARec Framework" width="100%"/>
  <p>Figure 1. The overall architecture of the proposed ADARec framework.</p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#-usage">Usage</a></li>
    <li><a href="#-results">Results</a></li>
    <li><a href="#-acknowledgement">Acknowledgement</a></li>
  </ol>
</details>

## 🚀 Usage

### Requirements
- python >= 3.10
- torch >= 2.0
- faiss-gpu
- numpy
- tqdm
- wandb

### Start

```
cd ./src
```

To run the full ADARec model on different datasets, you can use the following commands. The scripts will train the model and evaluate it.

**Training on the Sports & Outdoors dataset:**
```bash
python3 main.py --data_name Sports_and_Outdoors --cf_weight 0.1\
    --model_idx 1 --gpu_id 0\
    --output_dir output/Sports_and_Outdoors/ \
    --batch_size 256 --contrast_type Hybrid \
    --num_intent_cluster 256 --seq_representation_type mean \
    --intent_cf_weight 0.1 --num_hidden_layers 2\
    --enable_diffusion_aug --hidden_size 128 \
    --attention_probs_dropout_prob 0.5 --hidden_dropout_prob 0.5 \
    --enable_adaptive_diffusion --dual_expert \
    --epochs 400
```


**Training on the Beauty dataset:**
```bash
python3 main.py --data_name Beauty --cf_weight 0.1 \
    --model_idx 1 --gpu_id 0 \
    --output_dir output/Beauty/ \
    --batch_size 256 --contrast_type Hybrid \
    --num_intent_cluster 256 --seq_representation_type mean \
    --intent_cf_weight 0.1 --num_hidden_layers 1 \
    --enable_diffusion_aug --hidden_size 256 \
    --attention_probs_dropout_prob 0.5 --hidden_dropout_prob 0.5 \
    --enable_adaptive_diffusion --dual_expert \
    --epochs 400
```

**Training on the Toys & Games dataset:**
```bash
python3 main.py --data_name Toys_and_Games --cf_weight 0.1 \
    --model_idx 1 --gpu_id 0 \
    --output_dir output/Toys_and_Games/ \
    --batch_size 256 --contrast_type Hybrid \
    --num_intent_cluster 256 --seq_representation_type mean \
    --intent_cf_weight 0.1 --num_hidden_layers 3 \
    --enable_diffusion_aug --hidden_size 128 \
    --attention_probs_dropout_prob 0.5 --hidden_dropout_prob 0.5 \
    --enable_adaptive_diffusion --dual_expert \
    --epochs 400
```

**Training on the Yelp dataset:**
```bash
python3 main.py --data_name Yelp --cf_weight 0.1 \
    --model_idx 1 --gpu_id 0 \
    --output_dir output/Yelp/ \
    --batch_size 256 --contrast_type Hybrid \
    --num_intent_cluster 256 --seq_representation_type mean \
    --intent_cf_weight 0.1 --num_hidden_layers 2 \
    --enable_diffusion_aug --hidden_size 128 \
    --attention_probs_dropout_prob 0.2 --hidden_dropout_prob 0.35 \
    --enable_adaptive_diffusion --dual_expert \
    --epochs 400
```

**Note:** To evaluate the model's performance on truncated user histories, you can utilize the --shorten_seq_to argument. For instance, append --shorten_seq_to 5 or --shorten_seq_to 3 to the command line to simulate scenarios with shorter sequence lengths.


## 📊 Results

Our proposed ADARec framework demonstrates significant and consistent outperformance across all benchmarks, especially on sparse user sequences.


<div align="center">
  <img src="https://i.postimg.cc/9XYrwLvJ/table1-overall-performance.png" alt="Overall Performance Comparison" width="100%"/>
  <p>Table 1. Overall performance comparison on four benchmark datasets.</p>
</div>

<div align="center">
  <img src="https://i.postimg.cc/9F4wcr55/table2-sparse-performance.png" alt="Performance on Sparse Sequences" width="50%"/>
  <p>Table 2. Performance comparison on extremely sparse sequences (user history length ≤ 5).</p>
</div>


<div align="center">
  <img src="https://i.postimg.cc/cHXbbzDZ/table3-sparse-performance-seq3.png" alt="Performance on Sparse Sequences_3" width="50%"/>
  <p>Table 3. Performance comparison on extremely sparse sequences (user history length ≤ 3).</p>
</div>

## 🙏 Acknowledgement

Our implementation is built upon several excellent open-source projects. We extend our sincere gratitude to the authors for their valuable contributions.
- [**ELCRec**](https://github.com/yueliu1999/ELCRec): The official implementation for the NeurIPS'24 paper "End-to-end Learnable Clustering for Intent Learning in Recommendation", which inspired our work on intent learning.


<p align="right">(<a href="#top">back to top</a>)</p>

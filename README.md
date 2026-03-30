# Cooperative Graph Neural Networks

Official code for the paper [Cooperative Graph Neural Networks](https://arxiv.org/abs/2310.01267), accepted to ICML 2024.

## Environment Setup

For reproducibility, use the original stack:

- Python 3.9
- PyTorch 2.0.0
- CUDA 11.8
- PyTorch Geometric 2.3.0

Using a dedicated Conda environment is strongly recommended.

```bash - runs fine in ps (not need for legacy git bash)
conda create -n cgnn python=3.9 -y
conda activate cgnn
```

Install dependencies:

```bash - runs fine in ps (not need for legacy git bash)
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch-geometric==2.3.0
pip install torchmetrics ogb rdkit matplotlib tqdm numpy
```

If you run on CPU-only machines, replace the CUDA wheel index with the appropriate CPU wheel index from the PyTorch site.

## Datasets

### Synthetic

- `root_neighbours`
- `cycles`

For synthetic experiments, use `--seed 0` to match the paper protocol.

### Heterophilic Node Classification

- `roman_empire`
- `amazon_ratings`
- `minesweeper`
- `tolokers`
- `questions`

Loaded from PyTorch Geometric.

### Homophilic Node Classification

- `cora`
- `pubmed`

These use fold masks from `folds/`.

### Graph Classification

- `imdb_binary`
- `imdb_multi`
- `reddit_binary`
- `reddit_multi`
- `enzymes`
- `proteins`
- `nci1`

These use preprocessed `.pt` files in `datasets/` plus fold definitions in `folds/*_splits.json`.

### LRGB

- `func` (Peptides-functional)

## Run

Run from the repository root so imports resolve correctly:

```bash
python -u main.py [arguments]
```

Main CLI arguments:

- `--dataset`
- `--pool`
- `--learn_temp`
- `--temp_model_type`
- `--tau0`
- `--temp`
- `--max_epochs`
- `--batch_size`
- `--lr`
- `--dropout`
- `--env_model_type`
- `--env_num_layers`
- `--env_dim`
- `--skip`
- `--batch_norm`
- `--layer_norm`
- `--dec_num_layers`
- `--pos_enc`
- `--act_model_type`
- `--act_num_layers`
- `--act_dim`
- `--seed`
- `--gpu`
- `--fold`
- `--weight_decay`
- `--step_size`
- `--gamma`
- `--num_warmup_epochs`

Argument constraints enforced by code:

- Node-based datasets must use `--pool NONE`.
- Graph-based datasets must use `--pool MEAN` or `--pool SUM`.
- `--fold` is required for social/protein datasets and should be omitted elsewhere.
- `--step_size` and `--gamma` are only for social/protein datasets.
- `--num_warmup_epochs` and `--pos_enc` are only for `--dataset func`.

## Examples

Heterophilic node classification:

```bash
python -u main.py --dataset roman_empire --pool NONE --env_model_type MEAN_GNN --act_model_type MEAN_GNN --env_dim 64 --env_num_layers 3 --act_dim 16 --act_num_layers 1 --seed 0
```

Graph classification (requires fold and pooling):

```bash
python -u main.py --dataset imdb_binary --pool MEAN --fold 0 --env_model_type GIN --act_model_type GIN --env_dim 128 --env_num_layers 3 --act_dim 32 --act_num_layers 1 --step_size 50 --gamma 0.5
```

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{finkelshtein2023cooperative,
  title = "Cooperative Graph Neural Networks",
  author = "Ben Finkelshtein and Xingyue Huang and Michael Bronstein and {\.I}smail {\.I}lkan Ceylan",
  year = "2024",
  booktitle = "Proceedings of Forty-first International Conference on Machine Learning (ICML)",
  url = "https://arxiv.org/abs/2310.01267",
}
```

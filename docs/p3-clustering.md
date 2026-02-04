# Step 3: Joint Clustering (p3_clustering_main.py)

This script performs joint learning of feature representations and cluster assignments using deep clustering.

## Purpose

1. Initialize cluster centers using k-means on pretrained features
2. Jointly optimize feature learning and cluster assignments
3. Use KL divergence to refine cluster assignments
4. Generate final feature embeddings with cluster predictions

## Usage

```bash
# Training with 4 clusters (CPU)
python p3_clustering_main.py --mode train --cluster_number 4 --hours_from_admission 24 --num_variables 5 --num_timestamps 14252 --num_gpus 0

# Training with GPU
python p3_clustering_main.py --mode train --cluster_number 4 --hours_from_admission 24 --num_variables 5 --num_timestamps 14252 --num_gpus 1

# Evaluation only
python p3_clustering_main.py --mode eval --cluster_number 4 --restore
```

## Critical Parameters

| Parameter | Description | How to determine |
|-----------|-------------|------------------|
| `--cluster_number` | Number of clusters | From p2 analysis |
| `--hours_from_admission` | Hours of data | Same as p0/p1 |
| `--num_variables` | Number of vitals | Same as p1 |
| `--num_timestamps` | Max sequence length | Same as p1 |

## Command Line Arguments

### General Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | str | train | 'train' or 'eval' |
| `--cluster_number` | int | 4 | Number of clusters (from p2) |
| `--init_cluster_center` | str | kmeans | Initialization method |
| `--restore_metric` | str | ae_mse | Pretrained model to load |
| `--dc_restore_metric` | str | ae_mse | Deep cluster model metric |
| `--num_gpus` | int | 1 | Number of GPUs |

### Model Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--stopping_delta` | float | 0.0001 | Convergence threshold |
| `--update_interval` | int | 1 | Epochs between target updates |
| `--ref_points` | int | 6 | Interpolation reference points |
| `--dropout` | float | 0.2 | Dropout ratio |

### Learning Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--loss` | str | ae_mse_sup_fake_detect_kl | Loss function |
| `--aux_tasks` | dict | {'future_vital': 0.5} | Auxiliary tasks |
| `--max_epochs` | int | 8000 | Maximum epochs |
| `--init_lr` | float | 0.003 | Initial learning rate |
| `--early_stopping` | int | 50 | Early stopping patience |

## Input Files Required

### From p1 (Pretrained Model)
```
Results/Pretrain/
├── checkpoints/
│   └── best_ae_mse.pth
└── out_feat/
    └── ae_mse/
        ├── training.npy
        └── ...
```

### From p0 (Processed Data)
```
Data/model_data/split_processed/
├── training.pickle
├── validation.pickle
└── testing.pickle
```

## Output Files Generated

### Model Checkpoints
```
Results/Clustering/
├── checkpoints/
│   ├── best_ae_mse.pth
│   ├── best_loss.pth
│   ├── best_delta.pth       # Best by cluster stability
│   └── latest.pth
└── logs/
```

### Feature Embeddings with Clusters
```
Results/Clustering/out_feat/
├── ae_mse/
│   ├── training.npy
│   ├── validation.npy
│   └── testing.npy
├── loss/
│   └── ...
└── delta/
    └── ...
```

Each `.npy` file contains:
```python
{
    'encounter_id': list,       # Encounter IDs
    'hidden': np.array,         # Feature embeddings
    'ob': np.array,             # Original observations
    'padding_mask': np.array,   # Observation masks
    'cluster_pred': np.array,   # Soft cluster assignments [N, K]
    'cluster_label': np.array   # Target cluster labels [N, K]
}
```

## Deep Clustering Algorithm

### 1. Initialization
- Load pretrained interpolation network from p1
- Initialize cluster centers using k-means on training features

### 2. Joint Training Loop
For each epoch:
1. **Forward pass**: Compute features and cluster soft assignments
2. **Target distribution**: Compute sharpened target distribution
3. **KL loss**: Minimize KL divergence between soft and target
4. **Update**: Also minimize autoencoder reconstruction loss

### 3. Convergence
Training stops when:
- `delta` (fraction of samples changing clusters) < `stopping_delta`
- Or `max_epochs` reached
- Or early stopping triggered

## Loss Components

The default loss `ae_mse_sup_fake_detect_kl` includes:

| Component | Weight | Description |
|-----------|--------|-------------|
| `ae_mse` | 1.0 | Autoencoder reconstruction |
| `sup` | varies | Supervised auxiliary tasks |
| `fake_detect` | 1.0 | Fake sample detection |
| `kl` | 10.0 | Cluster assignment KL divergence |

## Monitoring Training

### Key Metrics
- `delta`: Cluster assignment change rate (lower = more stable)
- `ae_mse`: Reconstruction quality
- `loss`: Total loss

### TensorBoard
```bash
tensorboard --logdir Results/Clustering/logs
```

## Common Issues

### Model not improving
**Cause:** Poor initialization or wrong K
**Fix:** Try different `--cluster_number` or `--init_cluster_center random`

### Clusters collapse to one
**Cause:** KL weight too high or learning rate issues
**Fix:** Reduce KL weight in `--unsup_aux_tasks`

### FileNotFoundError: best_ae_mse.pth
**Cause:** p1 not run or didn't save checkpoints
**Fix:** Complete p1 training first

## Example Output

```
INFO - Root directory for saving and loading experiments: Results/Clustering
INFO - training data shape: (1598, 20, 14252)
INFO - Initializing cluster centers with kmeans...
INFO - Epoch 1: loss=2.345, ae_mse=0.123, delta=0.450
INFO - Epoch 2: loss=2.123, ae_mse=0.098, delta=0.234
...
INFO - Epoch 50: loss=1.234, ae_mse=0.045, delta=0.001
INFO - Converged! Delta < stopping_delta
```

## Next Step

After training completes, proceed to [Step 4: Final Clustering](p4-final-clustering.md) to generate final cluster assignments.

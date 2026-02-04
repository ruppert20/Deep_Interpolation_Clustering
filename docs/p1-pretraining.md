# Step 1: Pretraining (p1_pretrain_main.py)

This script trains the interpolation network to learn feature representations from irregularly sampled time series data.

## Purpose

1. Train a deep interpolation network on vital signs time series
2. Learn robust feature representations via autoencoder reconstruction
3. Optionally train auxiliary prediction tasks (future vitals, outcomes)
4. Generate feature embeddings for downstream clustering

## Usage

```bash
# Training mode (CPU)
python p1_pretrain_main.py --mode train --hours_from_admission 24 --num_variables 5 --num_timestamps 14252 --num_gpus 0

# Training mode (GPU)
python p1_pretrain_main.py --mode train --hours_from_admission 24 --num_variables 5 --num_timestamps 14252 --num_gpus 1

# Evaluation only (load trained model)
python p1_pretrain_main.py --mode eval --hours_from_admission 24 --num_variables 5 --num_timestamps 14252 --restore
```

## Critical Parameters

These must match your data from p0:

| Parameter | Description | How to determine |
|-----------|-------------|------------------|
| `--hours_from_admission` | Hours of data | Same as p0 (e.g., 24) |
| `--num_variables` | Number of vital signs | Length of `USE_FEATURES` in info.py |
| `--num_timestamps` | Max sequence length | `max_length` output from p0 |

## Command Line Arguments

### General Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | str | eval | 'train' or 'eval' |
| `--seed` | float | 7529 | Random seed |
| `--num_gpus` | int | 1 | Number of GPUs (0 for CPU) |
| `--restore` | flag | False | Restore from checkpoint |
| `--restore_metric` | str | ae_mse | Metric for model selection |

### Data Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--hours_from_admission` | int | 6 | Hours of data (match p0) |
| `--num_variables` | int | 6 | Number of vitals (match info.py) |
| `--num_timestamps` | int | 354 | Max timestamps (from p0 output) |
| `--batch_size` | int | 256 | Training batch size |
| `--num_workers` | int | 3 | Data loading workers |
| `--scale` | float | 5 | Scale factor for input normalization |

### Model Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--ref_points` | int | 6 | Number of reference points for interpolation |
| `--dropout` | float | 0.2 | Dropout ratio |
| `--fake_detection` | bool | True | Enable fake sample detection task |

### Learning Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--loss` | str | ae_mse_sup_fake_detect | Loss function |
| `--aux_tasks` | dict | {'future_vital': 0.5} | Auxiliary tasks and weights |
| `--max_epochs` | int | 10000 | Maximum training epochs |
| `--init_lr` | float | 0.003 | Initial learning rate |
| `--min_lr` | float | 1e-6 | Minimum learning rate |
| `--early_stopping` | int | 50 | Early stopping patience |

## Input Files Required

From p0_data_process.py:
```
Data/model_data/split_processed/
├── training.pickle
├── validation.pickle
└── testing.pickle
```

### Auxiliary Data

The default auxiliary task is `combined_endpoint`. Required files depend on which tasks you use:

```
Data/analysis_data/
├── table_data.csv           # For combined_endpoint, ICU, rapid_response
└── mortality_summary.csv    # For mort_status_30d, icu_mortality

Data/vital_data/
└── next_hour_abnormal_norm_val.csv  # For future_vital task (optional)
```

See [Data Format](data-format.md) for file specifications and how to generate from parquet.

## Configuring Auxiliary Tasks

The default is `--aux_tasks "{'combined_endpoint': 1.0}"`. To use different tasks:

```bash
# Use future_vital task instead
python p1_pretrain_main.py --mode train --aux_tasks "{'future_vital': 0.5}"

# Use multiple tasks
python p1_pretrain_main.py --mode train --aux_tasks "{'combined_endpoint': 1.0, 'ICU': 1.0}"

# Disable auxiliary tasks entirely
python p1_pretrain_main.py --mode train --aux_tasks "{}"
```

Available tasks: `combined_endpoint`, `ICU`, `rapid_response`, `mort_status_30d`, `icu_mortality`, `future_vital`

## Output Files Generated

### Model Checkpoints
```
Results/Pretrain/
├── checkpoints/
│   ├── best_ae_mse.pth      # Best model by autoencoder MSE
│   ├── best_loss.pth        # Best model by total loss
│   └── latest.pth           # Most recent checkpoint
└── logs/                    # TensorBoard logs
```

### Feature Embeddings
```
Results/Pretrain/out_feat/
├── ae_mse/
│   ├── training.npy
│   ├── validation.npy
│   └── testing.npy
└── loss/
    ├── training.npy
    ├── validation.npy
    └── testing.npy
```

Each `.npy` file contains:
```python
{
    'encounter_id': list,      # Encounter IDs
    'hidden': np.array,        # Feature embeddings [N, hidden_dim]
    'ob': np.array,            # Original observations
    'padding_mask': np.array   # Observation masks
}
```

## Loss Functions

Available loss configurations:

| Loss Name | Components |
|-----------|------------|
| `ae_mse` | Autoencoder reconstruction only |
| `ae_mse_sup` | + Supervised auxiliary tasks |
| `ae_mse_fake_detect` | + Fake sample detection |
| `ae_mse_sup_fake_detect` | All above combined |
| `ae_mse_kl` | + KL divergence regularization |
| `ae_mse_sup_fake_detect_kl` | All components |

## Training Progress

Monitor training via TensorBoard:
```bash
tensorboard --logdir Results/Pretrain/logs
```

Key metrics to watch:
- `ae_mse`: Autoencoder reconstruction error
- `loss`: Total training loss
- `delta`: Label change rate (for clustering)

## Common Issues

### FileNotFoundError: table_data.csv
**Cause:** Auxiliary tasks enabled but data files missing
**Fix:** Either provide the files or disable aux_tasks (see above)

### CUDA out of memory
**Fix:** Reduce `--batch_size` or use CPU with `--num_gpus 0`

### Shape mismatch errors
**Cause:** Parameters don't match data
**Fix:** Ensure `--num_variables` and `--num_timestamps` match p0 output

## Example Output

```
21:28:59 INFO - set_seed(39): The global seed: 7529
21:28:59 INFO - main(116): Root directory for saving and loading experiments: Results/Pretrain
21:29:00 INFO - _fix_input_format(63): dict_keys(['feat', 'time_step', 'padding_mask', 'encounter_id', 'drop_mask'])
21:29:01 INFO - _fix_input_format(71): training data shape: (1598, 20, 14252)
21:29:02 INFO - _scale_data(77): Scale input data to [-2.5  2.5]
...
```

## Next Step

After successful training, proceed to [Step 2: Optimal K Selection](p2-optimal-k.md).

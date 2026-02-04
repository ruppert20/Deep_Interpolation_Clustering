# Step 0: Data Processing (p0_data_process.py)

This script preprocesses raw vital signs data into a format suitable for the deep learning pipeline.

## Purpose

1. Load and validate input data files
2. Generate fixed-size arrays from irregularly sampled time series
3. Split data into training/validation/testing cohorts
4. Apply mean imputation for missing values
5. Create hold-out masks for autoencoder training
6. Normalize data using min-max scaling

## Usage

```bash
# Basic usage with defaults
python p0_data_process.py

# Specify parameters
python p0_data_process.py --hours_from_admission 24 --norm_method minmax

# Force reprocessing (clear all checkpoints)
python p0_data_process.py --clean
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--hours_from_admission` | int | 24 | Hours of vital data to include from admission |
| `--norm_method` | str | minmax | Normalization method (only 'minmax' supported) |
| `--clean` | flag | False | Clear all checkpoints and reprocess from scratch |

## Input Files Required

| File | Path | Description |
|------|------|-------------|
| Encounters | `Data/encounter_data/encounters.csv` | List of encounter IDs |
| Vital Data | `Data/vital_data/original_data_{hours}h.pickle` | Time series vital signs |
| Split Indices | `Data/model_data/idx.pickle` | Train/val/test split IDs |

See [Data Format](data-format.md) for detailed file specifications.

## Output Files Generated

### Checkpoints
```
Data/model_data/checkpoints/
└── data_dict_{hours}h.pickle    # Generated data arrays (expensive to compute)
```

### Split Original Data
```
Data/model_data/split_org/
├── training_feat.pickle
├── training_time_step.pickle
├── training_padding_mask.pickle
├── training_encounter_id.pickle
├── validation_*.pickle
└── testing_*.pickle
```

### Processed Data (Final Output)
```
Data/model_data/split_processed/
├── training.pickle
├── validation.pickle
└── testing.pickle
```

Each processed pickle contains:
```python
{
    'feat': np.array,           # [N, num_features, max_timesteps] - vital values
    'time_step': np.array,      # [N, num_features, max_timesteps] - time stamps
    'padding_mask': np.array,   # [N, num_features, max_timesteps] - observation mask
    'drop_mask': np.array,      # [N, num_features, max_timesteps] - autoencoder hold-out mask
    'encounter_id': list        # List of encounter IDs
}
```

## Processing Steps

### Step 1: Load Encounter File
Reads `encounters.csv` and extracts unique encounter IDs.

### Step 2: Load Vital Data
Loads the pickle file containing time series data for each vital sign.

### Step 3: Generate Data Arrays
Converts irregularly sampled time series into fixed-size arrays:
- Finds maximum number of timestamps across all patients/vitals
- Creates zero-padded arrays of shape `[num_patients, num_features, max_length]`
- Records which positions have actual observations via padding mask

### Step 4: Load Split Indices
Loads the predefined train/validation/test split from `idx.pickle`.

### Step 5: Split and Save Original Data
Separates data by cohort and saves intermediate files.

### Step 6: Mean Imputation
For patients missing an entire vital sign:
- Training: Impute with global mean of that vital
- Validation/Testing: Impute with training mean

### Step 7: Hold-Out Mask Creation
Randomly masks 20% of observed values for autoencoder reconstruction loss.

### Step 8: Normalization
Applies min-max scaling using predefined ranges from `info.py`.

## Checkpointing

The script supports resumable processing:

- **Data dict checkpoint**: Saves after Step 3 (most expensive step)
- **Split org files**: Saved per-cohort, skipped if already exist
- **Processed files**: Final output, triggers early exit if all exist

Use `--clean` to force complete reprocessing.

## Configuration

Edit `info.py` to modify:

```python
# Vital signs to process
USE_FEATURES = ['sbp', 'dbp', 'heartRate', 'spo2', 'respiratory']

# Normalization ranges
MIN_MAX_VALUES = {
    'sbp': [20, 300],
    'dbp': [5, 225],
    'heartRate': [0, 300],
    'spo2': [0, 100],
    'respiratory': [0, 60]
}
```

## Common Issues

### KeyError: encounter ID not found
**Cause:** Encounter IDs in vital data don't match encounters.csv
**Fix:** Ensure ID types are consistent (both strings or both integers)

### IndexError: index out of bounds for axis 1
**Cause:** `USE_FEATURES` in `info.py` doesn't match actual vital data keys
**Fix:** Update `USE_FEATURES` to match your data

### FileNotFoundError
**Cause:** Input files not in expected locations
**Fix:** Check directory structure matches expected paths

## Example Output

```
==================================================
Step 1: Loading encounter file...
  Loaded 2279 encounters
==================================================
Step 2: Loading vital data (24h)...
  Loaded 5 vital features: ['sbp', 'dbp', 'heartRate', 'spo2', 'respiratory']
==================================================
Step 3: Generating data arrays...
  [CHECKPOINT] Loading data_dict_24h from cache...
  Data arrays shape: (2279, 5, 14252)
==================================================
Step 4: Loading split indices...
  training: 1598 samples
  validation: 341 samples
  testing: 340 samples
==================================================
Step 5: Splitting and saving original data...
  training: loaded 4 from cache, saved 0 new
  validation: loaded 4 from cache, saved 0 new
  testing: loaded 4 from cache, saved 0 new
==================================================
All processed data files already exist!
  Use --clean flag to force reprocessing
  Output location: /path/to/Data/model_data/split_processed
```

## Next Step

After successful completion, proceed to [Step 1: Pretraining](p1-pretraining.md).

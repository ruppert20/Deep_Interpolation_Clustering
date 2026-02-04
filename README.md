# Deep_Interpolation_Clustering
This is the repo for the implementation of paper 'Identifying acute illness phenotypes via deep temporal interpolation and clustering network on physiologic signatures'.

![image](https://github.com/Prisma-pResearch/Deep_Interpolation_Clustering/assets/31426497/ee517db4-4990-400b-b4c3-77971124ec8e)

## Documentation

| Document | Description |
|----------|-------------|
| [Data Format](docs/data-format.md) | Complete file structure and data specifications |
| [Step 0: Data Processing](docs/p0-data-processing.md) | Preprocessing raw vital signs data |
| [Step 1: Pretraining](docs/p1-pretraining.md) | Training the interpolation network |
| [Step 2: Optimal K](docs/p2-optimal-k.md) | Determining optimal number of clusters |
| [Step 3: Clustering](docs/p3-clustering.md) | Joint feature learning and clustering |
| [Step 4: Final Clusters](docs/p4-final-clustering.md) | Generating final cluster assignments |

## Prerequisites
**Python 3.12**

Install dependencies:
```bash
pip install -r requirements.txt
```

Or install individually:
| Package          | Version  |
|------------------|----------|
| torch            | 2.6.0    |
| tensorboardX     | 2.6.2.2  |
| scikit-learn     | 1.8.0    |
| numpy            | 2.4.0    |
| pandas           | 2.3.3    |
| tensorflow       | 2.18.0   |
| seaborn          | 0.13.2   |
| matplotlib       | 3.10.8   |
| scipy            | 1.16.3   |
| kneed            | 0.8.5    |
| warmup-scheduler | 0.3      |

## Input data format
We require three inputs: <br />
* csv file: column 'encounter_deiden_id' lists the all encounter ids
* pickle file: a dictionary file records the time series vital signs within the first 24 hours (at least 7 hours) of hospital admission. For each <Key, Value> pair in the dictionary, Key represents the vital name and Value represents the dataframe of corresponding time series vital sign data.  Six vitals including 'sbp' (systolic blood pressure), 'dbp' (diastolic blood pressure), 'heartRate' (heart rate), 'temperature', 'spo2' (saturation of peripheral oxygen) and 'respiratory' (respiratory rate) were used. Each time series vital sign dataframe contains three columns: encounter_deiden_id, time_stamp, and measurement, where time_stamp represents the hours from admission, and measurement represents the vital sign value at that time stamp.
* pickle file: a dictionary file contains the ids of three cohorts (training, validation and testing). For each <Key, Value> pair in the dictionary, Key represents the cohort name and Value represents the list of encounter ids.

### Input data mockups

#### 1. Encounter CSV file (`encounters.csv`)
A CSV file with at minimum the `encounter_deiden_id` column listing all unique patient encounter IDs:

| encounter_deiden_id |
|---------------------|
| ENC001              |
| ENC002              |
| ENC003              |
| ENC004              |
| ...                 |

#### 2. Vital data pickle file (`original_data_6h.pickle`)
A Python dictionary where each key is a vital sign name and each value is a pandas DataFrame containing time series measurements:

```python
{
    'sbp': DataFrame,        # Systolic blood pressure (mmHg)
    'dbp': DataFrame,        # Diastolic blood pressure (mmHg)
    'heartRate': DataFrame,  # Heart rate (bpm)
    'temperature': DataFrame,# Body temperature (Celsius)
    'spo2': DataFrame,       # Oxygen saturation (%)
    'respiratory': DataFrame # Respiratory rate (breaths/min)
}
```

Each DataFrame has three columns:

| encounter_deiden_id | time_stamp | measurement |
|---------------------|------------|-------------|
| ENC001              | 0.0        | 120.0       |
| ENC001              | 0.5        | 118.0       |
| ENC001              | 1.2        | 122.0       |
| ENC001              | 2.0        | 119.0       |
| ENC002              | 0.0        | 135.0       |
| ENC002              | 0.8        | 132.0       |
| ENC002              | 1.5        | 140.0       |
| ...                 | ...        | ...         |

- `encounter_deiden_id`: Patient encounter identifier (must match IDs in encounter CSV)
- `time_stamp`: Hours from hospital admission (float, e.g., 0.0, 0.5, 1.25)
- `measurement`: Vital sign value at that time point

**Note:** Measurements are irregularly sampled - each patient may have different time stamps and different numbers of observations per vital sign.

#### 3. Split index pickle file (`idx.pickle`)
A Python dictionary defining train/validation/test splits by encounter ID:

```python
{
    'training_idx': ['ENC001', 'ENC002', 'ENC005', 'ENC008', ...],    # ~70% of encounters
    'validation_idx': ['ENC003', 'ENC006', 'ENC009', ...],           # ~15% of encounters
    'testing_idx': ['ENC004', 'ENC007', 'ENC010', ...]               # ~15% of encounters
}
```

### Expected value ranges
The following min/max ranges are used for normalization:

| Vital         | Min  | Max  | Unit          |
|---------------|------|------|---------------|
| sbp           | 20   | 300  | mmHg          |
| dbp           | 5    | 225  | mmHg          |
| heartRate     | 0    | 300  | bpm           |
| temperature   | 24   | 45   | Celsius       |
| spo2          | 0    | 100  | %             |
| respiratory   | 0    | 60   | breaths/min   |

### Creating mock data (Python example)
```python
import pandas as pd
import pickle

# 1. Create encounter CSV
encounters = pd.DataFrame({'encounter_deiden_id': ['ENC001', 'ENC002', 'ENC003']})
encounters.to_csv('encounters.csv', index=False)

# 2. Create vital data dictionary
vital_data = {
    'sbp': pd.DataFrame({
        'encounter_deiden_id': ['ENC001', 'ENC001', 'ENC002', 'ENC002', 'ENC003'],
        'time_stamp': [0.0, 1.5, 0.0, 2.0, 0.5],
        'measurement': [120.0, 118.0, 135.0, 132.0, 110.0]
    }),
    'dbp': pd.DataFrame({
        'encounter_deiden_id': ['ENC001', 'ENC001', 'ENC002', 'ENC003'],
        'time_stamp': [0.0, 1.5, 0.0, 0.5],
        'measurement': [80.0, 78.0, 90.0, 72.0]
    }),
    # ... repeat for heartRate, temperature, spo2, respiratory
}

with open('original_data_6h.pickle', 'wb') as f:
    pickle.dump(vital_data, f)

# 3. Create split indices
split_idx = {
    'training_idx': ['ENC001'],
    'validation_idx': ['ENC002'],
    'testing_idx': ['ENC003']
}

with open('idx.pickle', 'wb') as f:
    pickle.dump(split_idx, f)
```

## Quick Start

See the [detailed documentation](docs/) for complete instructions. Basic workflow:

```bash
# Step 0: Process data
python p0_data_process.py --hours_from_admission 24

# Step 1: Pretrain interpolation network
python p1_pretrain_main.py --mode train --hours_from_admission 24 \
    --num_variables 5 --num_timestamps <MAX_LENGTH> --num_gpus 0

# Step 2: Determine optimal K
python p2_clustering_optK.py --k_max 10

# Step 3: Joint clustering
python p3_clustering_main.py --mode train --cluster_number 4 \
    --hours_from_admission 24 --num_variables 5 --num_timestamps <MAX_LENGTH>

# Step 4: Generate final clusters
python p4_clustering_final.py --cluster_method kmeans --num_clusters 4
```

**Note:** Replace `<MAX_LENGTH>` with the `max_length` value output by p0 (e.g., 14252).

### Key Parameters

| Parameter | Description | Where to find |
|-----------|-------------|---------------|
| `--hours_from_admission` | Hours of data | Your data choice |
| `--num_variables` | Number of vitals | `len(USE_FEATURES)` in info.py |
| `--num_timestamps` | Max sequence length | `max_length` from p0 output |
| `--cluster_number` | Number of clusters | Determined from p2 |


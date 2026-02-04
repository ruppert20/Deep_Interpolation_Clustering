# Step 4: Final Clustering (p4_clustering_final.py)

This script generates final cluster assignments and saves them in a standardized format for downstream analysis.

## Purpose

1. Load trained feature embeddings from p3
2. Apply clustering algorithm (k-means, DBSCAN, or deep learning labels)
3. Align cluster labels based on clinical meaning (e.g., by average SBP)
4. Save final cluster assignments for each cohort

## Usage

```bash
# K-means clustering with 4 clusters
python p4_clustering_final.py --cluster_method kmeans --num_clusters 4

# Use deep learning cluster assignments
python p4_clustering_final.py --cluster_method dl --num_clusters 4

# DBSCAN clustering
python p4_clustering_final.py --cluster_method dbscan --opt_eps 1.9
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--cluster_method` | str | kmeans | Algorithm: kmeans, dbscan, dl, consensus |
| `--num_clusters` | int | 4 | Number of clusters (k-means, dl, consensus) |
| `--restore_metric` | list | ['ae_mse', 'loss', 'delta'] | Models to process |
| `--opt_eps` | float | 1.9 | Epsilon for DBSCAN |
| `--dl_cluster_label_type` | str | pred | For dl: use 'label' or 'pred' |

## Clustering Methods

### 1. K-Means (`--cluster_method kmeans`)
- Applies k-means to learned feature embeddings
- Most commonly used method
- Requires specifying `--num_clusters`

### 2. Deep Learning (`--cluster_method dl`)
- Uses cluster assignments directly from p3 training
- No additional clustering needed
- Choose between soft predictions (`pred`) or hard labels (`label`)

### 3. DBSCAN (`--cluster_method dbscan`)
- Density-based clustering
- Automatically determines number of clusters
- Requires tuning `--opt_eps` (from p2 analysis)

### 4. Consensus (`--cluster_method consensus`)
- Combines multiple clustering results
- Requires external consensus labels in specific format

## Input Files Required

Feature embeddings from p3:
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

## Output Files Generated

```
Results/Clustering/out_feat/{metric}_{method}_aligned/
├── training_{K}.npy
├── validation_{K}.npy
└── testing_{K}.npy
```

Each `.npy` file contains:
```python
{
    'encounter_id': list,    # Encounter IDs
    'hidden': np.array,      # Feature embeddings [N, hidden_dim]
    'cluster_id': np.array   # Cluster assignments [N,] (0 to K-1)
}
```

## Cluster Alignment

Clusters are aligned based on average systolic blood pressure (SBP) to ensure consistent clinical interpretation:

1. Compute average SBP for each cluster
2. Sort clusters by descending SBP
3. Relabel so cluster 0 has highest average SBP

This ensures:
- Cluster 0: Highest average SBP (potentially hypertensive phenotype)
- Cluster K-1: Lowest average SBP (potentially hypotensive phenotype)

## Example Workflow

### Standard Workflow (K-Means)
```bash
# Generate final clusters using k-means
python p4_clustering_final.py --cluster_method kmeans --num_clusters 4
```

### Using Deep Learning Labels
```bash
# Use the cluster predictions from p3
python p4_clustering_final.py --cluster_method dl --dl_cluster_label_type pred
```

### Multiple Metrics
The script processes all metrics in `--restore_metric`:
```bash
# Process ae_mse, loss, and delta
python p4_clustering_final.py --cluster_method kmeans --num_clusters 4 --restore_metric ae_mse loss delta
```

## Interpreting Results

### Loading Results
```python
import numpy as np

# Load final cluster assignments
data = np.load('Results/Clustering/out_feat/ae_mse_kmeans_aligned/training_4.npy',
               allow_pickle=True).item()

encounter_ids = data['encounter_id']
features = data['hidden']
cluster_ids = data['cluster_id']

# Count samples per cluster
unique, counts = np.unique(cluster_ids, return_counts=True)
for c, n in zip(unique, counts):
    print(f"Cluster {c}: {n} samples")
```

### Cluster Distribution
Good clustering typically shows:
- Relatively balanced cluster sizes (no cluster with <5% of samples)
- Distinct separation in feature space
- Clinically meaningful differences between clusters

## Common Issues

### All samples in one cluster
**Cause:** Poor feature representations or wrong K
**Fix:** Retrain p3 or try different K

### Alignment fails
**Cause:** Different clusters map to same aligned ID
**Fix:** Usually indicates overlapping clusters; try different K

### FileNotFoundError
**Cause:** p3 didn't complete or wrong metric specified
**Fix:** Verify p3 output files exist

## Downstream Analysis

After generating final clusters, typical analyses include:

1. **Clinical characterization**: Compare outcomes, demographics across clusters
2. **Visualization**: t-SNE/UMAP plots colored by cluster
3. **Trajectory analysis**: Examine vital sign patterns within clusters
4. **Outcome prediction**: Use cluster membership as a feature

### Example Analysis Code
```python
import numpy as np
import pandas as pd

# Load cluster assignments
train = np.load('Results/Clustering/out_feat/ae_mse_kmeans_aligned/training_4.npy',
                allow_pickle=True).item()

# Create DataFrame for analysis
df = pd.DataFrame({
    'encounter_id': train['encounter_id'],
    'cluster': train['cluster_id']
})

# Merge with outcomes data for analysis
outcomes = pd.read_csv('Data/analysis_data/table_data.csv')
analysis = df.merge(outcomes, on='encounter_id')

# Compare outcomes by cluster
print(analysis.groupby('cluster')['AKI_overall'].value_counts(normalize=True))
```

## Summary

The p4 script is the final step in the clustering pipeline:

| Step | Script | Output |
|------|--------|--------|
| p0 | p0_data_process.py | Processed time series data |
| p1 | p1_pretrain_main.py | Pretrained interpolation network |
| p2 | p2_clustering_optK.py | Optimal K determination |
| p3 | p3_clustering_main.py | Joint clustering model |
| **p4** | **p4_clustering_final.py** | **Final cluster assignments** |

The final cluster assignments can now be used for clinical analysis and phenotype characterization.

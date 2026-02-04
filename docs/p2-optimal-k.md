# Step 2: Optimal K Selection (p2_clustering_optK.py)

This script helps determine the optimal number of clusters using various statistical methods.

## Purpose

1. Load pretrained feature embeddings from p1
2. Evaluate clustering quality for different values of K
3. Generate visualizations (elbow plots, gap statistic)
4. Compute internal validation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)

## Usage

```bash
# K-means with default settings
python p2_clustering_optK.py

# Specify maximum K to evaluate
python p2_clustering_optK.py --k_max 15

# Use DBSCAN instead
python p2_clustering_optK.py --cluster_method dbscan
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--cluster_method` | str | kmeans | Clustering algorithm: kmeans, dbscan, optics |
| `--k_max` | int | 10 | Maximum K to evaluate (k-means only) |
| `--n_init` | int | 10 | Number of k-means initializations |
| `--gap_b` | int | 10 | Number of reference samples for gap statistic |
| `--restore_metric` | list | ['ae_mse', 'loss'] | Which pretrained models to evaluate |
| `--select_opt_k` | list | ['gap_sts', 'elbow'] | Methods for selecting optimal K |
| `--internal_metrics` | list | [...] | Internal validation metrics |
| `--opt_eps` | float | 1.9 | Epsilon for DBSCAN (if using) |

## Input Files Required

Feature embeddings from p1:
```
Results/Pretrain/out_feat/
├── ae_mse/
│   ├── training.npy
│   ├── validation.npy
│   └── testing.npy
└── loss/
    └── ...
```

## Output Files Generated

```
Results/Pretrain/out_feat/{metric}_kmeans_aligned/plot/
├── train_elbow.png          # Elbow method plot
├── valid_elbow.png          # Validation elbow plot
├── gap_statistic-1_v1.png   # Gap statistic (gap only)
├── gap_statistic-2_v1.png   # Gap statistic (with ref/act)
└── gap_sts_v1.csv           # Raw gap statistic data
```

For DBSCAN:
```
Results/Pretrain/out_feat/{metric}_dbscan_aligned/plot/
└── {k}-NN distance.png      # K-distance graph for eps selection
```

## Methods for Selecting K

### 1. Elbow Method
Plots distortion (average distance to cluster center) vs K:
- Look for the "elbow" where adding more clusters gives diminishing returns
- More objective than visual inspection

### 2. Gap Statistic
Compares within-cluster dispersion to a null reference distribution:
- Higher gap values indicate better clustering
- Optimal K is where gap peaks or stabilizes

### 3. Internal Metrics

| Metric | Range | Optimal |
|--------|-------|---------|
| Silhouette | [-1, 1] | Higher is better |
| Davies-Bouldin | [0, ∞) | Lower is better |
| Calinski-Harabasz | [0, ∞) | Higher is better |

## Interpreting Results

### Elbow Plot
```
Distortion
    │
    │\
    │ \
    │  \___________
    │
    └──────────────── K
           ^
         elbow
```

### Gap Statistic
```
Gap Value
    │      ___
    │     /   \
    │    /     \__
    │   /
    │__/
    └──────────────── K
           ^
      optimal K
```

## Example Workflow

1. Run with default settings:
   ```bash
   python p2_clustering_optK.py --k_max 10
   ```

2. Examine generated plots in `Results/Pretrain/out_feat/ae_mse_kmeans_aligned/plot/`

3. Look at `gap_sts_v1.csv` for numerical values:
   ```csv
   k,gap,ref,act,ref_s,Sihouette,Davies-Bouldin_Index,Calinski-Harabasz
   2,0.1234,5.67,5.55,0.02,0.45,0.89,234.5
   3,0.1456,5.89,5.73,0.03,0.52,0.76,345.6
   ...
   ```

4. Select K based on:
   - Elbow point in distortion plot
   - Peak in gap statistic
   - Best internal metrics

## Common Issues

### FileNotFoundError: training.npy
**Cause:** p1 hasn't been run or didn't generate features
**Fix:** Run p1 with `--mode train` first, ensure it completes successfully

### All clusters have same size
**Cause:** Poor feature representations
**Fix:** Try different `--restore_metric` or retrain p1

## Notes

- The optimal K is determined **manually** based on the plots
- Consider domain knowledge when selecting K
- Different metrics may suggest different optimal K values
- It's common to try a few values of K in subsequent steps

## Next Step

After determining optimal K, proceed to [Step 3: Joint Clustering](p3-clustering.md) with your chosen cluster count.

# Step 0: Saves metadata.json
  python p0_data_process.py --hours_from_admission 24

  # Step 1: Auto-loads num_timestamps, num_variables, hours (no manual args needed)
  python p1_pretrain_main.py --mode train --num_gpus 0 --num_workers 0 --max_epochs 5000 --early_stopping 100

  # Step 2: Analyze to choose K
  python p2_clustering_optK.py --k_max 10

  # Step 3: Specify cluster_number (saves to metadata for p4)
  python p3_clustering_main.py --mode train --cluster_number 3 --num_gpus 0 --num_workers 0

  # Step 4: Auto-loads cluster_number (no manual args needed)
  python p4_clustering_final.py --cluster_method kmeans
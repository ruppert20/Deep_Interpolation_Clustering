#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert outcomes_clean.parquet to auxiliary task CSV files.

This script reads the outcomes parquet file and generates the CSV files
required for auxiliary prediction tasks in the deep clustering pipeline.

Input:
    Data/outcomes_clean.parquet

Output:
    Data/analysis_data/table_data.csv
    Data/analysis_data/mortality_summary.csv
"""
import polars as pl
import os
from info import BASE_PATH

def int_to_yn(col):
    """Convert integer 0/1 column to 'Y'/'N' string format."""
    return pl.when(pl.col(col) == 1).then(pl.lit('Y')).otherwise(pl.lit('N')).alias(col)


def main():
    # Paths
    data_path = os.path.join(BASE_PATH, 'Data')
    input_path = os.path.join(data_path, 'outcomes_clean.parquet')
    output_path = os.path.join(data_path, 'analysis_data')

    # Read parquet
    print(f"Reading {input_path}...")
    df = pl.read_parquet(input_path)
    print(f"  Loaded {len(df)} rows")
    print(f"  Columns: {df.columns}")

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Create table_data.csv with combined_endpoint and ICU
    print("\nCreating table_data.csv...")
    table_data = df.select([
        pl.col('deiden_study_id').cast(pl.Utf8).alias('encounter_deiden_id'),
        int_to_yn('combined_endpoint'),
        pl.when(pl.col('Icu_Admission') == 1).then(pl.lit('Y')).otherwise(pl.lit('N')).alias('ICU'),
        pl.when(pl.col('rapid_response') == 1).then(pl.lit('Y')).otherwise(pl.lit('N')).alias('rapid_response'),
    ])

    table_data_path = os.path.join(output_path, 'table_data.csv')
    table_data.write_csv(table_data_path)
    print(f"  Saved to {table_data_path}")
    print(f"  Columns: {table_data.columns}")
    print(f"  combined_endpoint distribution:")
    print(table_data['combined_endpoint'].value_counts())

    # Create mortality_summary.csv
    print("\nCreating mortality_summary.csv...")
    mortality_data = df.select([
        pl.col('deiden_study_id').cast(pl.Utf8).alias('encounter_deiden_id'),
        pl.when(pl.col('in_hosp_mortality') == 1).then(pl.lit('Y')).otherwise(pl.lit('N')).alias('mort_status_30d'),
        pl.when(pl.col('icu_mortality') == 1).then(pl.lit('Y')).otherwise(pl.lit('N')).alias('icu_mortality'),
    ])

    mortality_path = os.path.join(output_path, 'mortality_summary.csv')
    mortality_data.write_csv(mortality_path)
    print(f"  Saved to {mortality_path}")
    print(f"  Columns: {mortality_data.columns}")

    print("\nDone! Auxiliary data files created.")


if __name__ == "__main__":
    main()

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask import delayed
import os
from pathlib import Path

def add_ordered_ids_to_parquet_dataset(
    input_dir: str,
    output_dir: str,
    id_field: str = "id",
    start_index: int = 0,
    partition_size: str = "256MB"
):
    """
    Add ordered integer IDs to a large dataset stored as parquet files.
    
    Args:
        input_dir: Directory containing input parquet files
        output_dir: Directory to save output parquet files
        id_field: Name of the ID column to add
        start_index: Starting ID number
        partition_size: Size of output partitions
    """
    
    # Read all parquet files as a single dask dataframe
    print("Loading dataset...")
    df = dd.read_parquet(input_dir)
    
    print(f"Dataset shape: {df.shape[0].compute()} rows, {df.shape[1]} columns")
    print(f"Number of partitions: {df.npartitions}")
    
    # Add ordered IDs using the same logic as the original code
    df_with_ids = add_ordered_ids(df, id_field, start_index)
    
    # Optionally repartition for better performance
    print(f"Repartitioning to {partition_size} partitions...")
    df_with_ids = df_with_ids.repartition(partition_size=partition_size)
    
    # Save to parquet
    print(f"Saving to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    df_with_ids.to_parquet(
        output_dir,
        engine='pyarrow',
        compression='snappy',
        write_index=False
    )
    
    print(f"Complete! Saved {df_with_ids.npartitions} parquet files to {output_dir}")
    return df_with_ids

def add_ordered_ids(df: dd.DataFrame, id_field: str, start_index: int = 0) -> dd.DataFrame:
    """
    Add ordered integer IDs to a dask dataframe, maintaining order across partitions.
    """
    
    # Create meta with the new ID column
    meta = df._meta.copy()
    meta[id_field] = pd.Series([], dtype='int64')
    
    # Convert to delayed objects
    delayed_partitions = df.to_delayed()
    
    # Calculate partition lengths (excluding the last partition initially)
    partition_lengths = [0]  # Start with 0 for cumsum
    for partition in delayed_partitions[:-1]:
        partition_lengths.append(delayed(len)(partition))
    
    # Calculate cumulative sum to get starting ID for each partition
    cumulative_lengths = delayed(np.cumsum)(partition_lengths)
    
    # Process each partition with its starting ID
    delayed_id_partitions = []
    for i, partition in enumerate(delayed_partitions):
        delayed_id_partitions.append(
            delayed(add_ids_to_partition)(partition, cumulative_lengths[i], start_index, id_field)
        )
    
    # Create new dask dataframe from delayed partitions
    return dd.from_delayed(delayed_id_partitions, meta=meta)

def add_ids_to_partition(partition: pd.DataFrame, partition_start_id: int, start_index: int, id_field: str) -> pd.DataFrame:
    """
    Add integer IDs to a single partition.
    """
    # Create sequential IDs for this partition
    end_id = partition_start_id + len(partition)
    partition[id_field] = range(partition_start_id + start_index, end_id + start_index)
    
    return partition

# CLI usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Add ordered integer IDs to parquet dataset')
    parser.add_argument('input_dir', help='Directory containing input parquet files')
    parser.add_argument('output_dir', help='Directory to save output parquet files')
    parser.add_argument('--id-field', default='id', help='Name of ID column (default: id)')
    parser.add_argument('--start-index', type=int, default=0, help='Starting ID number (default: 0)')
    parser.add_argument('--partition-size', default='100MB', help='Output partition size (default: 100MB)')
    parser.add_argument('--verify', action='store_true', help='Verify ID ordering after processing')
    
    args = parser.parse_args()
    
    # Process the dataset
    df_result = add_ordered_ids_to_parquet_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        id_field=args.id_field,
        start_index=args.start_index,
        partition_size=args.partition_size
    )
    
    # Optional verification
    if args.verify:
        print(f"\nFinal dataset has {df_result.npartitions} partitions")
        print("First few rows with IDs:")
        print(df_result.head())
        
        print("\nVerifying ID ordering...")
        first_partition_ids = df_result.get_partition(0)[args.id_field].compute()
        print(f"First partition IDs: {first_partition_ids.head().tolist()} ... {first_partition_ids.tail().tolist()}")
        
        if df_result.npartitions > 1:
            last_partition_ids = df_result.get_partition(-1)[args.id_field].compute()
            print(f"Last partition IDs: {last_partition_ids.head().tolist()} ... {last_partition_ids.tail().tolist()}")

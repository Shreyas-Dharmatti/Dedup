import argparse
import time
import os
import glob
import logging
from pathlib import Path
import pandas as pd
import json
from tqdm import tqdm
from typing import List, Dict
from datasketch import MinHash, MinHashLSH
import tokenize
from io import BytesIO
from multiprocessing import Pool, cpu_count
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# Tokenizes the given code text using Python's tokenizer
def extract_code_tokens(text: str) -> List[str]:
    try:
        tokens = []
        g = tokenize.tokenize(BytesIO(text.encode('utf-8')).readline)
        for toknum, tokval, _, _, _ in g:
            if toknum == tokenize.ENDMARKER:
                break
            if tokval.strip():
                tokens.append(tokval)
        return tokens
    except:
        return []


# measures time
def timed_step(name: str, log_key: str, func, log_dict: dict):
    start = time.time()
    result = func()
    end = time.time()
    duration = end - start
    log_dict[log_key] = duration
    logger.info(f"{name:<30}: {duration:.2f}s")
    return result


# Worker function for parallel row group processing (must be at module level for pickling)
def process_row_group_worker(args):
    """Process a single row group"""
    file_path, rg_idx, text_column, min_length = args
    parquet_file = pq.ParquetFile(file_path)
    df = parquet_file.read_row_group(rg_idx, columns=[text_column]).to_pandas()
    # Vectorized filtering
    mask = df[text_column].astype(str).str.len() >= min_length
    df_filtered = df[mask]
    records = df_filtered.to_dict('records')
    del df, df_filtered
    return records


# OPTIMIZED: Parallel parquet loading by row groups
def load_parquet_parallel(file_path: str, text_column: str, min_length: int, num_workers: int) -> List[Dict]:
    """Load parquet file in parallel using row groups"""
    parquet_file = pq.ParquetFile(file_path)
    total_rows = parquet_file.metadata.num_rows
    num_row_groups = parquet_file.num_row_groups
    
    logger.info(f"File has {num_row_groups} row groups, using {num_workers} workers")
    
    # Prepare arguments for parallel processing
    args_list = [(file_path, rg_idx, text_column, min_length) for rg_idx in range(num_row_groups)]
    
    # Process all row groups in parallel
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_row_group_worker, args_list),
            total=num_row_groups,
            desc="Loading row groups"
        ))
    
    # Flatten results
    records = []
    for result in results:
        records.extend(result)
    
    return records


# OPTIMIZED: Load input files with parallel parquet reading
def load_input_files(input_path: str, text_column: str, max_files: int = None, min_length: int = 0, num_workers: int = None) -> List[Dict]:
    input_path = Path(input_path)
    records = []
    num_workers = num_workers or cpu_count()

    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(list(input_path.glob("*.jsonl")) + list(input_path.glob("*.parquet")))
    else:
        files = sorted(glob.glob(input_path))

    if max_files:
        files = files[:max_files]

    for file in tqdm(files, desc="Reading input files"):
        if str(file).endswith(".jsonl"):
            with open(file, "r") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if len(obj.get(text_column, "")) >= min_length:
                            records.append(obj)
                    except json.JSONDecodeError:
                        continue
        elif str(file).endswith(".parquet"):
            parquet_file = pq.ParquetFile(file)
            total_rows = parquet_file.metadata.num_rows
            
            logger.info(f"Loading {file.name} ({total_rows:,} rows)...")
            logger.info(f"Using parallel loading with {num_workers} workers...")
            
            start_time = time.time()
            file_records = load_parquet_parallel(str(file), text_column, min_length, num_workers)
            load_time = time.time() - start_time
            
            logger.info(f"✓ Loaded {len(file_records):,} records in {load_time:.1f}s ({len(file_records)/load_time:.0f} rows/sec)")
            records.extend(file_records)
        else:
            raise ValueError("Unsupported file format")
    
    logger.info(f"Total loaded: {len(records):,} records")
    return records


# OPTIMIZED: Pre-compute shingles to avoid redundant work
def minhash_worker(args):
    idx, content, num_perm, ngram_size = args
    mh = MinHash(num_perm=num_perm)
    tokens = extract_code_tokens(content)
    
    # OPTIMIZATION: Use generator instead of creating full set first
    if len(tokens) >= ngram_size:
        for i in range(len(tokens) - ngram_size + 1):
            shingle = " ".join(tokens[i:i + ngram_size])
            mh.update(shingle.encode("utf-8"))
    
    return idx, mh


# OPTIMIZED: Main Deduplication Function with better chunk size and progress
def fuzzy_deduplicate(data: List[Dict], text_column: str, threshold: float = 0.85, num_perm: int = 64, ngram_size: int = 5, num_cpus: int = None, timings: dict = None) -> List[Dict]:
    num_workers = num_cpus or cpu_count()

    logger.info(f"Generating MinHash signatures using {num_workers} CPU cores...")
    args_list = [(i, item[text_column], num_perm, ngram_size) for i, item in enumerate(data)]

    # OPTIMIZATION: Better chunk size for multiprocessing
    optimal_chunksize = max(1, len(data) // (num_workers * 10))
    
    def compute_signatures():
        with Pool(processes=num_workers) as pool:
            return list(tqdm(pool.imap(minhash_worker, args_list, chunksize=optimal_chunksize), 
                           total=len(data), desc="Computing signatures"))

    results = timed_step("Processing (Signatures)", "Processing", compute_signatures, timings)

    # OPTIMIZATION: Batch insert into LSH for better performance
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    signatures = [None] * len(data)

    logger.info("Inserting signatures into LSH...")
    
    # Batch insertion with progress
    with tqdm(total=len(results), desc="Building LSH index") as pbar:
        for idx, mh in results:
            lsh.insert(str(idx), mh)
            signatures[idx] = mh
            pbar.update(1)

    # OPTIMIZATION: More efficient Union-Find with path compression
    logger.info("Finding duplicates with optimized LSH querying...")

    def filter_lsh():
        parent = list(range(len(data)))
        rank = [0] * len(data)

        # Optimized Union-Find with path compression and union by rank
        def find(u):
            if parent[u] != u:
                parent[u] = find(parent[u])  # Path compression
            return parent[u]

        def union(u, v):
            pu, pv = find(u), find(v)
            if pu != pv:
                # Union by rank
                if rank[pu] < rank[pv]:
                    parent[pu] = pv
                elif rank[pu] > rank[pv]:
                    parent[pv] = pu
                else:
                    parent[pv] = pu
                    rank[pu] += 1

        processed = set()

        # OPTIMIZATION: Process in batches to reduce overhead
        batch_size = 1000
        for start in tqdm(range(0, len(signatures), batch_size), desc="Querying LSH"):
            for i in range(start, min(start + batch_size, len(signatures))):
                if i in processed:
                    continue

                matches = lsh.query(signatures[i])
                match_indices = [int(m) for m in matches if int(m) != i]

                if match_indices:
                    for j in match_indices:
                        if find(i) != find(j):
                            union(i, j)
                    processed.update(match_indices)
                processed.add(i)

        # Collect unique representatives
        seen = set()
        output = []
        for i in range(len(data)):
            rep = find(i)
            if rep not in seen:
                seen.add(rep)
                output.append(data[i])

        return output

    deduped = timed_step("Filtering (LSH)", "Filtering", filter_lsh, timings)
    return deduped


# OPTIMIZED: Faster output writing
def save_output(records: List[Dict], output_path: str):
    logger.info(f"Saving {len(records):,} records to {output_path}")
    
    if output_path.endswith(".jsonl") or output_path.endswith(".json"):
        with open(output_path, "w") as f:
            for obj in records:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    elif output_path.endswith(".parquet"):
        # OPTIMIZATION: Use pyarrow directly for faster writing
        df = pd.DataFrame(records)
        df.to_parquet(output_path, index=False, engine='pyarrow', compression='snappy')
    else:
        raise ValueError("Unsupported output format")


def parse_args():
    parser = argparse.ArgumentParser(description="CPU fuzzy deduplication pipeline (fully optimized)")
    parser.add_argument("--input", required=True, help="Input path (dir, file, or glob)")
    parser.add_argument("--output", required=True, help="Output file (.jsonl or .parquet)")
    parser.add_argument("--text-column", default="content", help="Column for deduplication")
    parser.add_argument("--min-length", type=int, default=5, help="Minimum content length")
    parser.add_argument("--threshold", type=float, default=0.85, help="Jaccard threshold for MinHash LSH")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of input files")
    parser.add_argument("--num-cpus", type=int, default=None, help="Number of CPU cores to use")
    parser.add_argument("--num-perm", type=int, default=128, help="Number of permutations for MinHash (higher=more accurate)")
    parser.add_argument("--ngram-size", type=int, default=5, help="Size of token n-grams")
    return parser.parse_args()


def main():
    args = parse_args()
    timings = {}

    logger.info(f"Configuration:")
    logger.info(f"  - Threshold: {args.threshold}")
    logger.info(f"  - Num permutations: {args.num_perm}")
    logger.info(f"  - N-gram size: {args.ngram_size}")
    logger.info(f"  - Min length: {args.min_length}")
    logger.info(f"  - CPUs: {args.num_cpus or cpu_count()}")

    # Load data from disk with parallel reading
    data = timed_step("Loading", "Loading", 
                     lambda: load_input_files(args.input, args.text_column, args.max_files, args.min_length, args.num_cpus), 
                     timings)

    # Run MinHash + LSH-based deduplication
    deduped = fuzzy_deduplicate(data, args.text_column, 
                               threshold=args.threshold, 
                               num_perm=args.num_perm,
                               ngram_size=args.ngram_size,
                               num_cpus=args.num_cpus, 
                               timings=timings)

    # Save final deduplicated output
    timed_step("Saving", "Saving", lambda: save_output(deduped, args.output), timings)

    total_time = sum(timings.values())
    reduction = len(data) - len(deduped)
    reduction_pct = (reduction / len(data) * 100) if len(data) > 0 else 0

    logger.info("\n" + "=" * 20 + " FINAL SUMMARY " + "=" * 20)
    logger.info(f"{'Loading':<30}: {timings.get('Loading', 0):.2f}s")
    logger.info(f"{'Processing (Signatures)':<30}: {timings.get('Processing', 0):.2f}s")
    logger.info(f"{'Filtering (LSH)':<30}: {timings.get('Filtering', 0):.2f}s")
    logger.info(f"{'Saving':<30}: {timings.get('Saving', 0):.2f}s")
    logger.info(f"{'Total':<30}: {total_time:.2f}s")
    logger.info(f"{'Before':<30}: {len(data):,}")
    logger.info(f"{'After':<30}: {len(deduped):,}")
    logger.info(f"{'Removed':<30}: {reduction:,} ({reduction_pct:.1f}%)")


if __name__ == "__main__":
    main()

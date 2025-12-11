# Data Deduplication Toolkit

A comprehensive toolkit for exact and fuzzy-deduplication across different scales of data, from small datasets to terabyte-scale processing.

## Table of Contents

- [Acknowledgements](#acknowledgements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)

## Acknowledgements

This repository is heavily inspired by the following methods and code written in the following projects:

- [Text-Dedup](https://github.com/ChenghaoMou/text-dedup)
- [Nvidia-Nemo Curator](https://github.com/NVIDIA-NeMo/Curator)
- [Datasketch](https://github.com/ekzhu/datasketch) (MIT)
- [simhash-py](https://github.com/seomoz/simhash-py/tree/master/simhash) and [simhash-cpp](https://github.com/seomoz/simhash-cpp) (MIT)
- [Deduplicating Training Data Makes Language Models Better](https://github.com/google-research/deduplicate-text-datasets) (Apache 2.0)
- [Gaoya](https://github.com/serega/gaoya) (MIT)

## Installation

### Prerequisites

- Python 3.8+
- Java 11 (for Spark-based scripts)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### For Spark-based Scripts (exact_spark.py and fuzzy_spark2.py)

You'll also need Apache Spark. Install via:

```bash
# Option 1: Using pip
pip install pyspark[sql]

# Option 2: Download and set SPARK_HOME
# Download from https://spark.apache.org/downloads.html
export SPARK_HOME=/path/to/spark
export PATH=$SPARK_HOME/bin:$PATH
```
## Quick Start

### 1. Exact Deduplication 

```bash
python exact_cpu.py \
    --input-path "data/sample.parquet" \
    --output-path "output/dedup_exact" \
    --format parquet \
    --column text \
    --hash-func md5 \
    --num-proc 4
```
#### Argument Helper:  
*input-path*: Path to input dataset file or directory(jsonl or parquet)  
*output-path*: path to output deduplicated dataset  
*format*: output file format(parquet(default) or jsonl)  
*column*: which column to be deduplicated from the dataset  
*hash-func* : which hash function to use(md5(default), sha256)  

### 2. Spark Exact Deduplication

```bash
python exact_spark.py \
    --input-path "data/*.parquet" \
    --output-path "output/dedup_spark.parquet" \
    --dedup-column text \
    --executor-memory 8g \
    --driver-memory 4g
```
#### Argument Helper:  
*input-path* : Path to input dataset file or directory(jsonl or parquet)  
*output-path* : path to output deduplicated dataset   
*dedup-column* : which column to be deduplicated from the dataset  
*executor-memory* : the amount of memory to allot to each executor in the spark job
*driver-memory* : memory to allot to driver in the spark job

For more help on the argument, run the following command

```bash
python exact_spark.py --help
```
### 3. Spark Fuzzy Deduplication

```bash
python fuzzy_spark2.py \
    --input "data/" \
    --output "output/dedup_minhash" \
    --threshold 0.8 \
    --column content \
    --driver_memory 16g \
    --executor_memory 32g \
    --num_executors 10
```
#### Argument Helper:  
*input-path* : Path to input dataset file or directory(jsonl or parquet)  
*output-path* : path to output deduplicated dataset   
*threshold* : threshold used to consider deduplicates 
*column* :  column name to be deduplicated from the dataset  
*executor-memory* : the amount of memory to allot to each executor in the spark job
*driver-memory* : memory to allot to driver in the spark job
*num_executors* : the total number of executors to allot for the job.

For more help on the argument, run the following command

```bash
python fuzzy_spark2.py --help
```

**For additional help on the spark configurations used, please refer to [spark-documentation](https://spark.apache.org/docs/latest/configuration.html) to get a better idea as to what each configuration caters to.**

## Usage Examples

## Exact Deduplication

```bash
#input(on a sample dataset with exact duplicates present)
python exact_cpu.py \
    --input-path "sample.parquet" \
    --output-path "dedup_output.parquet" \
    --format parquet \
    --column text \
    --hash-func md5 \
    --num-proc 4

#output
INFO     Loading                       : 1.57s
INFO     Processing                    : 40.34s
INFO     Filtering                     : 17.09s
INFO     Saving                        : 44.83s
INFO     Total                         : 103.94s
INFO     Before                        : 5000000
INFO     After                         : 2999835
```

```bash
#input(on around 250gb of codeparrot dataset)
python3 exact_cpu.py \
  --input-path "tmp_300/*.parquet" \
  --output-path Dedup/output_files \
  --format parquet \
  --column content \
  --num-proc 36

#output
INFO     Loading                       : 152.49s
INFO     Processing                    : 7130.26s
INFO     Filtering                     : 1.24s
INFO     Saving                        : 3343.76s
INFO     Total                         : 10627.92s
INFO     Before                        : 30662700
INFO     After                         : 30662700

No exact duplicates found, in this case we can do a modification that the data is not saved again, as it is same as the input data thus saving us time. 
```
-------------------------------------------------------
## Spark Exact Deduplication

```bash
#input(on a sample dataset with exact duplicates present)
python exact_spark.py \
    --input-path "sample.parquet" \
    --output-path "dedup_output_spark.parquet" \
    --dedup-column "text" \
    --executor-memory "4g" \
    --driver-memory "4g" \
    --executor-cores "1" \
    --shuffle-partitions "300"

#output
INFO     Loading                       : 0.41s                                  
INFO     Processing                    : 0.01s
INFO     Saving                        : 7.82s
INFO     Total                         : 14.92s
INFO     Before                        : 5000000
INFO     After                         : 2999835
```

```bash
#input(on around 250gb of codeparrot dataset)
python3 exact_spark.py \
    --input-path "tmp_300/" \
    --output-path "output_files_parquet" \
    --dedup-column "content" \
    --driver-memory "32g" \
    --executor-cores "36" \
    --shuffle-partitions "300"

#output
INFO     Loading                       : 5.42s
INFO     Processing                    : 0.01s
INFO     Saving                        : 622.72s
INFO     Total                         : 877.40s
INFO     Before                        : 30662700
INFO     After                         : 30662700
Since data is the same, again no exact duplicates found. Again in this case, can save time by not saving the data again.
```

---

#### Can see significant speedup using spark configuration as opposed to simply on the CPU.
#### On 27gb of data speedup is ~7 (104s/14s).
#### On 250gb of codeparrot data speedup is ~12 (10627/877).

-----

## Spark Fuzzy Deduplication

```bash
#input(on 250gb of codeparrot-dataset)
python3 fuzzy_spark2.py \
  --input tmp_300/ \
  --output output_files_parquet_fuzzy \
  --driver_memory 40g \
  --executor_memory 32g \
  --executor_cores 6 \
  --num_executors 6 \
  --shuffle_partitions 12288 \
  --default_parallelism 96 \
  --threshold 0.75 \
  --ngram_size 3 \
  --broadcast_threshold 100mb \
  --memory_fraction 0.8 \
  --max_result_size 8g \
  --extra_spark_conf "spark.sql.adaptive.enabled=true" \
  --extra_spark_conf "spark.sql.adaptive.coalescePartitions.enabled=true" \
  --extra_spark_conf "spark.sql.adaptive.skewJoin.enabled=true" \
  --extra_spark_conf "spark.sql.adaptive.advisoryPartitionSizeInBytes=256MB" \
  --extra_spark_conf "spark.serializer.objectStreamReset=100" \
  --extra_spark_conf "spark.sql.execution.arrow.maxRecordsPerBatch=20000"

#output
_main_ - INFO - Input documents:        30,631,486
_main_ - INFO - Output documents:       20,598,272
_main_ - INFO - Duplicates removed:     10,033,214
_main_ - INFO - Deduplication rate:     32.75%
_main_ - INFO - Documents retained:     67.25%
_main_ - INFO - Output location:        output_files_parquet_fuzzy
_main_ - INFO - Total processing time:  13357.65s (03:42:37)
_main_ - INFO - Processing rate:        2293 docs/sec
```
-----

## Fuzzy CPU Deduplication
```bash
#input(on 250gb of codeparrot-dataset)
python3 fuzzy_cpu.py \
        --input /*.parquet \
        --output test.parquet \
        --num-cpus 8 \
        --text-column content

#output
INFO     Loading                       	: 2752.08s
INFO     Processing (Signatures)        : 41966s  
INFO     Filtering  (LSH)               : 452.06s
INFO     Saving                        	: 960s
INFO     Total                        	: 43624.01s
INFO     Before                        	: 30662146
INFO     After                          : 22976056

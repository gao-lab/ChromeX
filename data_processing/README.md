# Data preprocessing

This directory contains the Snakemake workflows and scripts used to convert raw Hi-C and RNA-seq data into ChromeX model input files.

## Directory structure

```text
preprocessing/
├── README.md
├── pipelines/
│   ├── Snakefile_hic_pipe.py       # Hi-C FASTQ -> contact maps and 3D genome features
│   ├── Snakefile_rna_pipe.py       # RNA-seq FASTQ -> RSEM expression labels
│   └── Snakefile_feat_build.py     # features + labels -> {cell}_data.hdf5
├── reference/                      # reference files
├── scripts/                        # custom scripts for specialized tasks
└── tests/                          # test cases
```

## Reference files

In order to run the pipeline from the beginning, you need to prepare the `bwa` index (see https://github.com/lh3/bwa?tab=readme-ov-file#getting-started), `STAR` index (see https://github.com/alexdobin/STAR/blob/master/doc/STARmanual.pdf), `RSEM` reference sequences (see https://github.com/deweylab/RSEM?tab=readme-ov-file#i-preparing-reference-sequences), fasta files, GTF files, and chromosome sizes files. It is recommended to store the prepared reference files in the `./reference` directory.

## Example: A549 test workflow

For Hi-C and RNA-seq, the processing workflows are described below, **using the human cell line A673 as an illustrative example**. The file structure of `./tests` is as follows:

```
tests
├── download_fq.sh
└── A549
    ├── data
    │   └── config.yaml
    ├── Hi-C
    │   ├── config.yaml
    │   └── fastq
    └── RNA-seq
        ├── config.yaml
        └── fastq
```

### Prepare the FASTQ and Configuration Files

First, prepare the FASTQ files and the Snakemake configuration files.

You can use the script `./tests/download_fq.sh` to download A549 FASTQ files from [ENCODE](https://www.encodeproject.org/). For sequencing data hosted on the SRA website, tools such as `prefetch` and `fasterq-dump` can be used to obtain FASTQ files.

There are three configuration files:

- `./tests/A549/data/config.yaml`
- `./tests/A549/Hi-C/config.yaml`
- `./tests/A549/RNA-seq/config.yaml`

Please make sure to update these files with the correct paths to the required software and reference files, as well as the necessary sample information. **It is recommended to use absolute paths for all files and tools to avoid ambiguity and ensure reproducibility.**

### FASTQ naming rules

#### Hi-C FASTQ

The Hi-C workflow reads files from `Hi-C/fastq/` using this naming pattern:

```text
fastq/{cell}_{biorep}.{techrep}_R1.fastq.gz
fastq/{cell}_{biorep}.{techrep}_R2.fastq.gz
```

Examples:

```text
fastq/A549_rep1.0_R1.fastq.gz
fastq/A549_rep1.0_R2.fastq.gz
fastq/A549_rep1.1_R1.fastq.gz
fastq/A549_rep1.1_R2.fastq.gz
```

For single-end data, only the `R1` file is required. The workflow checks whether the corresponding `R2` file exists.

#### RNA-seq FASTQ

The RNA-seq workflow reads files from `RNA-seq/fastq/` using this naming pattern:

```text
fastq/{cell}_{biorep}_R1.fastq.gz
fastq/{cell}_{biorep}_R2.fastq.gz
```

Examples:

```text
fastq/A549_rep1_R1.fastq.gz
fastq/A549_rep1_R2.fastq.gz
fastq/A549_rep2_R1.fastq.gz
fastq/A549_rep2_R2.fastq.gz
```

For single-end data, only the `R1` file is required.

### Run data processing pipelines

Next, execute the Hi-C and RNA-seq data processing pipelines using the following commands:

```
snakemake -s ./pipelines/hic_processing.py --configfile ./tests/A549/Hi-C/config.yaml --use-conda
snakemake -s ./pipelines/rna_processing.py --configfile ./tests/A549/RNA-seq/config.yaml --use-conda
```

Snakemake supports cluster execution through the `--cluster` option. It is recommended to use this with a job scheduler such as SLURM to submit the pipeline to a computing cluster, especially for large datasets.

#### Convert to model input format

Last, use the following command to convert the feature and label files into the model's input format.

```
snakemake -s ./pipelines/Snakefile_feat_build.py --configfile ./tests/A549/data/config.yaml --use-conda
```

The final output, an HDF5 file for model input, will be saved in the `./tests/A549/data` directory. That HDF5 file is the direct input for the model scripts in `ChromeX/model_training/`.

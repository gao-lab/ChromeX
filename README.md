# ChromeX: Chromosome Architecture-informed Expression Modeling

Welcome! This repository contains code for our data processing workflow and ChromeX-related tasks, which are provided for reference.

## Repository structure

```
ChromeX
├── data_processing/         # data processing pipeline implemented in Snakemake
├── model_training/          # ChromeX model source code and associated files
├── feature_analysis/        # source code related to 3D pattern computation
├── env.yml                  # reproducible environment via conda
├── LICENSE
└── README.md
```

## Software

Our data processing workflow uses the following dependencies:

- snakemake 7.32.4
- sra-tools 3.1.1
- Juicer encode_tag_disk1
- Juicer 2.0
- Juicer tools 2.13.06
- BWA 0.7.18
- SAMtools 1.21
- STAR 2.5.1b
- trim-galore 0.6.10
- RSEM 1.2.31
- cooler 0.9.3
- cooltools 0.7.1
- pairix 0.3.8
- bgzip 1.22
- BEDtools 2.31.1
- OnTAD 1.4
- python 3.8 and its libraries:
  - numpy 1.24.4
  - pandas 2.0.3
  - scipy 1.10.1
  - numba 0.58.1
  - seaborn 0.13.2
  - h5py 1.14.3
  - xgboost 2.1.4
  - scikit-learn 1.3.2
  - joblib 1.4.2
  - igraph 0.11.8
  - leidenalg 0.10.2
  - shap 0.44.1
- R 4.3 and its packages:
  - tidyverse 2.0.0
  - rtracklayer 1.62.0
  - CALDER 2.0

Please follow the official documentation to install [Juicer](https://github.com/aidenlab/juicer), [Juicer tools](https://github.com/aidenlab/juicer/wiki/Download), [OnTAD](https://github.com/anlin00007/OnTAD), [RSEM](https://github.com/deweylab/RSEM) and [fpgrowth](https://borgelt.net/fpgrowth.html). All other software can be installed via a fully pre-specified environment.

```
conda env create -f env.yml
```

## Reference files

Genome reference files we used are listed below.

|Description|Assembly|Download link|
|-|-|-|
|fasta|GRCh38.p14|[link](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/GRCh38.p14.genome.fa.gz)|
|GTF|GRCh38.p14|[link](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz)|
|chromosome sizes|GRCh38.p14|[link](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/latest/hg38.chrom.sizes)|
|fasta|GRCm38.p6|[link](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M23/GRCm38.p6.genome.fa.gz)|
|GTF|GRCm38.p6|[link](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M23/gencode.vM23.annotation.gtf.gz)|
|chromosome sizes|GRCm38.p6|[link](https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/latest/mm10.chrom.sizes)|

## End-to-end workflow

### 1. Build model-ready HDF5 files

See `data_processing/README.md` for details.

In brief, edit the example config files under `data_processing/tests/A549/`, then run the Hi-C pipeline, RNA-seq pipeline and final feature-building pipeline:

```bash
snakemake -s data_processing/pipelines/Snakefile_hic_pipe.py --configfile data_processing/tests/A549/Hi-C/config.yaml --cores 32
snakemake -s data_processing/pipelines/Snakefile_rna_pipe.py --configfile data_processing/tests/A549/RNA-seq/config.yaml --cores 8
snakemake -s data_processing/pipelines/Snakefile_feat_build.py --configfile data_processing/tests/A549/data/config.yaml --cores 8
```

The final output is one HDF5 file per cell or dataset:

```text
data_processing/tests/A549/data/{cell}_data.hdf5
```

### 2. Train or apply ChromeX model

See `model_training/README.md` for full usage.

Example for ChromeX prediction:

```bash
python ChromeX.py \
  -P predict \
  -D files/human_train_val_set.csv \
  -I /path/to/prediction_hdf5_directory \
  -m files/human_ChromeX.joblib \
  -n human \
  -N human \
  -o results/human
```

### 3. Interpret model features

See `feature_analysis/README.md` for details.

The feature-analysis section includes code for extracting XGBoost decision nodes, computing gene-node SHAP matrices, clustering genes by Leiden algorithm and identifying cluster-specific differential decision rules.

## Notes

It is recommended to use absolute paths for all files and tools to avoid ambiguity and ensure reproducibility.

## Citation

In coming.

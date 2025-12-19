# DBAT: Deep learning-based Bio-sequencing Analysis Toolkit
**Version 0.2.0**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Linux-lightgrey)]()

## Description

DBAT (Deep learning-based Bio-sequencing Analysis Toolkit) is a versatile toolkit for biological sequencing data analysis, consisting of two deep learning modules:

- **deepPeak**: A deep learning framework to predict sequencing tracks and enable cross-species prediction of mutant profiles.

- **deepLoci**: A CNN model to predict featured genomic loci based on omics data.

DBAT has been tested on a Linux system (Ubuntu 24.04.3, GCC 13.3.0, CUDA 12.9, Python 3.12.12)

## Authors and Contact

- **Authors**: Xuan Ma, Bochao Liu  
- **Email**: [skyxma@tjnu.edu.cn](mailto:skyxma@tjnu.edu.cn)

## Installation

### Step 1: Set up a Conda environment (recommended)
```bash
conda create -n dbat python=3.9
conda activate dbat
```

### Step 2: Install DBAT
```bash
cd dbat
pip install -e .
```

## Test Data

The test/ directory contains:

| File             | Description |
|------------------|-------------|
| ath.fa           | *Arabidopsis thaliana* Chr2, Chr3, Chr4 sequences |
| osa.fa           | Rice (*Oryza sativa*) Chr10 sequence |
| ath_h3k4.bw      | *Arabidopsis* H3K4me3 ChIP-seq (SRR3709930) |
| ath_h3k9.bw      | *Arabidopsis* H3K9me2 ChIP-seq (GSM5574912) |
| ath_wt.bw        | *Arabidopsis* WT sRNA-seq (GSE179796) |
| ath_ddm.bw       | *Arabidopsis* ddm1 mutant sRNA-seq (GSE179796) |
| osa_wt.bw        | Rice WT sRNA-seq |
| ath_omics.csv    | Integrated *Arabidopsis* omics data |

## Implementation

Navigate to the test directory:
```bash
cd test/
```

### 1. deepPeak predicts sequencing peaks (normal mode)

#### Train on Arabidopsis H3K4me3:
```bash
dbat deepPeak --mode norm \
              --action train \
              --genome data/ath.fa \
              --input data/ath_h3k4.bw \
              --output out1
```

#### Predict rice H3K4me3 track:
```bash
dbat deepPeak --mode norm \
              --action prediction \
              --model out1/Peak_model.pth \
              --stats out1/Peak_stats.npz \
              --genome data/osa.fa \
              --output out1
```

### 2. deepPeak conducts cross-species prediction of mutant profiles

#### Train on Arabidopsis WT and mutant:
```bash
dbat deepPeak --mode cross_spe \
              --action train \
              --genome data/ath.fa \
              --ref_wt data/ath_wt.bw \
              --ref_mut data/ath_ddm.bw \
              --output out2
```

#### Predict rice mutant profile:
```bash
dbat deepPeak --mode cross_spe \
              --action prediction \
              --model out2/Ref_model.pth \
              --stats out2/Ref_stats.npz \
              --genome data/osa.fa \
              --targ_wt data/osa_wt.bw \
              --output out2
```

### 3. deepLoci predicts featured genomic loci
```bash
dbat deepLoci --input data/ath_omics.csv --output out3
```

## Additional Notes

1. deepPeak allows to input suppressive data:
```bash
dbat deepPeak --mode norm \
              --action train \
              --genome data/ath.fa \
              --input data/ath_h3k4.bw \
              --suppress data/ath_h3k9.bw \
              --output out2
```

2. Training parameters are adjustable: Modify settings (e.g. batch size) in config/cfg.yml as needed.

3. deepLoci input format: The input CSV must include a header with columns as: ID,feature,data,sample,pos1,pos2,...

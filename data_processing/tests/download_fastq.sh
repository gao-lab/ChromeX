#!/bin/bash

# Download Hi-C FASTQ files
mkdir -p ./A549/Hi-C/fastq
## rep1
wget -c -O ./A549/Hi-C/fastq/A549_rep1.0_R1.fastq.gz https://www.encodeproject.org/files/ENCFF039FYU/@@download/ENCFF039FYU.fastq.gz
wget -c -O ./A549/Hi-C/fastq/A549_rep1.0_R2.fastq.gz https://www.encodeproject.org/files/ENCFF479RSE/@@download/ENCFF479RSE.fastq.gz
wget -c -O ./A549/Hi-C/fastq/A549_rep1.1_R1.fastq.gz https://www.encodeproject.org/files/ENCFF473XVG/@@download/ENCFF473XVG.fastq.gz
wget -c -O ./A549/Hi-C/fastq/A549_rep1.1_R2.fastq.gz https://www.encodeproject.org/files/ENCFF706REZ/@@download/ENCFF706REZ.fastq.gz
## rep2
wget -c -O ./A549/Hi-C/fastq/A549_rep2.0_R1.fastq.gz https://www.encodeproject.org/files/ENCFF622FZJ/@@download/ENCFF622FZJ.fastq.gz
wget -c -O ./A549/Hi-C/fastq/A549_rep2.0_R2.fastq.gz https://www.encodeproject.org/files/ENCFF796FXG/@@download/ENCFF796FXG.fastq.gz
wget -c -O ./A549/Hi-C/fastq/A549_rep2.1_R1.fastq.gz https://www.encodeproject.org/files/ENCFF231WIS/@@download/ENCFF231WIS.fastq.gz
wget -c -O ./A549/Hi-C/fastq/A549_rep2.1_R2.fastq.gz https://www.encodeproject.org/files/ENCFF479TQN/@@download/ENCFF479TQN.fastq.gz
## rep3
wget -c -O ./A549/Hi-C/fastq/A549_rep3.0_R1.fastq.gz https://www.encodeproject.org/files/ENCFF935APO/@@download/ENCFF935APO.fastq.gz
wget -c -O ./A549/Hi-C/fastq/A549_rep3.0_R2.fastq.gz https://www.encodeproject.org/files/ENCFF470IOJ/@@download/ENCFF470IOJ.fastq.gz
wget -c -O ./A549/Hi-C/fastq/A549_rep3.1_R1.fastq.gz https://www.encodeproject.org/files/ENCFF998HPF/@@download/ENCFF998HPF.fastq.gz
wget -c -O ./A549/Hi-C/fastq/A549_rep3.1_R2.fastq.gz https://www.encodeproject.org/files/ENCFF720PMQ/@@download/ENCFF720PMQ.fastq.gz
## rep4
wget -c -O ./A549/Hi-C/fastq/A549_rep4.0_R1.fastq.gz https://www.encodeproject.org/files/ENCFF456YXR/@@download/ENCFF456YXR.fastq.gz
wget -c -O ./A549/Hi-C/fastq/A549_rep4.0_R2.fastq.gz https://www.encodeproject.org/files/ENCFF862MJI/@@download/ENCFF862MJI.fastq.gz
wget -c -O ./A549/Hi-C/fastq/A549_rep4.1_R1.fastq.gz https://www.encodeproject.org/files/ENCFF520CXI/@@download/ENCFF520CXI.fastq.gz
wget -c -O ./A549/Hi-C/fastq/A549_rep4.1_R2.fastq.gz https://www.encodeproject.org/files/ENCFF834ELO/@@download/ENCFF834ELO.fastq.gz

# Download RNA-seq FASTQ files
mkdir -p ./A549/RNA-seq/fastq
## rep1
wget -c -O ./A549/RNA-seq/fastq/A549_rep1_R1.fastq.gz https://www.encodeproject.org/files/ENCFF000EJJ/@@download/ENCFF000EJJ.fastq.gz
wget -c -O ./A549/RNA-seq/fastq/A549_rep1_R2.fastq.gz https://www.encodeproject.org/files/ENCFF000EJV/@@download/ENCFF000EJV.fastq.gz
## rep2
wget -c -O ./A549/RNA-seq/fastq/A549_rep2_R1.fastq.gz https://www.encodeproject.org/files/ENCFF000EJW/@@download/ENCFF000EJW.fastq.gz
wget -c -O ./A549/RNA-seq/fastq/A549_rep2_R2.fastq.gz https://www.encodeproject.org/files/ENCFF000EKB/@@download/ENCFF000EKB.fastq.gz

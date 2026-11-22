#!/bin/bash

# Parsing command-line arguments
while getopts ":i:O:g:r:" opt; do
    case ${opt} in
        i) input_file="$OPTARG" ;;
        O) subcompart_output_prefix="$OPTARG" ;;
        g) genomeID="$OPTARG" ;;
        r) resolution="$OPTARG" ;;
        \?) echo "Invalid option: $OPTARG" 1>&2 ;;
    esac
done

if [[ $genomeID == "hg38" || $genomeID == "hg19" ]]; then
    chrom=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX)
elif [[ $genomeID == "mm10" || $genomeID == "mm9" ]]; then
    chrom=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chrX)
fi

for chr in "${chrom[@]}"; do
    awk -v chr="$chr" '$1 == chr' $input_file | cut -f2,5,8 | gzip -f > ${subcompart_output_prefix}_${chr}_${resolution}_ob_for_Calder.txt.gz &
done

wait

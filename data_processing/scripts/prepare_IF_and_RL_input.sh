#!/bin/bash

# Parsing command-line arguments
while getopts ":i:o:r:b:B:" opt; do
    case ${opt} in
        i) input="$OPTARG" ;;
        o) output="$OPTARG" ;;
        r) resolution="$OPTARG" ;;
        b) ref_bed1="$OPTARG" ;;
        B) ref_bed2="$OPTARG" ;;
        \?) echo "Invalid option: $OPTARG" 1>&2 ;;
    esac
done

Rscript_dir=$(dirname "$0")

# Define max level
max_level=8
awk 'BEGIN {OFS="\t"} {if ($4 >'${max_level}') $4 ='${max_level}'; print}' ${input} > ${output}_tad_region_${resolution}.tmp.bed

# Process overlap regions among different levels
for ((i = 1; i <= 20; i++)); do
    awk '{OFS="\t"; if($4=="'${i}'") print $0}' ${output}_tad_region_${resolution}.tmp.bed > ${output}_level${i}_${resolution}.tmp.bed
    awk '{OFS="\t"; if($4=="'${i}'") print $0}' ${input} > ${output}_level${i}_origin.tmp.bed
done

> ${output}_TAD_region_${resolution}_tmp.bed
> ${output}_TAD_origin_region_tmp.bed

for ((i = 1; i < 20; i++)); do
    j=$((i + 1))
    bedtools subtract -a ${output}_level${i}_${resolution}.tmp.bed -b ${output}_level${j}_${resolution}.tmp.bed >> ${output}_TAD_region_${resolution}_tmp.bed
    bedtools subtract -a ${output}_level${i}_origin.tmp.bed -b ${output}_level${j}_origin.tmp.bed >> ${output}_TAD_origin_region_tmp.bed
done

cat ${output}_TAD_region_${resolution}_tmp.bed ${output}_level8_${resolution}.tmp.bed | sort -k 1,1V -k 2,2n > ${output}_TAD_regions_${resolution}.tmp.bed
bedtools intersect -sorted -loj -a ${ref_bed1} -b ${output}_TAD_regions_${resolution}.tmp.bed | awk '{OFS="\t"; if($4!=".") print $1,$2,$3,$7,$8,$9}' | sort -k 1,1V -k 2,2n | gzip -f > ${output}_TADs_${resolution}.bed.gz
cat ${output}_TAD_origin_region_tmp.bed | sort -k 1,1V -k 2,2n | bedtools intersect -sorted -loj -a ${ref_bed1} -b - | awk '{OFS="\t"; if($4!=".") print $1,$2,$3,$8; else print $1,$2,$3,0}' > ${output}_if.tmp.bed

Rscript ${Rscript_dir}/prepare_interaction_frequency_input.R -r ${resolution} -i ${output}_if.tmp.bed -o ${output}_interaction_frequency_input_${resolution}.txt -b ${ref_bed2}

# Map TAD data to regions of gene TSS
bedtools intersect -sorted -loj -a ${ref_bed2} -b ${output}_TADs_${resolution}.bed.gz | awk '{OFS="\t"; if($7!=".") print $4,$5,$6,$10; else print $4,$5,$6,0}' | sort -k1,1V -k2,2V -k3,3 -k4,4nr | awk '!seen[$1,$2,$3]++' | gzip -f > ${output}_TSS_flanking1M_TAD_region_level_long.txt.gz

# Convert to model input
Rscript ${Rscript_dir}/prepare_region_level_input.R -r ${resolution} -i ${output}_TSS_flanking1M_TAD_region_level_long.txt.gz -o ${output}_region_level_input_${resolution}.txt &

wait

rm ${output}_*.tmp.bed ${output}*long.txt.gz

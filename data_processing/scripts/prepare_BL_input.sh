#!/bin/bashref_bed1

# Parsing command-line arguments
while getopts ":i:o:r:B:" opt; do
    case ${opt} in
        i) input="$OPTARG" ;;
        o) output="$OPTARG" ;;
        r) resolution="$OPTARG" ;;
        B) ref_bed2="$OPTARG" ;;
        \?) echo "Invalid option: $OPTARG" 1>&2 ;;
    esac
done

Rscript_dir=$(dirname "$0")

# Define max level
max_level=8
awk 'BEGIN {OFS="\t"} {if ($4 >'${max_level}') $4 ='${max_level}'; print}' ${input} > ${output}_tad_region_${resolution}.tmp.bed

# Process overlap boundaries among different levels
awk '{OFS="\t"; print $1,$2}' ${output}_tad_region_${resolution}.tmp.bed | sort | uniq -c | awk '{OFS="\t"; print $2,$3,$3+"'${resolution}'",$1}' > ${output}_tad_${resolution}_tmp1.bed
awk '{OFS="\t"; print $1,$3}' ${output}_tad_region_${resolution}.tmp.bed | sort | uniq -c | awk '{OFS="\t"; print $2,$3,$3+"'${resolution}'",$1}' > ${output}_tad_${resolution}_tmp2.bed
cat ${output}_tad_${resolution}_tmp1.bed ${output}_tad_${resolution}_tmp2.bed | sort -k 1,1V -k 2,2n | bedtools groupby -g 1,2,3 -c 4 -o max | gzip -f > ${output}_TAD_boundaries.bed.gz

# Map TAD data to regions of gene TSS
bedtools intersect -sorted -loj -a ${ref_bed2} -b ${output}_TAD_boundaries.bed.gz | awk '{OFS="\t"; if($7!=".") print $4,$5,$6,$10; else print $4,$5,$6,0}' | sort -k 1,1V -k 3,3n | gzip -f > ${output}_TSS_flanking1M_TAD_boundary_level_long.txt.gz

# Convert to model input
Rscript ${Rscript_dir}/prepare_boundary_level_input.R -i ${output}_TSS_flanking1M_TAD_boundary_level_long.txt.gz -o ${output}_boundary_level_input_${resolution}.txt -r ${resolution} &

wait

rm ${output}_*tmp* ${output}*TSS_flanking1M_TAD_boundary_level_long.txt.gz

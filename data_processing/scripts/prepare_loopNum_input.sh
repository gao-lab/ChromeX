#!/bin/bash

# Parsing command-line arguments
while getopts ":i:o:r:b:B:" opt; do
    case ${opt} in
        i) input="$OPTARG" ;;
        o) output_loop="$OPTARG" ;;
        r) resolution="$OPTARG" ;;
	b) ref_bed1="$OPTARG" ;;
        B) ref_bed2="$OPTARG" ;;
        \?) echo "Invalid option: $OPTARG" 1>&2 ;;
    esac
done

line_count=$(wc -l < ${input})
if [ "$line_count" -lt 101 ]; then
    exit 0
fi

Rscript_dir=$(dirname "$0")

sed '1d' ${input} | awk '{OFS="\t"; print $1,$2,$3,$4,$5,$6}' | sort -k 1,1 -k 2,2n -k 3,3n -k 4,4 -k 5,5n -k 6,6n | bedtools pairtobed -a stdin -b ${ref_bed1} | awk '{OFS="\t"; print $7,$8,$9}' > ${input}.loop.tmp.txt

Rscript ${Rscript_dir}/prepare_loopNum_input.R -i ${input}.loop.tmp.txt -o ${output_loop} -b ${ref_bed1} -B ${ref_bed2} -r ${resolution}

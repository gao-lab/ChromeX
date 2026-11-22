#!/bin/bash

# Parsing command-line arguments
while getopts ":i:o:g:r:s:" opt; do
    case ${opt} in
        i) input_prefix="$OPTARG" ;;
        o) output_bed="$OPTARG" ;;
        g) genomeID="$OPTARG" ;;
        r) resolution="$OPTARG" ;;
        s) onTAD="$OPTARG" ;;
        \?) echo "Invalid option: $OPTARG" 1>&2 ;;
    esac
done

if [[ "$genomeID" == "hg38" ]]; then
    declare -A chr_lengths=(
        ["chr1"]=248956422
        ["chr2"]=242193529
        ["chr3"]=198295559
        ["chr4"]=190214555
        ["chr5"]=181538259
        ["chr6"]=170805979
        ["chr7"]=159345973
        ["chr8"]=145138636
        ["chr9"]=138394717
        ["chr10"]=133797422
        ["chr11"]=135086622
        ["chr12"]=133275309
        ["chr13"]=114364328
        ["chr14"]=107043718
        ["chr15"]=101991189
        ["chr16"]=90338345
        ["chr17"]=83257441
        ["chr18"]=80373285
        ["chr19"]=58617616
        ["chr20"]=64444167
        ["chr21"]=46709983
        ["chr22"]=50818468
        ["chrX"]=156040895
    )
elif [[ "$genomeID" == "mm10" ]]; then
    declare -A chr_lengths=(
        ["chr1"]=195471971
        ["chr2"]=182113224
        ["chr3"]=160039680
        ["chr4"]=156508116
        ["chr5"]=151834684
        ["chr6"]=149736546
        ["chr7"]=145441459
        ["chr8"]=129401213
        ["chr9"]=124595110
        ["chr10"]=130694993
        ["chr11"]=122082543
        ["chr12"]=120129022
        ["chr13"]=120421639
        ["chr14"]=124902244
        ["chr15"]=104043685
        ["chr16"]=98207768
        ["chr17"]=94987271
        ["chr18"]=90702639
        ["chr19"]=61431566
        ["chrX"]=171031299
    )
fi

if [ "$resolution" -eq 10000 ]; then
    maxsz=200
    minsz=3
elif [ "$resolution" -eq 25000 ]; then
    maxsz=80
    minsz=1
elif [ "$resolution" -eq 40000 ]; then
    maxsz=50
    minsz=1
elif [ "$resolution" -eq 50000 ]; then
    maxsz=40
    minsz=1
elif [ "$resolution" -eq 5000 ]; then
    maxsz=400
    minsz=6
else
    echo "Invalid resolution."
    exit 1
fi

for chr in "${!chr_lengths[@]}"; do
    length=${chr_lengths[$chr]}
    $onTAD ${input_prefix}_${chr}.matrix -maxsz ${maxsz} -minsz ${minsz} -bedout "$chr" "$length" $resolution -o ${input_prefix}_tads_"$chr"_${resolution}
done

> ${output_bed}
for chr in "${!chr_lengths[@]}"; do
    awk '{OFS="\t"; if($3>0) print "'${chr}'",($1-1)*"'${resolution}'",($2-1)*"'${resolution}'",$3,$4,$5}' ${input_prefix}_tads_"$chr"_${resolution}.tad >> ${output_bed}
done

sort -k1,1V -k2,2n ${output_bed} > ${output_bed}_tmp && mv ${output_bed}_tmp ${output_bed}

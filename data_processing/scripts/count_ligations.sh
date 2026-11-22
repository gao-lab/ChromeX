#!/bin/bash

export LC_ALL=C
export LC_COLLATE=C

# Parsing command-line arguments
while getopts ":i:I:o:l:u:" opt; do
    case ${opt} in
        i) input1="$OPTARG" ;;
        I) input2="$OPTARG" ;;
        o) output="$OPTARG" ;;
        l) ligation="$OPTARG" ;;
        u) singleend="$OPTARG" ;;
        \?) echo "Invalid option: $OPTARG" 1>&2 ;;
    esac
done

if [ "$ligation" = "XXXX" ]; then
    num1="0"
else
    if [ $singleend -eq 1 ]; then
        num1=$(paste <(gunzip -c $input1) | awk '!((NR+2)%4)' | grep -cE $ligation)
    else
        num1=$(paste <(gunzip -c $input1) <(gunzip -c $input2) | awk '!((NR+2)%4)' | grep -cE $ligation)
    fi
fi
echo -ne "$num1 " > ${output}_norm.txt.res.txt

import numpy as np
import pandas as pd
import cooler
import os
import glob

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, help="Input cool file", dest="cool")
parser.add_argument("-o", "--output", type=str, help="Output matrix", dest="output")
parser.add_argument("-g", "--genomeID", type=str, help="Genome ID", dest="genomeID")

args = parser.parse_args()
cool_file = args.cool
output = args.output
genomeID = args.genomeID

if genomeID == "hg38" or genomeID == "hg19":
    chroms = [
        "chr1",
        "chr2",
        "chr3",
        "chr4",
        "chr5",
        "chr6",
        "chr7",
        "chr8",
        "chr9",
        "chr10",
        "chr11",
        "chr12",
        "chr13",
        "chr14",
        "chr15",
        "chr16",
        "chr17",
        "chr18",
        "chr19",
        "chr20",
        "chr21",
        "chr22",
        "chrX",
    ]
elif genomeID == "mm10" or genomeID == "mm9":
    chroms = [
        "chr1",
        "chr2",
        "chr3",
        "chr4",
        "chr5",
        "chr6",
        "chr7",
        "chr8",
        "chr9",
        "chr10",
        "chr11",
        "chr12",
        "chr13",
        "chr14",
        "chr15",
        "chr16",
        "chr17",
        "chr18",
        "chr19",
        "chrX",
    ]
else:
    raise ValueError("Invalid genome ID.")

clr = cooler.Cooler(cool_file)
name = os.path.basename(cool_file).rsplit("_", 1)[0]

for chr in chroms:
    matrix = clr.matrix(balance=True).fetch(chr)
    matrix = np.nan_to_num(matrix)
    np.savetxt(output + f"_{chr}.matrix", matrix, fmt="%g", delimiter="\t")

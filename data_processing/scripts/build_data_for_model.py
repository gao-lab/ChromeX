import sys
import h5py
import os
import math
import pandas as pd
import numpy as np
from scipy.stats import mode
import argparse


parser = argparse.ArgumentParser(
    formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
        prog, max_help_position=40, width=100
    )
)
parser.add_argument(
    "-b",
    "--bed",
    type=str,
    required=True,
    help="Genes bed file. [required]",
    dest="gene_bed",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    required=True,
    help="Output file. [required]",
    dest="output",
)
parser.add_argument(
    "-r", "--rna", type=str, required=True, help="RNA data file. [required]", dest="rna"
)
parser.add_argument(
    "-f",
    "--feature",
    type=str,
    required=True,
    help="Feature input path. [required]",
    dest="feature",
)


def sort_df_order(df, column, order_list):
    df[column] = pd.Categorical(df[column], categories=order_list, ordered=True)
    df = df.sort_values(by=column)
    return df


def get_gene_length(gene_df, genelist):
    gene_len_df = pd.DataFrame(
        {"0": gene_df["gene_name"], "Difference": gene_df["end"] - gene_df["start"] + 1}
    )
    gene_len_df["group"] = np.ceil(gene_len_df["Difference"] / 10000).astype(int)
    gene_length_group = [
        gene_len_df.loc[gene_len_df["0"] == x, "group"].values[0] for x in genelist
    ]
    gene_length = [
        gene_len_df.loc[gene_len_df["0"] == x, "Difference"].values[0] for x in genelist
    ]
    return gene_length, gene_length_group


def build_featMat(path, gene_order, resolution):
    if not os.path.isfile(path):
        if resolution == 10000:
            return np.zeros((len(gene_order), 201))
        if resolution == 50000:
            return np.zeros((len(gene_order), 41))
    df = pd.read_csv(path, sep="\t", header=None)
    if resolution == 10000:
        df.columns = ["gene"] + list(range(1, 202))
    if resolution == 50000:
        df.columns = ["gene"] + list(range(1, 42))
    df_filtered = df[df["gene"].isin(gene_order)]
    #
    missing_genes = [
        gene for gene in gene_order if gene not in df_filtered["gene"].values
    ]
    for gene in missing_genes:
        zero_row = pd.DataFrame(
            [[gene] + [0] * (df_filtered.shape[1] - 1)], columns=df_filtered.columns
        )
        df_filtered = pd.concat([df_filtered, zero_row], ignore_index=True)
    #
    df_sorted = df_filtered.set_index("gene").reindex(gene_order).reset_index()
    array = df_sorted.iloc[:, 1:].values
    return array


def build_genebodyAB(ABmatrix, gene_bed, gene_order):
    level_num = 8
    genelenlist, _ = get_gene_length(gene_bed, gene_order)
    genebodyAB = np.zeros([len(gene_order), level_num])
    for i in range(len(genelenlist)):
        genelen = math.ceil(genelenlist[i] / 10000)
        tem = ABmatrix[i, 100 : (100 + genelen)]
        counts = np.bincount(tem, minlength=9)
        if len(tem) - counts[0] == 0:
            genebody_ab = [0, 0, 0, 0, 0, 0, 0, 0]
        else:
            genebody_ab = counts[1:9] / (len(tem) - counts[0])
        genebodyAB[i,] = genebody_ab
    return genebodyAB


def build_RNAseq(RNA_path, gene_bed):
    gene_order = list(gene_bed["gene_name"])
    #
    rna_df = pd.read_csv(RNA_path, sep="\t", header=None)
    rna_df.columns = ["gene_name", "FPKM"]
    #
    gene_order = list(set(gene_order).intersection(set(rna_df["gene_name"])))
    gene_order = sorted(gene_order)
    rna_df = rna_df[rna_df["gene_name"].isin(gene_order)]
    gene_df = pd.DataFrame({"gene_name": gene_order})
    df_sorted = gene_df.merge(rna_df, on="gene_name", how="left")
    label = np.asarray(list(df_sorted["FPKM"]))
    label = np.nan_to_num(label, nan=0.0, posinf=0.0, neginf=0.0)
    return label, gene_order


def build_Compartment(path10, path50, gene_bed, gene_order):
    df = pd.read_csv(path10, sep="\t", header=None)
    df.columns = ["gene_name", "subComp"]
    char_to_num = {
        "N": 0,
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 4,
        "e": 5,
        "f": 6,
        "g": 7,
        "h": 8,
    }
    df["subComp"] = df["subComp"].apply(lambda x: [char_to_num[ch] for ch in x])

    df = df[df["gene_name"].isin(gene_order)]
    df = df.drop_duplicates(subset="gene_name", keep="first")
    df = sort_df_order(df, "gene_name", gene_order)

    ABmatrix10 = np.vstack(df["subComp"].values)
    genebodyAB = build_genebodyAB(ABmatrix10, gene_bed, gene_order)

    df = pd.read_csv(path50, sep="\t", header=None)
    df.columns = ["gene_name", "subComp"]
    char_to_num = {
        "N": 0,
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 4,
        "e": 5,
        "f": 6,
        "g": 7,
        "h": 8,
    }
    df["subComp"] = df["subComp"].apply(lambda x: [char_to_num[ch] for ch in x])

    df = df[df["gene_name"].isin(gene_order)]
    df = df.drop_duplicates(subset="gene_name", keep="first")
    df = sort_df_order(df, "gene_name", gene_order)

    ABmatrix50 = np.vstack(df["subComp"].values)
    return ABmatrix50, ABmatrix10, genebodyAB


def build_Loop(loop_path, gene_order):
    loop = build_featMat(loop_path, gene_order, 10000)
    return loop


def build_TAD(BL_path, RL_path, IF_path, gene_order):
    boundary_level = build_featMat(BL_path, gene_order, 10000)
    region_level = build_featMat(RL_path, gene_order, 50000)
    interaction_frequency = build_featMat(IF_path, gene_order, 50000)
    return boundary_level, region_level, interaction_frequency


def save_hdf5(
    output,
    gene_bed_file,
    rna_file,
    compart10_file,
    compart50_file,
    loop_file,
    BL_file,
    RL_file,
    IF_file,
):
    gene_bed = pd.read_csv(gene_bed_file, sep="\t", header=None)
    gene_bed.columns = ["chromosome", "start", "end", "strand", "gene_name"]
    Y, gene_order = build_RNAseq(rna_file, gene_bed)
    _, gene_len = get_gene_length(gene_bed, gene_order)
    AB, AB_pre, genebodyAB = build_Compartment(
        compart10_file, compart50_file, gene_bed, gene_order
    )
    loop = build_Loop(loop_file, gene_order)
    boundary_level, region_level, interaction_frequency = build_TAD(
        BL_file, RL_file, IF_file, gene_order
    )

    dt = h5py.special_dtype(vlen=str)
    with h5py.File(output, "w") as f:
        f.create_dataset("gene_order", data=gene_order, dtype=dt)
        f.create_dataset("Y", data=Y)
        f.create_dataset("X_gene_len", data=gene_len)
        f.create_dataset("X_AB", data=AB)
        f.create_dataset("X_AB_pre", data=AB_pre)
        f.create_dataset("X_genebodyAB", data=genebodyAB)
        f.create_dataset("X_loop", data=loop)
        f.create_dataset("X_boundary_level", data=boundary_level)
        f.create_dataset("X_region_level", data=region_level)
        f.create_dataset("X_interaction_frequency", data=interaction_frequency)


if __name__ == "__main__":
    args = parser.parse_args()
    gene_bed_file = args.gene_bed
    output = args.output
    rna_file = args.rna
    feature_path = args.feature

    compart10_file = f"{feature_path}_subcompartment_input_10000.txt"
    compart50_file = f"{feature_path}_subcompartment_input_50000.txt"
    loop_file = f"{feature_path}_loop_input_10000.txt"
    BL_file = f"{feature_path}_boundary_level_input_10000.txt"
    RL_file = f"{feature_path}_region_level_input_50000.txt"
    IF_file = f"{feature_path}_interaction_frequency_input_50000.txt"

    save_hdf5(
        output,
        gene_bed_file,
        rna_file,
        compart10_file,
        compart50_file,
        loop_file,
        BL_file,
        RL_file,
        IF_file,
    )

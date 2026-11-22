#!/usr/bin/env python3
"""Cluster genes from repeated assignment files via a co-association graph."""

import argparse
import logging
import os

import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
from scipy import sparse as sp


logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog,
            max_help_position=40,
            width=100,
        )
    )
    parser.add_argument(
        "--assignments",
        required=True,
        nargs="+",
        help="",
        default=[],
        dest="assignments",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Result path. [required]",
        dest="output",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        required=True,
        help="Leiden resolution. [required]",
        dest="resolution",
    )
    return parser.parse_args()


def _read_assignments(path):
    df = pd.read_csv(path)
    df = df[["gene", "cluster"]].copy()
    df["gene"] = df["gene"].astype(str)
    cats = df["cluster"].astype("category")
    df["cluster_code"] = cats.cat.codes.astype(np.int32)
    return df[["gene", "cluster_code"]]


def _build_signature_matrix(per_run, genes):
    n_genes = len(genes)
    n_runs = len(per_run)
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
    signature_matrix = np.full((n_runs, n_genes), -1, dtype=np.int32)

    for run_idx, df in enumerate(per_run):
        idx = df["gene"].map(gene_to_idx)
        mask = idx.notna().to_numpy()
        if not mask.any():
            continue
        gene_idx = idx[mask].astype(np.int32).to_numpy()
        codes = df.loc[mask, "cluster_code"].astype(np.int32).to_numpy()
        signature_matrix[run_idx, gene_idx] = codes
        logging.info("run %s/%s: filled %s entries", run_idx + 1, n_runs, mask.sum())
    return signature_matrix


def _group_by_signature(signature_matrix):
    n_genes = signature_matrix.shape[1]
    signatures = signature_matrix.T.copy()
    view_dtype = np.dtype(
        [("r%d" % i, signatures.dtype) for i in range(signatures.shape[1])]
    )
    view = signatures.view(view_dtype).reshape(n_genes)
    order = np.argsort(view, kind="mergesort")
    signatures_sorted = signatures[order]

    groups = []
    start = 0
    for idx in range(1, n_genes + 1):
        if idx == n_genes or not np.array_equal(
            signatures_sorted[idx], signatures_sorted[start]
        ):
            if idx - start >= 2:
                sorted_idx = np.arange(start, idx, dtype=np.int32)
                groups.append(order[sorted_idx])
            start = idx
    return groups


def _iter_upper_pairs(ids):
    """Yield all i<j pairs from a one-dimensional integer array of indices."""
    n_ids = ids.size
    for first in range(n_ids - 1):
        left = int(ids[first])
        for second in range(first + 1, n_ids):
            yield left, int(ids[second])


def build_coassoc_matrix(groups, n_genes):
    rows = []
    cols = []
    data = []

    for idx in range(n_genes):
        rows.append(idx)
        cols.append(idx)
        data.append(1.0)

    for group_idx in groups:
        if group_idx.size < 2:
            continue
        for i, j in _iter_upper_pairs(group_idx):
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([1.0, 1.0])

    return sp.csr_matrix(
        (data, (rows, cols)), shape=(n_genes, n_genes), dtype=np.float32
    )


def generate_pairs_df(groups, genes):
    edge_count = 0
    pairs_data = []

    for group_idx in groups:
        if group_idx.size < 2:
            continue
        for i, j in _iter_upper_pairs(group_idx):
            pairs_data.append([genes[i], genes[j]])
            edge_count += 1

    return pd.DataFrame(pairs_data, columns=["gene_i", "gene_j"]), edge_count


def run(args):
    assignment_paths = sorted(dict.fromkeys(args.assignments))
    logging.info("n_runs=%s", len(assignment_paths))

    per_run = [_read_assignments(path) for path in assignment_paths]
    gene_set = set()
    for df in per_run:
        gene_set.update(df["gene"].tolist())
    genes = sorted(gene_set)
    logging.info("n_genes=%s", len(genes))

    os.makedirs(args.output, exist_ok=True)

    signature_matrix = _build_signature_matrix(per_run, genes)
    groups = _group_by_signature(signature_matrix)
    n_genes = len(genes)

    logging.info("Building co-association sparse matrix ...")
    matrix = build_coassoc_matrix(groups, n_genes)
#    sp.save_npz(os.path.join(args.output, "coassociation_matrix_train.npz"), matrix)

    pairs_df, _ = generate_pairs_df(groups, genes)
    genes_leiden = sorted(set(pairs_df["gene_i"]) | set(pairs_df["gene_j"]))
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes_leiden)}
    edges = [
        (gene_to_idx[left], gene_to_idx[right])
        for left, right in zip(pairs_df["gene_i"], pairs_df["gene_j"])
        if left != right
    ]

    logging.info("Building igraph Graph... edges: %s", len(edges))
    graph = ig.Graph(n=len(genes_leiden), edges=edges, directed=False)
    graph.vs["name"] = genes_leiden

    logging.info("Running Leiden (resolution=%s) ...", args.resolution)
    part = la.find_partition(
        graph,
        la.RBConfigurationVertexPartition,
        resolution_parameter=args.resolution,
    )
    labels = np.array(part.membership, dtype=int)
    logging.info("Clusters = %s", len(np.unique(labels)))

    df_out = pd.DataFrame({"gene": genes_leiden, "cluster": labels})
    df_out.to_csv(
        os.path.join(args.output, "consensus_leiden_assignments.csv"), index=False
    )

    cluster_anno = df_out.copy()
    cutoff = int(np.ceil(0.01 * len(cluster_anno)))
    small = cluster_anno["cluster"].value_counts().loc[lambda x: x < cutoff].index
    cluster_anno.loc[cluster_anno["cluster"].isin(small), "cluster"] = np.nan

    mask = cluster_anno["cluster"].notna()
    cluster_anno.loc[mask, "cluster"] = (
        pd.factorize(cluster_anno.loc[mask, "cluster"])[0] + 1
    )
    cluster_anno.astype({"cluster": "Int64"}).to_csv(
        os.path.join(args.output, "human_train_cluster_label.txt"),
        sep="\t",
        index=False,
    )


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()

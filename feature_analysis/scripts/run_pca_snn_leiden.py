import argparse, json, os, re
import heapq
import logging

import numpy as np
import pandas as pd
from scipy import sparse as sp
import igraph as ig
import leidenalg as la
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from contextlib import contextmanager


logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

_RE_DROP = re.compile(r"^f([0-9]|[1-3]?[0-9]|40)>=1.0$")


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog,
            max_help_position=40,
            width=100,
        )
    )
    parser.add_argument(
        "--input", required=False, nargs="+", help="", default=[], dest="input"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Result path. [required]",
        dest="output",
    )
    parser.add_argument(
        "--index_file",
        type=str,
        required=False,
        default=None,
        help="Feature index file (2 cols: index, name)",
        dest="index_file",
    )
    parser.add_argument(
        "--seed", type=int, required=True, default=100, help="", dest="seed"
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=64,
        required=False,
        help="",
        dest="n_components",
    )
    parser.add_argument("--k_query", type=int, default=100, dest="k_query")
    parser.add_argument("--top_m", type=int, default=64, dest="top_m")
    parser.add_argument("--mutual", action="store_true")
    parser.add_argument("--use_jaccard", action="store_true")
    parser.add_argument(
        "--min_weight",
        type=float,
        required=False,
        default=3.0,
        help="shared-count threshold; if --use_jaccard then J>=this",
        dest="min_weight",
    )
    parser.add_argument(
        "--pca_file", type=str, required=False, help="", dest="pca_file"
    )
    return parser.parse_args()


def _read_input(input_files):
    dfs = []
    cell_genes = []
    for f in input_files:
        df = pd.read_csv(f, index_col=0)
        cell_gene = df.index.tolist()
        dfs.append(df)
        cell_genes.append(cell_gene)
    return dfs, cell_genes


def filter_shap_mat(shap_df, feature_index, cell_genes, output):
    cols = shap_df.columns.to_list()
    keep_mask = [(_RE_DROP.match(c) is None) for c in cols]
    clean_cols = [c for c, k in zip(cols, keep_mask) if k]
    shap_df = shap_df[clean_cols]

    split_df = shap_df.columns.to_series().str.split(">=", expand=True)
    split_df.columns = ["feature", "split"]
    split_df = split_df.merge(
        feature_index, left_on="feature", right_on="index", how="left"
    )
    shap_df.columns = split_df["name"] + ">=" + split_df["split"]

    abs_vals = np.abs(shap_df.values)
    nodes_abs_q95 = np.quantile(abs_vals, 0.95, axis=0)
    node_cutoff = np.quantile(nodes_abs_q95, 0.1)
    clean_nodes = np.where(nodes_abs_q95 >= node_cutoff)[0]
    shap_df = shap_df.iloc[:, clean_nodes]

    abs_vals = np.abs(shap_df.values)
    genes_abs_q95 = np.quantile(abs_vals, 0.95, axis=1)
    gene_cutoff = np.quantile(genes_abs_q95, 0.1)
    clean_genes = np.where(genes_abs_q95 >= gene_cutoff)[0]
    shap_df = shap_df.iloc[clean_genes, :]

    valid_idx = [i for i in cell_genes[0] if i in shap_df.index]
    shap_train = shap_df.loc[valid_idx]
    shap_train.to_csv(
        os.path.join(output, "train_shap_filtered.csv"), index=True, header=True
    )
    valid_idx = [i for i in cell_genes[1] if i in shap_df.index]
    shap_test = shap_df.loc[valid_idx]
    shap_test.to_csv(
        os.path.join(output, "test_shap_filtered.csv"), index=True, header=True
    )

    shap_train = shap_train.apply(pd.to_numeric, errors="coerce")
    shap_train = shap_train.astype(np.float32, copy=False)

    before = shap_train.shape[0]
    shap_train = shap_train.loc[shap_train.abs().sum(axis=1) > 0]
    removed = before - shap_train.shape[0]
    if removed > 0:
        logging.warning(f"Removed {removed} all-zero rows")
    if shap_train.empty:
        raise ValueError("No valid rows left after filtering.")
    shap_train = shap_train.fillna(0.0)
    return shap_train, shap_test


def pca_embed(X, n_components=None):
    Xv = X.values.astype(np.float32)
    n_samples, n_features = Xv.shape

    if n_features <= 2:
        logging.info(f"[PCA] n_features={n_features} <= 2, skip PCA.")
        return Xv

    if n_components is None:
        logging.info("[PCA] Auto-select components for ~80% explained variance ...")
        pca_tmp = PCA(random_state=0, svd_solver="full").fit(Xv)
        cum = np.cumsum(pca_tmp.explained_variance_ratio_)
        d = int(np.searchsorted(cum, 0.8) + 1)
        d = max(2, min(d, n_features - 1))
    else:
        d = int(max(2, min(n_components, n_features - 1)))

    logging.info(f"[PCA] Using {d} components")
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(Xv)
    pca = PCA(n_components=d, random_state=0, svd_solver="full")
    emb = pca.fit_transform(X_scaled)
    return emb.astype(np.float32, copy=False)


def compute_pca(shap_df, output, n_components):
    pca_path = os.path.join(output, "PCA_results.csv")
    emb = pca_embed(shap_df, n_components)
    df_emb = pd.DataFrame(
        emb, index=shap_df.index, columns=[f"PC{i+1}" for i in range(emb.shape[1])]
    )
    df_emb.to_csv(pca_path)
    logging.info(f"[PCA] Saved: {pca_path}")
    return emb


def get_kNN_indices(emb, k_query, n_jobs=1):
    nn = NearestNeighbors(n_neighbors=k_query + 1, algorithm="auto", n_jobs=n_jobs)
    nn.fit(emb)
    I = nn.kneighbors(emb, return_distance=False)[:, 1:]
    return I.astype(np.int32, copy=False)


def _shared_count_sorted(a, b):
    i = j = c = 0
    la, lb = a.shape[0], b.shape[0]
    while i < la and j < lb:
        ai = a[i]
        bj = b[j]
        if ai == bj:
            c += 1
            i += 1
            j += 1
        elif ai < bj:
            i += 1
        else:
            j += 1
    return c


def build_SNN(indices, top_m=64, mutual=False, use_jaccard=False, min_weight=3.0):
    n, kq = indices.shape
    idx_sorted = np.sort(indices, axis=1)

    nbr_sets = [set(row.tolist()) for row in idx_sorted] if mutual else None

    heaps = [[] for _ in range(n)]

    def push(i, w, j):
        h = heaps[i]
        if len(h) < top_m:
            heapq.heappush(h, (w, j))
        elif w > h[0][0]:
            heapq.heapreplace(h, (w, j))

    for u in range(n):
        Nu = idx_sorted[u]
        for v in Nu:
            if mutual and (u not in nbr_sets[v]):
                continue
            Nv = idx_sorted[v]
            sc = _shared_count_sorted(Nu, Nv)
            if use_jaccard:
                denom = Nu.shape[0] + Nv.shape[0] - sc
                w = (sc / denom) if denom > 0 else 0.0
            else:
                w = float(sc)
            if w < min_weight:
                continue
            push(u, w, v)
            push(v, w, u)

    edges, weights = [], []
    seen = set()
    for u in range(n):
        for w, v in heaps[u]:
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            edges.append((a, b))
            weights.append(float(w))

    g = ig.Graph(n=n, edges=edges, directed=False)
    if edges:
        g.es["weight"] = weights

    edge_df = pd.DataFrame(
        {"src": [a for a, _ in edges], "dst": [b for _, b in edges], "weight": weights}
    )
    return g, edge_df


def graph_to_csr(g, n):
    e = np.array(g.get_edgelist(), dtype=np.int64)
    if e.size == 0:
        return sp.csr_matrix((n, n), dtype=np.float32)
    w = np.asarray(g.es["weight"], dtype=np.float32)
    W = sp.csr_matrix((w, (e[:, 0], e[:, 1])), shape=(n, n), dtype=np.float32)
    return W


def cluster_by_leiden(g, resolution, seed=100, min_comp=5):
    comps = g.connected_components()
    labels = np.full(g.vcount(), -1, dtype=int)
    next_id = 0
    for vertices in comps:
        if len(vertices) < min_comp or g.ecount() == 0:
            for v in vertices:
                labels[v] = next_id
                next_id += 1
            continue
        sub = g.subgraph(vertices)
        part = la.find_partition(
            sub,
            la.RBConfigurationVertexPartition,
            weights=sub.es["weight"] if sub.ecount() > 0 else None,
            resolution_parameter=resolution,
            seed=seed,
        )
        mem = np.asarray(part.membership, dtype=int)
        uniq = np.unique(mem)
        remap = {c: (next_id + i) for i, c in enumerate(sorted(uniq))}
        for i, v in enumerate(vertices):
            labels[v] = remap[mem[i]]
        next_id += len(uniq)
    return labels


def main():
    args = parse_args()
    input = args.input
    output = args.output
    seed = args.seed
    n_components = args.n_components
    k_query = args.k_query
    top_m = args.top_m
    min_weight = args.min_weight
    mutual = args.mutual
    use_jaccard = args.use_jaccard
    index_file = args.index_file
    pca_file = args.pca_file

    np.random.seed(seed)
    os.makedirs(output, exist_ok=True)

    logging.info("Processing data matrix...")
    dfs, cell_genes = _read_input(input)
    shap_df = pd.concat(dfs, axis=0, ignore_index=False)

    feature_index = pd.read_csv(index_file, sep="\t", header=None)
    feature_index.columns = ["index", "name"]
    feature_index.set_index("index", inplace=True)
    shap_df, _ = filter_shap_mat(shap_df, feature_index, cell_genes, output)

    if pca_file == None:
        logging.info("Performing PCA...")
        emb = compute_pca(shap_df, output, n_components=n_components)

    else:
        emb = pd.read_csv(pca_file, index_col=0)
        emb = emb.to_numpy()

    emb = emb.astype(np.float32, copy=False)
    logging.info("Performing kNN...")
    kq = int(np.clip(k_query, 8, 1024))
    indices = get_kNN_indices(emb, kq)

    logging.info("Performing SNN from kNN edges...")
    g, edge_df = build_SNN(
        indices,
        top_m=top_m,
        mutual=mutual,
        use_jaccard=use_jaccard,
        min_weight=min_weight,
    )
    genes = shap_df.index.to_numpy()
    W = graph_to_csr(g, len(genes))
    sp.save_npz(os.path.join(output, "snn_graph_train.npz"), W)
    np.savetxt(os.path.join(output, "cell_gene_train.txt"), genes, fmt="%s")

    resolutions = np.linspace(0.1, 0.7, 7)
    for res in resolutions:
        logging.info(f"Running Leiden with resolution = {res:.1f}")
        labels = cluster_by_leiden(g, resolution=res, seed=seed, min_comp=5)
        logging.info(f"Leiden clusters: {len(set(labels))}")
        df_out = pd.DataFrame(
            {
                "gene": shap_df.index.values,
                "cluster": labels,
                "resolution": res,
                "randomseed": seed,
                "k_query": k_query,
                "top_m": top_m,
                "mutual": bool(mutual),
                "use_jaccard": bool(use_jaccard),
                "min_weight": min_weight,
                "n_components": emb.shape[1],
            }
        )
        df_out.to_csv(
            os.path.join(output, f"leiden_assignments_res{res:.1f}.csv"), index=False
        )


if __name__ == "__main__":
    main()

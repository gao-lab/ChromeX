#!/usr/bin/env python3
import os
import math
import argparse
import pandas as pd
import subprocess
import gzip
import itertools
import heapq
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple

# Optional: JIT for calculate_support
try:
    from numba import njit
except ImportError:

    def njit(func):
        return func


@njit
def calculate_support_jit(bitmask_list, total):
    """
    Intersect provided item bitmasks and return popcount of the result.
    'total' kept for signature compatibility (unused).
    """
    if len(bitmask_list) == 0:
        return 0
    mask = bitmask_list[0]
    for bm in bitmask_list[1:]:
        mask &= bm
        if mask == 0:
            return 0
    return bin(mask).count("1")


class SupportCalculator:
    """Builds item → tid‐bitmask mapping"""

    def __init__(self, transactions: List[List[str]]):
        self.total = len(transactions)
        self.bitmasks = [0] * self.total  # tids as single bits (not used in calc, kept for compat)
        self.item2idx: Dict[str, int] = {}
        for tid, t in enumerate(transactions):
            mask = 1 << tid
            self.bitmasks[tid] = mask
            for it in t:
                self.item2idx[it] = self.item2idx.get(it, 0) | mask

    def calculate(self, items: Tuple[str, ...]) -> int:
        mask = (1 << self.total) - 1
        for it in items:
            mask &= self.item2idx.get(it, 0)
            if mask == 0:
                return 0
        return bin(mask).count("1")


# Globals for worker processes
_target_calc: SupportCalculator = None
_other_calc: SupportCalculator = None
_TOTAL_T: int = 0
_TOTAL_O: int = 0
_USE_JIT: bool = False


def init_worker(target_trans, other_trans, total_t, total_o, use_jit):
    global _target_calc, _other_calc, _TOTAL_T, _TOTAL_O, _USE_JIT
    _target_calc = SupportCalculator(target_trans)
    _other_calc = SupportCalculator(other_trans)
    _TOTAL_T = total_t
    _TOTAL_O = total_o
    _USE_JIT = use_jit
    if _USE_JIT:
        try:
            # warm-up compile
            calculate_support_jit([1, 1], 2)
        except Exception:
            pass


def process_batch(args_tuple):
    """
    Read lines like: items_csv <TAB> b_value
    Compute: precision, recall, F1 (all in 0–1),
    keep local top by F1.
    """
    lines, top_k = args_tuple
    local_counter = itertools.count()
    local_rank: List[Tuple[float, int, Dict]] = []

    for line in lines:
        parts = line.rstrip("\n").rsplit("\t", 1)
        if len(parts) != 2:
            continue
        items_part, b_str = parts
        try:
            items = tuple(sorted(items_part.split(",")))
            b = float(b_str)
        except ValueError:
            continue

        # supports
        if _USE_JIT:
            t_masks = [_target_calc.item2idx.get(it, 0) for it in items]
            o_masks = [_other_calc.item2idx.get(it, 0) for it in items]
            sup_t = calculate_support_jit(t_masks, _TOTAL_T)
            sup_o = calculate_support_jit(o_masks, _TOTAL_O)
        else:
            sup_t = _target_calc.calculate(items)
            sup_o = _other_calc.calculate(items)

        tp = sup_t
        fp = sup_o
        fn = _TOTAL_T - sup_t
        # tn = _other_calc.total - sup_o  # not needed

        # precision/recall in probability (0–1)
        precision = (tp / (tp + fp)) if (tp + fp) else 0.0
        recall = (tp / _TOTAL_T) if _TOTAL_T else 0.0
        f1 = (
            (2.0 * precision * recall / (precision + recall))
            if (precision + recall)
            else 0.0
        )

        rec = {
            "items": items,
            "b": b,
            "recall": recall,
            "precision": precision,
            "f1": f1,
        }

        idx = next(local_counter)
        entry = (f1, idx, rec)  # rank by F1

        if len(local_rank) < top_k:
            heapq.heappush(local_rank, entry)
        elif f1 > local_rank[0][0]:
            heapq.heapreplace(local_rank, entry)

    return local_rank


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Optimized Eclat Miner with multiprocessing (rank by F1)"
    )
    parser.add_argument(
        "--target_cluster",
        required=True,
        help="Target cluster ID",
        dest="target_cluster",
    )
    parser.add_argument("--gn_mat", default="", dest="gn_mat")
    parser.add_argument("--labels", default="", dest="labels")
    parser.add_argument("--DF", default="", dest="DF")
    parser.add_argument(
        "--exe", default="", help="Path to fpgrowth program", dest="exe"
    )
    parser.add_argument("--minsup", type=float, default=0.8, dest="minsup")
    parser.add_argument("--maxsup", type=float, default=1.0, dest="maxsup")
    parser.add_argument(
        "--max_len",
        type=int,
        default=7,
        help="Maximum itemset length for fpgrowth (-n)",
        dest="max_len",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="Lines per batch; larger to reduce overhead",
        dest="batch_size",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10000,
        help="Keep top-K itemsets by F1",
        dest="top_k",
    )
    parser.add_argument(
        "--use_jit",
        action="store_true",
        help="Enable numba-accelerated support calc",
        dest="use_jit",
    )
    parser.add_argument("--output", default="", help="", dest="output")
    return parser.parse_args()


def process_cluster(args):
    output = args.output
    os.makedirs(output, exist_ok=True)
    gn_mat = pd.read_csv(args.gn_mat, index_col=0, sep="\t")
    labels = pd.read_csv(args.labels, sep="\t")
    DFs_df = pd.read_csv(args.DF, sep="\t")

    gn_mat_genes = pd.DataFrame({"gene": gn_mat.index})
    labels = pd.merge(gn_mat_genes, labels, on="gene", how="left")

    target_cluster = int(args.target_cluster)
    target_samples = labels[labels["cluster"] == target_cluster].index
    other_samples = labels[labels["cluster"] != target_cluster].index
    cluster_nodes = DFs_df[DFs_df["cluster"] == target_cluster]["nodes"].tolist()

    target_matrix = gn_mat.iloc[target_samples].loc[:, cluster_nodes]
    other_matrix = gn_mat.iloc[other_samples].loc[:, cluster_nodes]

    target_trans = [row[row == 1].index.tolist() for _, row in target_matrix.iterrows()]
    other_trans = [row[row == 1].index.tolist() for _, row in other_matrix.iterrows()]

    input_file = os.path.join(output, f"cluster_{target_cluster}_fpgrowth_input.txt")
    with open(input_file, "w") as f:
        for t in target_trans:
            f.write(" ".join(t) + "\n")

    abs_min = max(math.ceil(args.minsup * len(target_trans)), 1)
    abs_max = max(math.ceil(args.maxsup * len(target_trans)), 1)

    # fpgrowth command with parameterized -n (max_len)
    fpgrowth = args.exe
    cmd = (
        f"{fpgrowth} -ts -s{-abs_min} -S{-abs_max} -n{args.max_len} -eb -v '\t'%e -d-100 -k, {input_file} - | "
        "awk 'BEGIN{OFS=\"\\t\"} {print $1,$2}'"
    )
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, text=True)

    batch_size = args.batch_size
    top_k = args.top_k
    use_jit = args.use_jit
    total_t = len(target_trans)
    total_o = len(other_trans)

    batches = []
    batch = []
    for line in proc.stdout:
        batch.append(line)
        if len(batch) >= batch_size:
            batches.append((batch.copy(), top_k))
            batch.clear()
    if batch:
        batches.append((batch, top_k))

    proc.stdout.close()
    return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"fpgrowth command failed with exit code {return_code}")

    # Global top-K by F1
    global_counter = itertools.count()
    global_rank: List[Tuple[float, int, Dict]] = []

    if batches:
        nproc = min(max(1, cpu_count() - 10), len(batches))
        with Pool(
            processes=nproc,
            initializer=init_worker,
            initargs=(target_trans, other_trans, total_t, total_o, use_jit),
        ) as pool:
            for local_rank in pool.imap_unordered(process_batch, batches):
                for score, _, rec in local_rank:
                    gid = next(global_counter)
                    entry = (score, gid, rec)
                    if len(global_rank) < top_k:
                        heapq.heappush(global_rank, entry)
                    elif score > global_rank[0][0]:
                        heapq.heapreplace(global_rank, entry)

    # Extract and build final DataFrame
    kept = [rec for score, _, rec in global_rank]
    if kept:
        df_final = pd.DataFrame(kept)
        df_final["target_cluster"] = target_cluster
        df_final["itemset"] = df_final["items"].map(lambda t: ",".join(t))
        df_final["length"] = df_final["items"].map(len)

        # Sort by F1 desc and select requested columns
        df_final = df_final.sort_values("f1", ascending=False)[
            ["target_cluster", "itemset", "b", "length", "recall", "precision", "f1"]
        ]
    else:
        df_final = pd.DataFrame(
            columns=[
                "target_cluster",
                "itemset",
                "b",
                "length",
                "recall",
                "precision",
                "f1",
            ]
        )

    out_final = os.path.join(output, f"cluster_{target_cluster}_final.txt.gz")
    with gzip.open(out_final, "wt") as f_out:
        df_final.to_csv(f_out, sep="\t", index=False)

    print(f"Processing complete. Results saved to {out_final}")


def main():
    args = parse_arguments()
    process_cluster(args)


if __name__ == "__main__":
    main()

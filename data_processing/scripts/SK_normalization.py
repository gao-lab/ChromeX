import sys
import argparse
import h5py
import numpy as np
import pandas as pd
import copy
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from operator import add
import logging
import cooler
from cooler.parallel import partition, split
from cooler.util import mad, parse_cooler_uri

parser = argparse.ArgumentParser(
    formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
        prog, max_help_position=40, width=100
    )
)
parser.add_argument(
    "-i",
    "--input",
    type=str,
    required=True,
    help="Input cool file. [required]",
    dest="cool",
)
parser.add_argument(
    "-p",
    "--nproc",
    type=int,
    default=8,
    help="Number of processes to split the work between. [default: 8]",
    dest="nproc",
)
parser.add_argument(
    "--max-iters",
    type=int,
    default=1000,
    help="Maximum number of iterations to perform if convergence is not achieved. [default: 1000]",
    dest="max_iters",
)
parser.add_argument(
    "--tol",
    type=float,
    default=1e-5,
    help="Threshold value of variance of the marginals for the algorithm to converge. [default: 1e-05]",
    dest="tol",
)
parser.add_argument(
    "-n",
    "--name",
    type=str,
    default="weight",
    help="Name of column to write to",
    dest="name",
)
parser.add_argument(
    "--ignore-diags",
    type=int,
    default=2,
    help="Number of diagonals of the contact matrix to ignore, including the main diagonal.",
    dest="ignore_diags",
)
parser.add_argument(
    "--mad-max",
    type=int,
    default=5,
    help="Ignore bins from the contact matrix using the 'MAD-max' filter: bins whose log marginal sum is less than 'mad-max' median absolute deviations below the median log marginal sum of all the bins in the same chromosome. [default: 5]",
    dest="mad_max",
)
parser.add_argument(
    "--min-nnz",
    type=int,
    default=10,
    help="Ignore bins from the contact matrix whose marginal number of nonzeros is less than this number. [default: 10]",
    dest="min_nnz",
)
parser.add_argument(
    "--min-count",
    type=int,
    default=0,
    help="Ignore bins from the contact matrix whose marginal count is less than this number. [default: 0]",
    dest="min_count",
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def _zero_trans(chunk, data):
    chrom_ids = chunk["bins"]["chrom"]
    pixels = chunk["pixels"]
    mask = chrom_ids[pixels["bin1_id"]] != chrom_ids[pixels["bin2_id"]]
    data[mask] = 0
    return data


def _zero_diags(n_diags, chunk, data):
    pixels = chunk["pixels"]
    mask = np.abs(pixels["bin1_id"] - pixels["bin2_id"]) < n_diags
    data[mask] = 0
    return data


def _binarize(chunk, data):
    data[data != 0] = 1
    return data


def _init(chunk):
    return np.copy(chunk["pixels"]["count"])


def _marginalize(chunk, data):
    n = len(chunk["bins"]["chrom"])
    pixels = chunk["pixels"]
    marg = (
        np.bincount(pixels["bin1_id"], weights=data, minlength=n)
        + np.bincount(pixels["bin2_id"], weights=data, minlength=n)
    )
    return marg


def SK_balance_cisonly(
    clr,
    spans,
    chunksize,
    filters,
    bias_filter,
    tol,
    max_iters,
    nproc,
):
    chroms = clr.chroms()["name"][:]
    chrom_offsets = clr._load_dset("indexes/chrom_offset")
    bin1_offsets = clr._load_dset("indexes/bin1_offset")
    n_bins = len(bias_filter)
    converged = np.full_like(chroms, np.nan)
    df = pd.DataFrame(
        {"chroms": chroms, "lo": chrom_offsets[:-1], "hi": chrom_offsets[1:]}
    )

    # Calculate separately for each chromosome
    def _each(chromosome):
        chr_converged = False
        chr = chromosome[0]
        lo = chromosome[1]
        hi = chromosome[2]
        marg = copy.deepcopy(bias_filter)
        marg = marg[lo:hi]
        mask = _binarize(marg, marg)
        bias_matrix = np.outer(mask, mask)
        A_ = clr.matrix(balance=False).fetch(chr)
        A_ori = copy.deepcopy(A_)
        A = A_ * bias_matrix
        n, m = A.shape
        row_mask = np.any(A != 0, axis=1)
        col_mask = np.any(A != 0, axis=0)
        A_nonzero = A[row_mask][:, col_mask]
        if A_nonzero.size == 0:
            row_scaling_out = np.full(n, np.nan)
            col_scaling_out = np.full(m, np.nan)
            chr_bias = np.sqrt(col_scaling_out * row_scaling_out)
            return chr_bias, chr_converged
        row_sums = A_nonzero.sum(axis=1)
        col_sums = A_nonzero.sum(axis=0)
        r, c = A_nonzero.shape
        for i in range(min(r, c)):
            if row_sums[i] == 0 or col_sums[i] == 0:
                A_nonzero[i, i] += 1
        row_scaling = np.ones(r)
        col_scaling = np.ones(c)
        for _ in range(max_iters):
            # Update column scaling
            col_scaling = 1 / A_nonzero.dot(col_scaling)
            # Update row scaling
            row_scaling = 1 / (col_scaling.dot(A_nonzero.T))
            # Calculate balanced matrix for checking convergence
            A_balanced = row_scaling[:, np.newaxis] * A_nonzero * col_scaling
            new_row_sums = A_balanced.sum(axis=1)
            new_col_sums = A_balanced.sum(axis=0)
            # Check convergence
            if np.allclose(new_row_sums, 1, atol=tol) and np.allclose(
                new_col_sums, 1, atol=tol
            ):
                logging.info(
                    f"{chr}: The farthest distance of row elements from 1 is {np.max(new_row_sums) - 1}, The farthest distance of column elements from 1 is {np.max(new_col_sums) - 1}"
                )
                logging.info(f"Converged on {chr}")
                chr_converged = True
                break
        else:
            logging.warning(
                f"{chr}: The farthest distance of row elements from 1 is {np.max(new_row_sums) - 1}, The farthest distance of column elements from 1 is {np.max(new_col_sums) - 1}"
            )
            logging.warning(f"Iteration limit reached without convergence on {chr}.")
        row_scaling_out = np.full((A.shape[0],), np.nan)
        col_scaling_out = np.full((A.shape[0],), np.nan)
        row_scaling_out[row_mask] = row_scaling
        col_scaling_out[col_mask] = col_scaling
        chr_bias = np.sqrt(col_scaling_out * row_scaling_out)
        return chr_bias, chr_converged

    with ThreadPoolExecutor(max_workers=nproc) as executor:
        results = list(executor.map(_each, df.values))
    bias = np.concatenate([array for array, flag in results])
    converged = np.array([flag for array, flag in results])
    return bias, converged


def balance_cooler(
    clr,
    *,
    cis_only=True,
    ignore_diags=2,
    mad_max=5,
    min_nnz=10,
    min_count=0,
    tol=1e-5,
    max_iters=1000,
    map=map,
    use_lock=False,
    nproc,
):
    # Divide the number of elements into non-overlapping chunks
    nnz = int(clr.info["nnz"])
    chunksize = nnz
    spans = [(0, nnz)]
    # List of pre-marginalization data transformations
    base_filters = []
    base_filters.append(_zero_trans)
    base_filters.append(partial(_zero_diags, ignore_diags))
    # Initialize the bias weights
    n_bins = int(clr.info["nbins"])
    bias_filter = np.ones(n_bins, dtype=float)
    # Drop bins with too few nonzeros from bias
    if min_nnz > 0:
        filters = [_binarize, *base_filters]
        marg_nnz = (
            split(clr, spans=spans, map=map, use_lock=False)
            .prepare(_init)
            .pipe(filters)
            .pipe(_marginalize)
            .reduce(add, np.zeros(n_bins))
        )
    bias_filter[marg_nnz < min_nnz] = 0
    filters = base_filters
    marg = (
        split(clr, spans=spans, map=map, use_lock=False)
        .prepare(_init)
        .pipe(filters)
        .pipe(_marginalize)
        .reduce(add, np.zeros(n_bins))
    )
    # Drop bins with too few total counts from bias
    if min_count:
        bias_filter[marg < min_count] = 0
    # MAD-max filter on the marginals
    if mad_max > 0:
        offsets = clr._load_dset("indexes/chrom_offset")
        for lo, hi in zip(offsets[:-1], offsets[1:]):
            c_marg = marg[lo:hi]
            if c_marg[c_marg > 0].size == 0:
                marg[lo:hi] = np.full(c_marg.shape, np.nan)
            else:
                marg[lo:hi] /= np.median(c_marg[c_marg > 0])
        logNzMarg = np.log(marg[marg > 0])
        med_logNzMarg = np.median(logNzMarg)
        dev_logNzMarg = mad(logNzMarg)
        cutoff = np.exp(med_logNzMarg - mad_max * dev_logNzMarg)
        bias_filter[marg < cutoff] = 0
    # Do balancing
    bias, converged = SK_balance_cisonly(
        clr,
        spans,
        chunksize,
        filters,
        bias_filter,
        tol,
        max_iters,
        nproc,
    )
    stats = {
        "tol": tol,
        "min_nnz": min_nnz,
        "min_count": min_count,
        "mad_max": mad_max,
        "cis_only": cis_only,
        "converged": converged,
    }
    return bias, stats


if __name__ == "__main__":
    args = parser.parse_args()
    cool_file = args.cool
    nproc = args.nproc
    max_iters = args.max_iters
    tol = args.tol
    name = args.name
    ignore_diags = args.ignore_diags
    mad_max = args.mad_max
    min_nnz = args.min_nnz
    min_count = args.min_count

    cool_path, group_path = parse_cooler_uri(cool_file)
    with h5py.File(cool_path, "r+") as h5:
        grp = h5[group_path]
        if name in grp["bins"]:
            del grp["bins"][name]

    logging.info(f'Balancing "{cool_path}"')
    clr = cooler.Cooler(cool_path)
    bias, stats = balance_cooler(
        clr,
        cis_only=True,
        ignore_diags=ignore_diags,
        mad_max=mad_max,
        min_nnz=min_nnz,
        min_count=min_count,
        tol=tol,
        max_iters=max_iters,
        map=map,
        use_lock=False,
        nproc=nproc,
    )
    with h5py.File(cool_path, "r+") as h5:
        grp = h5[group_path]
        # add the bias column to the file
        h5opts = {"compression": "gzip", "compression_opts": 6}
        grp["bins"].create_dataset(name, data=bias, **h5opts)
        grp["bins"][name].attrs.update(stats)

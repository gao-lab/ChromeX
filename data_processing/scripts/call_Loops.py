import pandas as pd
import bioframe
import cooltools
import cooler
import numpy as np
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input",
    type=str,
    help="input cool file",
    dest="cool"
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    help="output tsv file",
    dest="output"
)
parser.add_argument(
    "-g",
    "--genomeID",
    type=str,
    help="genome ID(e.g. 'hg38' or 'mm10')",
    dest="genomeID",
)
parser.add_argument(
    "-p",
    "--nproc",
    type=int,
    default=1,
    help="Number of processes to use.",
    dest="nproc",
)

args = parser.parse_args()

cool_file = args.cool
output = args.output
genomeID = args.genomeID
nproc = args.nproc

clr = cooler.Cooler(cool_file)
binsize = clr.info["bin-size"]
kernels = cooltools.api.dotfinder.recommend_kernels(binsize)
# create a grid of coordinates, to define round kernels

# half width of the kernel
if binsize > 28000:
    raise ValueError(
        f"Provided cooler has resolution of {binsize} bases,"
        " which is too coarse for automated kernel recommendation."
        " Provide custom kernels to proceed."
    )
elif binsize >= 18000:
    half = 3
elif binsize >= 8000:
    half = 5
elif binsize >= 4000:
    half = 7
else:
    raise ValueError(
        f"Provided cooler has resolution {binsize} bases,"
        " which is too fine for automated kernel recommendation. Provide custom kernels"
        " to proceed."
    )

x, y = np.meshgrid(
    np.linspace(-half, half, 2 * half + 1),
    np.linspace(-half, half, 2 * half + 1),
)
# now define a donut-like mask as pixels between 2 radii: sqrt(7) and sqrt(30):
mask = (x**2 + y**2 > 7) & (x**2 + y**2 <= 30)
mask[:, half] = 0
mask[half, :] = 0

# lowleft mask - zero out neccessary parts
mask_ll = mask.copy()
mask_ll[:, :half] = 0
mask_ll[half:, :] = 0

# new kernels with more round donut and lowleft masks:
kernels_round = {
    "donut": mask,
    "vertical": kernels["vertical"].copy(),
    "horizontal": kernels["horizontal"].copy(),
    "lowleft": mask_ll,
}

if genomeID == "hg38":
    chromsizes = bioframe.fetch_chromsizes(genomeID)
    cens = bioframe.fetch_centromeres(genomeID)
    arms = bioframe.make_chromarms(chromsizes, cens)
    expected = cooltools.expected_cis(clr, view_df=arms, nproc=nproc)
    dots_round_df_all = cooltools.dots(
        clr,
        expected=expected,
        view_df=arms,
        kernels=kernels_round,  # provide custom kernels
        max_loci_separation=10000000,
        clustering_radius=None,
        nproc=nproc,
    )
elif genomeID == "mm10":
    expected = cooltools.expected_cis(clr, nproc=nproc)
    dots_round_df_all = cooltools.dots(
        clr,
        expected=expected,
        kernels=kernels_round,  # provide custom kernels
        max_loci_separation=10000000,
        clustering_radius=None,
        nproc=nproc,
    )
else:
    print("Unknown genomeID")
    sys.exit()

dots_round_df_all.to_csv(output, index=False, sep="\t")

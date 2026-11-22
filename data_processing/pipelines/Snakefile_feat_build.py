"""
To convert data structure of features and labels to model inputs.
"""


workdir: config["workdir"]


# ------------------------------------------------------------------->>>>>>
# Parameters
# ------------------------------------------------------------------->>>>>>
resolution10 = config["resolution"]
resolution50 = config["resolution"] * 5
genomeID = config["genomeID"]

# ------------------------------------------------------------------->>>>>>
# Tools
# ------------------------------------------------------------------->>>>>>
subcompartment_pre = config["subcompartment_pre"]
BL_pre = config["BL_pre"]
IF_and_RL_pre = config["IF_and_RL_pre"]
loop_pre = config["loop_pre"]
build_data = config["build_data"]

# ------------------------------------------------------------------->>>>>>
# Reference
# ------------------------------------------------------------------->>>>>>
ref_bed10_1 = config["ref_bed10_1"]
ref_bed10_2 = config["ref_bed10_2"]
ref_bed50_1 = config["ref_bed50_1"]
ref_bed50_2 = config["ref_bed50_2"]
gene_bed = config["gene_bed"]


# ------------------------------------------------------------------->>>>>>
# rule all
# ------------------------------------------------------------------->>>>>>
rule all:
    input:
        expand("Hi-C/model_x/{cell}_subcompartment_input.done", cell=config["samples"]),
        expand("Hi-C/model_x/{cell}_TAD_input.done", cell=config["samples"]),
        expand("Hi-C/model_x/{cell}_loop_input.done", cell=config["samples"]),
        expand("data/{cell}_data.done", cell=config["samples"]),


# ------------------------------------------------------------------->>>>>>
# Data structure conversion
# ------------------------------------------------------------------->>>>>>
# SUBCOMPARTMENT
rule subcompartment:
    input:
        file10="Hi-C/features/Compartments/{cell}_subcompartments_%s.bed" % resolution10,
        file50="Hi-C/features/Compartments/{cell}_subcompartments_%s.bed" % resolution50,
    output:
        file10="Hi-C/model_x/{cell}_subcompartment_input_%s.txt" % resolution10,
        file50="Hi-C/model_x/{cell}_subcompartment_input_%s.txt" % resolution50,
        done="Hi-C/model_x/{cell}_subcompartment_input.done",
    params:
        ref_bed10_2={ref_bed10_2},
        ref_bed50_2={ref_bed50_2},
        resolution10={resolution10},
        resolution50={resolution50},
    conda:
        "snakemake"
    shell:
        """
        set -euo pipefail
        Rscript {subcompartment_pre} -i {input.file10} -o {output.file10} -r {params.resolution10} -B {params.ref_bed10_2} &
        Rscript {subcompartment_pre} -i {input.file50} -o {output.file50} -r {params.resolution50} -B {params.ref_bed50_2} &
        wait
        touch {output.done}
        """


# TAD
rule tad:
    input:
        file10="Hi-C/features/TADs/{cell}_tads_%s.bed" % resolution10,
        file50="Hi-C/features/TADs/{cell}_tads_%s.bed" % resolution50,
    output:
        done="Hi-C/model_x/{cell}_TAD_input.done",
    conda:
        "snakemake"
    params:
        resolution10={resolution10},
        resolution50={resolution50},
        ref_bed50_1={ref_bed50_1},
        ref_bed10_2={ref_bed10_2},
        ref_bed50_2={ref_bed50_2},
        output_prefix="Hi-C/model_x/{cell}",
        genomeID={genomeID},
    shell:
        """
        set -euo pipefail
        bash {BL_pre} -i {input.file10} -o {params.output_prefix} -r {params.resolution10} -B {params.ref_bed10_2}
        bash {IF_and_RL_pre} -i {input.file50} -o {params.output_prefix} -r {params.resolution50} -b {params.ref_bed50_1} -B {params.ref_bed50_2}
        touch {output.done}
        """


# LOOP
rule loop:
    input:
        "Hi-C/features/Loops/{cell}_loops_%s.tsv" % resolution10,
    output:
        loop="Hi-C/model_x/{cell}_loop_input_%s.txt" % resolution10,
        done="Hi-C/model_x/{cell}_loop_input.done",
    conda:
        "snakemake"
    params:
        resolution10={resolution10},
        ref_bed1={ref_bed10_1},
        ref_bed2={ref_bed10_2},
    shell:
        """
        bash {loop_pre} -i {input} -o {output.loop} -r {params.resolution10} -b {params.ref_bed1} -B {params.ref_bed2}  \
        && touch {output.done}
        """


# ------------------------------------------------------------------->>>>>>
# Build candidate features for model
# ------------------------------------------------------------------->>>>>>
rule build_feature:
    input:
        compart_done="Hi-C/model_x/{cell}_subcompartment_input.done",
        tad_done="Hi-C/model_x/{cell}_TAD_input.done",
        loop_done="Hi-C/model_x/{cell}_loop_input.done",
        rna="RNA-seq/model_y/{cell}_fpkm.txt",
    output:
        hdf5="data/{cell}_data.hdf5",
        done="data/{cell}_data.done",
    conda:
        "snakemake"
    params:
        feature_prefix="Hi-C/model_x/{cell}",
        gene_bed={gene_bed},
    shell:
        """
        python {build_data} -b {params.gene_bed} -o {output.hdf5} -r {input.rna} -f {params.feature_prefix} \
        && touch {output.done}
        """

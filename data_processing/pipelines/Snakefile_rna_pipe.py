"""
# Snakemake workflow for RNA-seq data processing from .FASTQ to gene expression
"""


workdir: config["workdir"]


# ------------------------------------------------------------------->>>>>>
# Paths
# ------------------------------------------------------------------->>>>>>
genomeDir = config["genomeDir"]

# ------------------------------------------------------------------->>>>>>
# Tools
# ------------------------------------------------------------------->>>>>>
rsem_calculate_expression = config["rsem_calculate_expression"]
rsem_pre = config["rsem_pre"]
label_pre = config["label_pre"]

# ------------------------------------------------------------------->>>>>>
# Parameters
# ------------------------------------------------------------------->>>>>>
threads = config["threads"]
BAMsortingthreads = config["threads"] * 2

# ------------------------------------------------------------------->>>>>>
# Parameters
# ------------------------------------------------------------------->>>>>>
gtf = config["gtf"]

# ------------------------------------------------------------------->>>>>>
# rule all
# ------------------------------------------------------------------->>>>>>
trimmed = []
for cell, bioreps in config["samples"].items():
    for biorep in bioreps:
        trimmed.append(f"trimmed/{cell}_{biorep}.trimmed.done")

aligned = []
for cell, bioreps in config["samples"].items():
    for biorep in bioreps:
        aligned.append(f"aligned/{cell}_{biorep}.aligned.done")

quantified = []
for cell, bioreps in config["samples"].items():
    for biorep in bioreps:
        quantified.append(f"quantified/{cell}_{biorep}.gene.result.done")


rule all:
    input:
        trimmed,
        aligned,
        quantified,
        expand("model_y/{cell}_fpkm.done", cell=config["samples"].keys()),


# ------------------------------------------------------------------->>>>>>
# Pre-processing
# ------------------------------------------------------------------->>>>>>
# CUT ADAPTER
rule cutadapt:
    input:
        r1="fastq/{cell}_{biorep}_R1.fastq.gz",
    output:
        done="trimmed/{cell}_{biorep}.trimmed.done",
    params:
        dir="trimmed",
        threads={threads},
        r2="fastq/{cell}_{biorep}_R2.fastq.gz",
        output_prefix="trimmed/{cell}_{biorep}",
    conda:
        "snakemake"
    shell:
        """
        if [ -f {params.r2} ]; then
            # Paired-end alignment
            trim_galore --cores {params.threads} --quality 20 --stringency 3 --phred33 --paired --output_dir {params.dir} {input.r1} {params.r2}
            mv {params.output_prefix}_R1_val_1.fq.gz {params.output_prefix}_R1_trimmed.fq.gz
            mv {params.output_prefix}_R2_val_2.fq.gz {params.output_prefix}_R2_trimmed.fq.gz
        else
            # Single-end alignment
            trim_galore --cores {params.threads} --quality 20 --stringency 3 --phred33 --output_dir {params.dir} {input.r1}
        fi
        touch {output.done}
        """


# ------------------------------------------------------------------->>>>>>
# Alignment
# ------------------------------------------------------------------->>>>>>
# ALIGN FASTQ
rule align:
    input:
        done="trimmed/{cell}_{biorep}.trimmed.done",
    output:
        done="aligned/{cell}_{biorep}.aligned.done",
    params:
        threads={threads},
        BAMsortingthreads={BAMsortingthreads},
        r1="trimmed/{cell}_{biorep}_R1_trimmed.fq.gz",
        r2="trimmed/{cell}_{biorep}_R2_trimmed.fq.gz",
        output_prefix="aligned/{cell}_{biorep}.",
        genomeDir={genomeDir},
    conda:
        "snakemake"
    shell:
        """
        if [ -f {params.r2} ]; then
            # Paired-end alignment
            STAR --chimOutType WithinBAM --runThreadN {params.threads} --genomeDir {params.genomeDir} --readFilesIn {params.r1} {params.r2} --readFilesCommand zcat --outSAMtype BAM SortedByCoordinate --outBAMsortingThreadN {params.BAMsortingthreads} --outSAMunmapped Within --outSAMstrandField intronMotif --outSAMattributes NH HI AS NM MD --outFilterType BySJout --alignIntronMin 20 --alignIntronMax 1000000 --alignMatesGapMax 1000000 --alignSJoverhangMin 8 --alignSJDBoverhangMin 1 --sjdbScore 1 --quantMode TranscriptomeSAM --outFilterMismatchNmax 999 --outFilterMismatchNoverReadLmax 0.04 --outFilterMultimapNmax 20 --outFileNamePrefix {params.output_prefix} --limitBAMsortRAM 8000000000
        else
            # Single-end alignment
            STAR --chimOutType WithinBAM --runThreadN {params.threads} --genomeDir {params.genomeDir} --readFilesIn {params.r1} --readFilesCommand zcat --outSAMtype BAM SortedByCoordinate --outBAMsortingThreadN {params.BAMsortingthreads} --outSAMunmapped Within --outSAMstrandField intronMotif --outSAMattributes NH HI AS NM MD --outFilterType BySJout --alignIntronMin 20 --alignIntronMax 1000000 --alignMatesGapMax 1000000 --alignSJoverhangMin 8 --alignSJDBoverhangMin 1 --sjdbScore 1 --quantMode TranscriptomeSAM --outFilterMismatchNmax 999 --outFilterMismatchNoverReadLmax 0.04 --outFilterMultimapNmax 20 --outFileNamePrefix {params.output_prefix} --limitBAMsortRAM 8000000000
        fi
        touch {output.done}
        """


# ------------------------------------------------------------------->>>>>>
# Quantification
# ------------------------------------------------------------------->>>>>>
# QUANTIFY EXPRESSION
rule quantify:
    input:
        done="aligned/{cell}_{biorep}.aligned.done",
    output:
        done="quantified/{cell}_{biorep}.gene.result.done",
    params:
        threads={threads},
        rsem_pre={rsem_pre},
        bam="aligned/{cell}_{biorep}.Aligned.toTranscriptome.out.bam",
        output_prefix="quantified/{cell}_{biorep}",
        r2="fastq/{cell}_{biorep}_R2.fastq.gz",
    conda:
        "snakemake"
    shell:
        """
        if [ -f {params.r2} ]; then
            # Paired-end alignment
            {rsem_calculate_expression} --paired-end --bam --estimate-rspd --calc-ci --seed 12345 -p {params.threads} --no-bam-output --ci-memory 16000 --forward-prob 0.5 {params.bam} {params.rsem_pre} {params.output_prefix}
        else
            # Single-end alignment
            {rsem_calculate_expression} --bam --estimate-rspd --calc-ci --seed 12345 -p {params.threads} --no-bam-output --ci-memory 16000 --forward-prob 0.5 {params.bam} {params.rsem_pre} {params.output_prefix}
        fi
        touch {output.done}
        """


# ------------------------------------------------------------------->>>>>>
# Data structure conversion
# ------------------------------------------------------------------->>>>>>
rule rna:
    input:
        done=lambda wildcards: expand(
            "quantified/{cell}_{biorep}.gene.result.done",
            cell=wildcards.cell,
            biorep=config["samples"][wildcards.cell],
        ),
    output:
        label="model_y/{cell}_fpkm.txt",
        done="model_y/{cell}_fpkm.done",
    conda:
        "snakemake"
    params:
        input="quantified",
    shell:
        """
        Rscript {label_pre} -i {params.input} -g {gtf} -o {output.label} \
        && touch {output.done}
        """

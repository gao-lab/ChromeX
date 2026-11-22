"""
Snakemake workflow for Hi-C data processing from .FASTQ files to 3D genome features
"""


workdir: config["workdir"]


# ------------------------------------------------------------------->>>>>>
# Functions
# ------------------------------------------------------------------->>>>>>
def set_ligation_junction(site):
    ligation_mapping = {
        "HindIII": "AAGCTAGCTT",
        "MseI": "TTATAA",
        "DpnII": "GATCGATC",
        "MboI": "GATCGATC",
        "NcoI": "CCATGCATGG",
        "none": "XXXX",
        "Arima": "'(GAATAATC|GAATACTC|GAATAGTC|GAATATTC|GAATGATC|GACTAATC|GACTACTC|GACTAGTC|GACTATTC|GACTGATC|GAGTAATC|GAGTACTC|GAGTAGTC|GAGTATTC|GAGTGATC|GATCAATC|GATCACTC|GATCAGTC|GATCATTC|GATCGATC|GATTAATC|GATTACTC|GATTAGTC|GATTATTC|GATTGATC)'",
    }
    ligation = ligation_mapping.get(site, "XXXX")
    return ligation


# ------------------------------------------------------------------->>>>>>
# Reference
# ------------------------------------------------------------------->>>>>>
genomePath = config["genomePath"]
site_file = config["site_file"]
chromsize = config["chromsize"]
mainonly_chromsize = config["mainonly_chromsize"]

# ------------------------------------------------------------------->>>>>>
# Tools
# ------------------------------------------------------------------->>>>>>
chimeric_sam = config["chimeric_sam"]
adjust_insert_size = config["adjust_insert_size"]
dups_sam = config["dups_sam"]
sam_to_pre = config["sam_to_pre"]
shortform2pairs = config["shortform2pairs"]
merge_pairs = config["merge_pairs"]
juicer_tools = config["juicer_tools"]
count_ligations = config["count_ligations"]
stats_sub = config["stats_sub"]
merge_stats = config["merge_stats"]
SK_balance = config["SK_balance"]
split_chrom = config["split_chrom"]
convert_df2matrix = config["convert_df2matrix"]
call_TAD = config["call_TAD"]
call_subcompartment = config["call_subcompartment"]
call_loop = config["call_loop"]
onTAD = config["OnTAD"]

# ------------------------------------------------------------------->>>>>>
# Parameters
# ------------------------------------------------------------------->>>>>>
singleend = config["singleend"]
threads = config["threads"]
mapq = config["mapq"]
resolution10 = config["resolution"]
resolution50 = config["resolution"] * 5
genomeID = config["genomeID"]
site = config["site"]
ligation = set_ligation_junction(site)

# ------------------------------------------------------------------->>>>>>
# rule all
# ------------------------------------------------------------------->>>>>>
chimeric_output_done = []
for cell, bioreps in config["samples"].items():
    for biorep, techreps in bioreps.items():
        for techrep in techreps:
            chimeric_output_done.append(f"splits/{cell}_{biorep}.{techrep}.bam.done")

merged_bam_done = []
statistics = []
for cell, bioreps in config["samples"].items():
    for biorep in bioreps.keys():
        merged_bam_done.append(f"splits/{cell}_{biorep}_merged_sort.bam.done")
        statistics.append(f"splits/{cell}_{biorep}_inter.done")

convert_pairs = []
for cell, bioreps in config["samples"].items():
    for biorep in bioreps.keys():
        convert_pairs.append(f"aligned/{cell}_{biorep}_short2pairs.done")


rule all:
    input:
        chimeric_output_done,
        merged_bam_done,
        statistics,
        expand("aligned/{cell}_stats.done", cell=config["samples"].keys()),
        expand("aligned/{cell}_stats_hists_m.done", cell=config["samples"].keys()),
        convert_pairs,
        expand("aligned/{cell}_merge_pairs.done", cell=config["samples"].keys()),
        expand("aligned/{cell}.cool.done", cell=config["samples"].keys()),
        expand("aligned/{cell}.hic.done", cell=config["samples"].keys()),
        expand(
            "features/Contacts/{cell}_intra_contact.done",
            cell=config["samples"].keys(),
        ),
        expand(
            "features/Compartments/{cell}_subcompartments.done",
            cell=config["samples"].keys(),
        ),
        expand("features/TADs/{cell}_tads.done", cell=config["samples"].keys()),
        expand("features/Loops/{cell}_loops.done", cell=config["samples"].keys()),


# ------------------------------------------------------------------->>>>>>
# Alignment & chimera handling
# ------------------------------------------------------------------->>>>>>
# Align FASTQ
rule align:
    input:
        ref={genomePath},
        r1="fastq/{cell}_{biorep}.{techrep}_R1.fastq.gz",
    output:
        sam="splits/{cell}_{biorep}.{techrep}.sam",
        done="splits/{cell}_{biorep}.{techrep}.sam.done",
    params:
        threads={threads},
        prefix="splits/{cell}_{biorep}.{techrep}",
        r2="fastq/{cell}_{biorep}.{techrep}_R2.fastq.gz",
    conda:
        "snakemake"
    shell:
        """
        if [ -f {params.r2} ]; then
            # Paired-end alignment
            bash {count_ligations} -i {input.r1} -I {params.r2} -o {params.prefix} -l {ligation} -u {singleend}
            bwa mem -SP5M -t {params.threads} {input.ref} {input.r1} {params.r2} > {output.sam} && touch {output.done}
        else
            # Single-end alignment
            bash {count_ligations} -i {input.r1} -o {params.prefix} -l {ligation} -u {singleend}
            bwa mem -SP5M -t {params.threads} {input.ref} {input.r1} > {output.sam} && touch {output.done}
        fi
        """


# Handle Chimeric Reads
rule handle_chimeric_reads:
    input:
        sam="splits/{cell}_{biorep}.{techrep}.sam",
        done="splits/{cell}_{biorep}.{techrep}.sam.done",
    output:
        bam="splits/{cell}_{biorep}.{techrep}.bam",
        done="splits/{cell}_{biorep}.{techrep}.bam.done",
    params:
        singleend={singleend},
        threads={threads},
        stem="splits/{cell}_{biorep}.{techrep}_norm",
        sam2="splits/{cell}_{biorep}.{techrep}.sam2",
        site_file={site_file},
    conda:
        "snakemake"
    shell:
        """
        if [ "{params.site_file}" != "none" ]; then
            if [ {params.singleend} -eq 1 ]; then
                awk -v stem={params.stem} -v site_file={params.site_file} -v singleend=1 -f {chimeric_sam} {input.sam} | \
                samtools sort -m 5G -t cb -n -@ {params.threads} > {output.bam} && touch {output.done}
            else
                awk -v stem={params.stem} -v site_file={params.site_file} -f {chimeric_sam} {input.sam} | \
                samtools sort -m 5G -t cb -n -@ {params.threads} > {output.bam} && touch {output.done}
            fi
        else
            if [ {params.singleend} -eq 1 ]; then
                awk -v stem={params.stem} -v singleend=1 -f {chimeric_sam} {input.sam} | \
                samtools sort -m 5G -t cb -n -@ {params.threads} > {output.bam} && touch {output.done}
            else
                awk -v stem={params.stem} -f {chimeric_sam} {input.sam} > {params.sam2}
                awk -v avgInsertFile={params.stem}.txt.res.txt -f {adjust_insert_size} {params.sam2} | \
                samtools sort -m 5G -t cb -n -@ {params.threads} > {output.bam} && touch {output.done}
            fi
        fi
        if [ -f {output.done} ]; then
            rm {input.sam}
        fi
        """


# ------------------------------------------------------------------->>>>>>
# Post-processing
# ------------------------------------------------------------------->>>>>>
# Merge Sorted and Aliged Files
rule merge_bams:
    input:
        bam=lambda wildcards: expand(
            "splits/{cell}_{biorep}.{techrep}.bam",
            cell=wildcards.cell,
            biorep=wildcards.biorep,
            techrep=config["samples"][wildcards.cell][wildcards.biorep],
        ),
        done=lambda wildcards: expand(
            "splits/{cell}_{biorep}.{techrep}.bam.done",
            cell=wildcards.cell,
            biorep=wildcards.biorep,
            techrep=config["samples"][wildcards.cell][wildcards.biorep],
        ),
    output:
        bam="splits/{cell}_{biorep}_merged_sort.bam",
        done="splits/{cell}_{biorep}_merged_sort.bam.done",
    params:
        threads={threads},
    conda:
        "snakemake"
    shell:
        """
        if [ $(echo {input.bam} | wc -w) -gt 1 ]; then
            samtools merge -c -t cb -n -@ {params.threads} {output.bam} {input.bam}
        else
            cp {input.bam} {output.bam}
        fi
        touch {output.done}
        if [ -f {output.done} ]; then
            rm {input.bam}
        fi
        """


# Remove Duplicates
rule rm_dups:
    input:
        sorted_merged_bam=lambda wildcards: expand(
            "splits/{cell}_{biorep}_merged_sort.bam",
            cell=wildcards.cell,
            biorep=wildcards.biorep,
        ),
        done=lambda wildcards: expand(
            "splits/{cell}_{biorep}_merged_sort.bam.done",
            cell=wildcards.cell,
            biorep=wildcards.biorep,
        ),
    output:
        dedup_bam="splits/{cell}_{biorep}_dedup.bam",
        done="splits/{cell}_{biorep}_dedup.bam.done",
    params:
        threads={threads},
        dedup_sam="splits/{cell}_{biorep}_dedup.sam",
    conda:
        "snakemake"
    shell:
        """
        samtools view -@ {params.threads} -h {input.sorted_merged_bam} | awk -f {dups_sam} > {params.dedup_sam}
        samtools view -@ {params.threads} -b {params.dedup_sam} > {output.dedup_bam}  && touch {output.done}
        rm {params.dedup_sam}
        """


# Convert SAM to JUICER SHORT FORMAT
rule sam2shortform:
    input:
        dedup_bam=lambda wildcards: expand(
            "splits/{cell}_{biorep}_dedup.bam",
            cell=wildcards.cell,
            biorep=wildcards.biorep,
        ),
    output:
        short="splits/{cell}_{biorep}_short.txt",
        done="splits/{cell}_{biorep}_sam2shortform.done",
    params:
        threads={threads},
        mapq={mapq},
    conda:
        "snakemake"
    shell:
        """
        samtools view -@ {params.threads} -F 1024 -O sam {input.dedup_bam} | \
        awk -v mapq={params.mapq} -f {sam_to_pre} > {output.short} && touch {output.done}
        """


# Statistics
rule statistics:
    input:
        dedup_bam=lambda wildcards: expand(
            "splits/{cell}_{biorep}_dedup.bam",
            cell=wildcards.cell,
            biorep=wildcards.biorep,
        ),
        short=lambda wildcards: expand(
            "splits/{cell}_{biorep}_short.txt",
            cell=wildcards.cell,
            biorep=wildcards.biorep,
        ),
    output:
        stats="splits/{cell}_{biorep}_inter.txt",
        done="splits/{cell}_{biorep}_inter.done",
    params:
        threads={threads},
        stem="splits/{cell}_{biorep}*.res.txt",
        singleend={singleend},
        ligation={ligation},
        site_file={site_file},
        chromsize={chromsize},
    conda:
        "snakemake"
    shell:
        """
        if [ {params.singleend} -eq 1 ]; then
            ret=$(samtools view -@ {params.threads} -f 1024 -F 256 {input.dedup_bam} | awk '{{if ($0~/rt:A:7/){{singdup++}}else{{dup++}}}}END{{print dup,singdup}}')
            dups=$(echo $ret | awk '{{print $1}}')
            singdups=$(echo $ret | awk '{{print $2}}')
            cat {params.stem} | awk -v dups=$dups -v singdups=$singdups -v ligation={params.ligation} -v singleend=1 -f {stats_sub} > {output.stats}
        else
            dups=$(samtools view -c -f 1089 -F 256 -@ {params.threads} {input.dedup_bam})
            cat {params.stem} | awk -v dups=$dups -v ligation={params.ligation} -f {stats_sub} > {output.stats}
        fi
        java -Xms8000m -Xmx64000m -jar {juicer_tools} statistics {params.site_file} {output.stats} {input.short} {params.chromsize} && touch {output.done}
        """


# Convert JUICER SHORT FORMAT to PAIRS
rule short2pairs:
    input:
        short=lambda wildcards: expand(
            "splits/{cell}_{biorep}_short.txt",
            cell=wildcards.cell,
            biorep=wildcards.biorep,
        ),
        done=lambda wildcards: expand(
            "splits/{cell}_{biorep}_sam2shortform.done",
            cell=wildcards.cell,
            biorep=wildcards.biorep,
        ),
    output:
        done="aligned/{cell}_{biorep}_short2pairs.done",
    params:
        chromsize={chromsize},
        prefix="splits/{cell}_{biorep}_short",
        pairs="splits/{cell}_{biorep}_short.bsorted.pairs",
        pairs_gz="{cell}_{biorep}_short.bsorted.pairs.gz",
        intra_frag="aligned/{cell}_{biorep}_intra_frag.pairs",
        no_intrafrag="aligned/{cell}_{biorep}_short.pairs.gz",
    conda:
        "snakemake"
    shell:
        """
        {shortform2pairs} {input.short} {params.chromsize} {params.prefix}
        if [ ! -f "{params.pairs_gz}" ]; then
            bgzip -f {params.pairs}
        fi
        mv splits/{params.pairs_gz}* aligned/
        zcat aligned/{params.pairs_gz} | sed '1,2d' | awk -F'\t' '$2 == $4 && $8 == $9' > {params.intra_frag}
        gunzip -c aligned/{params.pairs_gz} | grep -Fxv -f {params.intra_frag} | bgzip -f > {params.no_intrafrag} && touch {output.done}
        rm aligned/{params.pairs_gz} {params.intra_frag}
        """


# ------------------------------------------------------------------->>>>>>
# Map creation
# ------------------------------------------------------------------->>>>>>
# Merge PAIRS
rule merge_pairs:
    input:
        lambda wildcards: expand(
            "aligned/{cell}_{biorep}_short2pairs.done",
            cell=wildcards.cell,
            biorep=list(config["samples"][wildcards.cell].keys()),
        ),
    output:
        pairs_gz="aligned/{cell}_short.pairs.gz",
        done="aligned/{cell}_merge_pairs.done",
    params:
        prefix="aligned/{cell}",
        pairs="aligned/{cell}_short.pairs",
    conda:
        "snakemake"
    shell:
        """
        file_count=$(ls aligned/*_short.pairs.gz 2> /dev/null | wc -l)
        if [ "$file_count" -gt 1 ]; then
            {merge_pairs} {params.prefix}_short aligned/*_short.pairs.gz
            if [ ! -f "{output.pairs_gz}" ]; then
                bgzip -f {params.pairs}
            fi
        else
            mv {params.prefix}_rep1_short.pairs.gz {output.pairs_gz}
        fi
        pairix -p pairs -f {output.pairs_gz} && touch {output.done}
        """


# Merge Statistical Files of Bio-replicates
rule merge_stats:
    input:
        stats=lambda wildcards: expand(
            "splits/{cell}_{biorep}_inter.txt",
            cell=wildcards.cell,
            biorep=list(config["samples"][wildcards.cell].keys()),
        ),
    output:
        stats_done="aligned/{cell}_stats.done",
        hists_m_done="aligned/{cell}_stats_hists_m.done",
    params:
        hists_m=lambda wildcards: expand(
            "splits/{cell}_{biorep}_inter_hists.m",
            cell=wildcards.cell,
            biorep=list(config["samples"][wildcards.cell].keys()),
        ),
        stats="aligned/{cell}_stats",
    conda:
        "snakemake"
    shell:
        """
        if [ $(echo {input.stats} | wc -w) -gt 1 ]; then
            java -Xmx2g -jar {merge_stats} {params.stats} {input.stats} && touch {output.stats_done}
            java -Xmx2g -jar {merge_stats} {params.stats} {params.hists_m} && touch {output.hists_m_done}
        else
            mv {input.stats} {params.stats}.txt && touch {output.stats_done}
            mv {params.hists_m} {params.stats}_hists.m && touch {output.hists_m_done}
        fi
        """


# Create .COOL file
rule create_cool:
    input:
        pairs_gz="aligned/{cell}_short.pairs.gz",
        done="aligned/{cell}_merge_pairs.done",
    output:
        cool10="aligned/{cell}_%s.cool" % resolution10,
        cool50="aligned/{cell}_%s.cool" % resolution50,
        done="aligned/{cell}.cool.done",
    params:
        resolution10=resolution10,
        resolution50=resolution50,
        threads={threads},
    conda:
        "snakemake"
    shell:
        """
        cooler cload pairix {mainonly_chromsize}:{params.resolution10} -p {params.threads} {input.pairs_gz} {output.cool10}
        cooler cload pairix {mainonly_chromsize}:{params.resolution50} -p {params.threads} {input.pairs_gz} {output.cool50}
        python {SK_balance} --max-iters 1000 -p 8 -i {output.cool10}
        python {SK_balance} --max-iters 1000 -p 8 -i {output.cool50}
        touch {output.done}
        """


# Create .HIC file
rule create_hic:
    input:
        pairs_gz="aligned/{cell}_short.pairs.gz",
        done="aligned/{cell}_merge_pairs.done",
    output:
        hic="aligned/{cell}.hic",
        done="aligned/{cell}.hic.done",
    params:
        threads={threads},
        genomeID={genomeID},
        site_file={site_file},
    conda:
        "snakemake"
    shell:
        """
        time java -Xms8000m -Xmx64000m -jar {juicer_tools} pre --threads {params.threads} -f {params.site_file} \
        -n -r 1000000,500000,250000,100000,50000,25000,10000,5000,2000 {input.pairs_gz} {output.hic} {params.genomeID}
        time java -Xms8000m -Xmx64000m -jar {juicer_tools} addNorm --threads {params.threads} -k KR,SCALE {output.hic} \
        && touch {output.done}

        """


# ------------------------------------------------------------------->>>>>>
# Featrue calculation
# ------------------------------------------------------------------->>>>>>
# Extract intra-CONTACTs
rule intra_contact:
    input:
        cool10="aligned/{cell}_%s.cool" % resolution10,
        cool50="aligned/{cell}_%s.cool" % resolution50,
    output:
        whole10="features/Contacts/{cell}_wholegenome_contact_%s.txt" % resolution10,
        intra10="features/Contacts/{cell}_intra_contact_%s.txt" % resolution10,
        whole50="features/Contacts/{cell}_wholegenome_contact_%s.txt" % resolution50,
        intra50="features/Contacts/{cell}_intra_contact_%s.txt" % resolution50,
        intra_done="features/Contacts/{cell}_intra_contact.done",
    conda:
        "snakemake"
    shell:
        """
        cooler dump --join -b -o {output.whole10} {input.cool10} &
        cooler dump --join -b -o {output.whole50} {input.cool50} &
        wait
        awk '$1 == $4' {output.whole10} > {output.intra10} &
        awk '$1 == $4' {output.whole50} > {output.intra50} &
        wait
        touch {output.intra_done}
        """


# Call COMPARTMENTs
rule Calder:
    input:
        intra10="features/Contacts/{cell}_intra_contact_%s.txt" % resolution10,
        intra50="features/Contacts/{cell}_intra_contact_%s.txt" % resolution50,
    output:
        subcomp10="features/Compartments/{cell}_subcompartments_%s.bed" % resolution10,
        subcomp50="features/Compartments/{cell}_subcompartments_%s.bed" % resolution50,
        done="features/Compartments/{cell}_subcompartments.done",
    conda:
        "snakemake"
    params:
        genomeID={genomeID},
        resolution10={resolution10},
        resolution50={resolution50},
        input_prefix="features/Compartments/{cell}",
        output="features/Compartments",
        subcomp="features/Compartments/sub_compartments/all_sub_compartments.bed",
    shell:
        """
        bash {split_chrom} -i {input.intra10} -O {params.input_prefix} -g {params.genomeID} -r {params.resolution10}
        bash {split_chrom} -i {input.intra50} -O {params.input_prefix} -g {params.genomeID} -r {params.resolution50}
        Rscript {call_subcompartment} -i {params.input_prefix} -o {params.output} -r {params.resolution10} -g {params.genomeID}
        mv {params.subcomp} {output.subcomp10}
        Rscript {call_subcompartment} -i {params.input_prefix} -o {params.output} -r {params.resolution50} -g {params.genomeID}
        mv {params.subcomp} {output.subcomp50}
        touch {output.done}
        """


# Call TADs
rule onTAD:
    input:
        cool10="aligned/{cell}_%s.cool" % resolution10,
        cool50="aligned/{cell}_%s.cool" % resolution50,
    output:
        tad10="features/TADs/{cell}_tads_%s.bed" % resolution10,
        tad50="features/TADs/{cell}_tads_%s.bed" % resolution50,
        done="features/TADs/{cell}_tads.done",
    conda:
        "snakemake"
    params:
        genomeID={genomeID},
        resolution10={resolution10},
        resolution50={resolution50},
        matrix_prefix10="features/TADs/{cell}_%s" % resolution10,
        matrix_prefix50="features/TADs/{cell}_%s" % resolution50,
    shell:
        """
        python {convert_df2matrix} -g {params.genomeID} -i {input.cool10} -o {params.matrix_prefix10} &
        python {convert_df2matrix} -g {params.genomeID} -i {input.cool50} -o {params.matrix_prefix50} &
        wait
        bash {call_TAD} -i {params.matrix_prefix10} -o {output.tad10} -r {params.resolution10} -g {params.genomeID} -s {onTAD}
        bash {call_TAD} -i {params.matrix_prefix50} -o {output.tad50} -r {params.resolution50} -g {params.genomeID} -s {onTAD}
        touch {output.done}        
        """


# Call LOOPs
rule cooltools_dots:
    input:
        cool="aligned/{cell}_%s.cool" % resolution10,
    output:
        loop="features/Loops/{cell}_loops_%s.tsv" % resolution10,
        done="features/Loops/{cell}_loops.done",
    conda:
        "snakemake"
    params:
        genomeID={genomeID},
        threads={threads},
        output_prefix="features/Loops/{cell}_loops",
    shell:
        """
        python {call_loop} -i {input.cool} -o {output.loop} -g {params.genomeID} -p {params.threads} \
        && touch {output.done}
        """

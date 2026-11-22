library(tidyverse)
library(rtracklayer)
library(optparse)

options(scipen = 999)

option_list <- list(
  make_option(c("-i", "--input"), type = "character", default = NULL, help = "input file path", dest = "input"),
  make_option(c("-o", "--output"), type = "character", default = NULL, help = "output file", dest = "output"),
  make_option(c("-g", "--gtf"), type = "character", default = NULL, help = "gtf file", dest = "gtf")
)
opt_parser <- OptionParser(option_list = option_list)
options <- parse_args(opt_parser)

input <- options$input
output <- options$output
gtf <- options$gtf

gtf_data <- as.data.frame(rtracklayer::import(gtf))
gtf_data <- filter(gtf_data, type == "gene") |>
  dplyr::select(seqnames, gene_id, gene_name, gene_type) |>
  filter(gene_type == "protein_coding") |>
  filter(!seqnames %in% c("chrM", "chrY"))
df <- gtf_data[, c("gene_id", "gene_name", "gene_type")]
df <- separate(df, gene_id, into = c("gene_id", "c"), sep = "[.]") |> dplyr::select(-c)
df <- unique(df)
rownames(df) <- NULL
df <- filter(df, !gene_name %in% names(which(table(df$gene_name) > 1)))

files <- list.files(input, pattern = "genes.results$", full.names = F)
f <- sub("_rep.*", "", files[1])
rna <- read_delim(file.path(input, files[1]), delim = "\t", escape_double = FALSE, trim_ws = TRUE) |>
  dplyr::select(gene_id, FPKM)
rna <- separate(rna, gene_id, into = c("gene_id", "c"), sep = "[.]") |> dplyr::select(-c)
rna <- rna[rna$gene_id %in% df$gene_id, ]
colnames(rna) <- c("gene_name", strsplit(files[1], "[.]")[[1]][1])
if (length(files) >= 2) {
  for (i in files[2:length(files)]) {
    exp <- read_delim(file.path(input, i), delim = "\t", escape_double = FALSE, trim_ws = TRUE) |>
      dplyr::select(gene_id, FPKM)
    exp <- separate(exp, gene_id, into = c("gene_id", "c"), sep = "[.]") |> dplyr::select(-c)
    exp <- exp[exp$gene_id %in% df$gene_id, ]
    strsplit(i, ",")
    colnames(exp) <- c("gene_name", strsplit(i, "[.]")[[1]][1])
    rna <- full_join(rna, exp, by = "gene_name")
  }
}
fpkm <- left_join(df, rna, by = c("gene_id" = "gene_name"))
fpkm <- dplyr::select(fpkm, -gene_type, -gene_id)
fpkm <- column_to_rownames(fpkm, var = "gene_name")
fpkm <- log2(fpkm + 1)
fpkm1 <- mutate(fpkm, !!f := rowMeans(fpkm, na.rm = T))
fpkm1 <- rownames_to_column(fpkm1, var = "gene_name")
fpkm1 <- dplyr::select(fpkm1, gene_name, !!f)
long_fpkm <- pivot_longer(fpkm1,
  cols = -c("gene_name"),
  names_to = "cell",
  values_to = "fpkm"
)
long_fpkm <- dplyr::select(long_fpkm, gene_name, fpkm)
long_fpkm <- arrange(long_fpkm, gene_name)
write.table(long_fpkm, file = output, quote = F, col.names = F, row.names = F, sep = "\t")

library(CALDER)
library(optparse)

option_list <- list(
  make_option(c("-i", "--input"), type = "character", default = NULL, help = "input directory", dest = "input"),
  make_option(c("-o", "--output"), type = "character", default = NULL, help = "choose output path", dest = "output"),
  make_option(c("-g", "--genomeID"), type = "character", default = NULL, help = "choose genomeID", dest = "genomeID"),
  make_option(c("-r", "--resolution"), type = "character", default = NULL, help = "resolution of Hi-C data", dest = "resolution")
)
opt_parser <- OptionParser(option_list = option_list)
options <- parse_args(opt_parser)

resolution <- options$resolution
input <- options$input
output <- options$output
genomeID <- options$genomeID

if (!dir.exists(file.path(output))) {
  dir.create(file.path(output))
}

if (genomeID %in% c("mm10", "mm9")) {
  chrs <- c("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "X")
}
if (genomeID %in% c("hg19", "hg38")) {
  chrs <- c("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "X")
}

contact_file_dump <- as.list(paste0(input, sprintf(paste0("_chr%s_", resolution, "_ob_for_Calder.txt.gz"), chrs)))
names(contact_file_dump) <- chrs

if (resolution == "1000000") bin_size <- 1E6
if (resolution == "500000") bin_size <- 500E3
if (resolution == "250000") bin_size <- 250E3
if (resolution == "100000") bin_size <- 100E3
if (resolution == "50000") bin_size <- 50E3
if (resolution == "25000") bin_size <- 25E3
if (resolution == "10000") bin_size <- 10E3
if (resolution == "5000") bin_size <- 5E3

CALDER(
  contact_file_dump = contact_file_dump,
  chrs = chrs,
  bin_size = bin_size,
  genome = genomeID,
  save_dir = file.path(output),
  save_intermediate_data = F,
  n_cores = 32,
  sub_domains = F
)

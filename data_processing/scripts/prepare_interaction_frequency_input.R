library(tidyverse)
library(optparse)

options(scipen = 999)

option_list <- list(
  make_option(c("-i", "--input"), type = "character", default = NULL, help = "input directory", dest = "input"),
  make_option(c("-o", "--output"), type = "character", default = NULL, help = "choose output path", dest = "output"),
  make_option(c("-r", "--resolution"), type = "numeric", default = NULL, help = "resolution", dest = "resolution"),
  make_option(c("-b", "--bed"), type = "character", default = NULL, help = "bed file of genes", dest = "bed")
)
opt_parser <- OptionParser(option_list = option_list)
options <- parse_args(opt_parser)

input <- options$input
bed <- options$bed
output <- options$output
resolution <- options$resolution

df <- read_delim(input,
  delim = "\t", escape_double = FALSE,
  col_names = FALSE, trim_ws = TRUE
)
colnames(df) <- c("X1", "X2", "X3", "IF")

bed <- read_delim(bed,
  delim = "\t",
  escape_double = FALSE,
  col_names = FALSE, trim_ws = TRUE
)

df <- left_join(bed, df, by = c("X1", "X2", "X3"))

df <- dplyr::select(df, X4, X6, IF)
wide_data <- df %>%
  pivot_wider(names_from = X6, values_from = IF) %>%
  arrange(X4)

if (resolution == 50000) {
  col_index <- c("X4", as.character(seq(-20, 20)))
}
if (resolution == 10000) {
  col_index <- c("X4", as.character(seq(-100, 100)))
}
if (resolution == 100000) {
  col_index <- c("X4", as.character(seq(-10, 10)))
}
if (resolution == 25000) {
  col_index <- c("X4", as.character(seq(-40, 40)))
}

for (col in col_index) {
  if (!col %in% names(wide_data)) {
    wide_data[[col]] <- NA
  }
}
wide_data <- wide_data[, col_index] %>% as.data.frame()
wide_data[is.na(wide_data)] <- 0
write.table(wide_data, file = output, quote = F, row.names = F, sep = "\t", col.names = F)

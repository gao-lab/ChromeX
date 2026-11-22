library(tidyverse)
library(optparse)

options(scipen = 999)

option_list <- list(
  make_option(c("-i", "--input"), type = "character", default = NULL, help = "input directory", dest = "input"),
  make_option(c("-o", "--output"), type = "character", default = NULL, help = "choose output path", dest = "output"),
  make_option(c("-r", "--resolution"), type = "numeric", default = NULL, help = "resolution", dest = "resolution")
)
opt_parser <- OptionParser(option_list = option_list)
options <- parse_args(opt_parser)
input <- options$input
output <- options$output
resolution <- options$resolution

df <- read_delim(input, delim = "\t", escape_double = FALSE, col_names = FALSE, trim_ws = TRUE)
df <- unique(df)
colnames(df) <- c("X1", "X2", "X3", "X4")
df_wide <- pivot_wider(df, names_from = X3, values_from = X4, id_cols = c(X1))

if (resolution == 50000) {
  col_index <- c("X1", as.character(seq(-20, 20)))
}
if (resolution == 10000) {
  col_index <- c("X1", as.character(seq(-100, 100)))
}
if (resolution == 100000) {
  col_index <- c("X1", as.character(seq(-10, 10)))
}
if (resolution == 25000) {
  col_index <- c("X1", as.character(seq(-40, 40)))
}

df_wide <- df_wide[, col_index] %>% as.data.frame()
df_wide[is.na(df_wide)] <- 0
write.table(df_wide, file = output, row.names = F, col.names = F, sep = "\t", quote = F)

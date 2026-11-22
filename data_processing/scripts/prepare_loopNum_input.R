library(tidyverse)
library(optparse)

options(scipen = 999)

option_list <- list(
  make_option(c("-i", "--input"), type = "character", default = NULL, help = "input directory", dest = "input"),
  make_option(c("-o", "--output"), type = "character", default = NULL, help = "choose output path", dest = "output"),
  make_option(c("-b", "--bed1"), type = "character", default = NULL, help = "bed file of bins", dest = "bed1"),
  make_option(c("-B", "--bed2"), type = "character", default = NULL, help = "bed file of genes", dest = "bed2"),
  make_option(c("-r", "--resolution"), type = "numeric", default = NULL, help = "resolution", dest = "resolution")
)
opt_parser <- OptionParser(option_list = option_list)
options <- parse_args(opt_parser)

input <- options$input
bed1 <- options$bed1
bed2 <- options$bed2
output <- options$output
resolution <- options$resolution

normalize_by_quantile <- function(df, p = 0.95) {
  qv <- sapply(df, function(x) {
    x_nonzero <- x[x != 0]
    return(quantile(x_nonzero, p, na.rm = TRUE))
  })
  df_norm <- as.data.frame(mapply(function(x, q) {
    x[x > q] <- q
    x / max(x, na.rm = TRUE)
  }, df, qv))
  rownames(df_norm) <- rownames(df)
  colnames(df_norm) <- colnames(df)
  return(df_norm)
}

df <- read_delim(input,
  delim = "\t", escape_double = FALSE,
  col_names = FALSE, trim_ws = TRUE
)
colnames(df) <- c("X1", "X2", "X3")
count <- df %>%
  group_by_all() %>%
  count() %>%
  ungroup()

# bed1 <- "/lustre/grp/gglab/zhout/workspace/snakemake_pipeline/reference/hg38_bin_10kb.bed.gz"
bed1 <- read_delim(bed1,
  delim = "\t",
  escape_double = FALSE,
  col_names = FALSE, trim_ws = TRUE
)
colnames(bed1) <- c("X1", "X2", "X3")

count <- left_join(bed1, count, by = c("X1", "X2", "X3"))
count[is.na(count)] <- 0
# count <- mutate(count, n = ifelse(n > 10, 10, n))
# count <- mutate(count, n = n / max(n, na.rm = TRUE))

# bed2 <- "/lustre/grp/gglab/zhout/workspace/snakemake_pipeline/reference/human_TSS_anno1M_10kb.bed.gz"
bed2 <- read_delim(bed2,
  delim = "\t",
  escape_double = FALSE,
  col_names = FALSE, trim_ws = TRUE
)
colnames(bed2) <- c("X1", "X2", "X3", "X4", "X5", "X6")
count <- left_join(bed2, count, by = c("X1", "X2", "X3"))
count <- dplyr::select(count, X4, X6, n)
wide_data <- count %>%
  pivot_wider(names_from = X6, values_from = n) %>%
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

wide_data <- column_to_rownames(wide_data, var = "X4")
dat_norm <- normalize_by_quantile(wide_data, 0.95)
dat_norm <- rownames_to_column(dat_norm, var = "X1")

write.table(dat_norm, file = output, quote = F, row.names = F, sep = "\t", col.names = F)

library(tidyverse)
library(optparse)

options(scipen = 999)

option_list <- list(
  make_option(c("-i", "--input"), type = "character", default = NULL, help = "input directory", dest = "input"),
  make_option(c("-r", "--resolution"), type = "numeric", default = NULL, help = "resolution", dest = "resolution"),
  make_option(c("-B", "--ref_bed"), type = "character", default = NULL, help = "referece", dest = "ref_bed"),
  make_option(c("-o", "--output"), type = "character", default = NULL, help = "choose output path", dest = "output")
)
opt_parser <- OptionParser(option_list = option_list)
options <- parse_args(opt_parser)

input <- options$input
resolution <- options$resolution
output <- options$output
ref_bed <- options$ref_bed

sub_comp <- read_delim(input,
  delim = "\t", escape_double = FALSE,
  col_names = FALSE, trim_ws = TRUE
)
sub_comp <- sub_comp[, c(1, 2, 3, 4)]

sub_comp <- mutate(sub_comp, X4 = str_extract(X4, "^[AB]\\.[12].[12][^.]*"))
sub_comp$X2 <- sub_comp$X2 - 1
sub_comp_perbin <- sub_comp %>%
  group_by(X1) %>%
  mutate(id = row_number()) %>%
  ungroup() %>%
  rowwise() %>%
  do({
    data <- .
    n <- ceiling((data$X3 - data$X2) / resolution)
    new_start <- seq(data$X2, by = resolution, length.out = n)
    new_end <- c(new_start[-1], data$X3)
    new_end <- pmin(new_end, data$X3)
    data.frame(
      chr = data$X1,
      start = new_start,
      end = new_end,
      string = data$X4
    )
  })

annotated <- read_delim(ref_bed,
  delim = "\t", escape_double = FALSE,
  col_names = FALSE, trim_ws = TRUE
)
annotated <- filter(annotated, !(X1 %in% c("chrY", "chrM")))
colnames(annotated) <- c("X1", "X2", "X3", "X4", "X5", "X6")
dat <- left_join(annotated, sub_comp_perbin, by = c("X1" = "chr", "X2" = "start", "X3" = "end"))
dat[is.na(dat)] <- "N"
dat <- mutate(dat, string = case_when(
  string == "A.1.1" ~ "a",
  string == "A.1.2" ~ "b",
  string == "A.2.1" ~ "c",
  string == "A.2.2" ~ "d",
  string == "B.1.1" ~ "e",
  string == "B.1.2" ~ "f",
  string == "B.2.1" ~ "g",
  string == "B.2.2" ~ "h",
  string == "N" ~ "N",
  TRUE ~ string
))
dat1 <- dplyr::select(dat, X4, X6, string)
wide_data <- dat1 %>% pivot_wider(names_from = X6, values_from = string)
wide_data[is.na(wide_data)] <- "N"

if (resolution == 10000) {
  cols_to_order <- names(wide_data)[2:202]
}
if (resolution == 25000) {
  cols_to_order <- names(wide_data)[2:82]
}
if (resolution == 50000) {
  cols_to_order <- names(wide_data)[2:42]
}
if (resolution == 100000) {
  cols_to_order <- names(wide_data)[2:22]
}
ordered_indices <- order(as.integer(cols_to_order))
wide_data <- wide_data %>% dplyr::select(1, all_of(cols_to_order[ordered_indices]))

if (resolution == 100000) {
  wide_data$merged_column <- apply(wide_data[, 2:22], 1, paste, collapse = "")
}
if (resolution == 50000) {
  wide_data$merged_column <- apply(wide_data[, 2:42], 1, paste, collapse = "")
}
if (resolution == 25000) {
  wide_data$merged_column <- apply(wide_data[, 2:82], 1, paste, collapse = "")
}
if (resolution == 10000) {
  wide_data$merged_column <- apply(wide_data[, 2:202], 1, paste, collapse = "")
}

wide_data1 <- dplyr::select(wide_data, X4, merged_column)
write.table(wide_data1, file = output, sep = "\t", row.names = F, col.names = F, quote = F)

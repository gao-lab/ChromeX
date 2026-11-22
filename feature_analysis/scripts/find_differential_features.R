library(tidyverse)
library(optparse)
library(parallel)

option_list <- list(
  make_option(c("--label_file"), type = "character", default = NULL, help = "cell-gene labels", dest = "labels"),
  make_option(c("--shap"), type = "character", default = NULL, help = "gene-node shap matrix", dest = "shap"),
  make_option(c("--binary"), type = "character", default = NULL, help = "gene-node binary matrix", dest = "binary"),
  make_option(c("--index_file"), type = "character", default = NULL, help = "Feature index file (2 cols: index, name)", dest = "index_file"),
  make_option(c("--output"), type = "character", default = NULL, help = "choose output path", dest = "output"),
  make_option(c("-p", "--threads"), type = "integer", default = 1, help = "Number of threads to use [default 1]", dest = "threads")
)
opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

labels <- opt$labels
shap <- opt$shap
binary <- opt$binary
index_file <- opt$index_file
output <- opt$output
cores <- opt$threads

dir.create(output, showWarnings = FALSE, recursive = TRUE)

find_DFs <- function(c1, c2, dim_df, ng_matrix) {
  clus1_genes <- filter(dim_df, cluster %in% c1)$gene
  clus2_genes <- filter(dim_df, cluster %in% c2)$gene
  all_neg_genes <- setdiff(rownames(ng_matrix), clus1_genes)

  nodes <- colnames(ng_matrix)

  TP_vec <- colSums(ng_matrix[clus1_genes, , drop = FALSE])
  n_c1 <- length(clus1_genes)
  FN_vec <- n_c1 - TP_vec

  FP_all_vec <- colSums(ng_matrix[all_neg_genes, , drop = FALSE])

  pct1_vec <- TP_vec / pmax(n_c1, 1L)
  pct1_safe <- ifelse(pct1_vec == 0, 0.001, pct1_vec)

  FP_vec <- colSums(ng_matrix[clus2_genes, , drop = FALSE]) # 在C2中的命中
  n_c2 <- length(clus2_genes)
  TN_vec <- n_c2 - FP_vec

  pct2_vec <- FP_vec / pmax(n_c2, 1L)
  pct2_safe <- ifelse(pct2_vec == 0, 0.001, pct2_vec)
  log2fc_vec <- log2(pct1_safe / pct2_safe)

  precision_all <- ifelse((TP_vec + FP_all_vec) == 0, 0, TP_vec / (TP_vec + FP_all_vec))
  recall_all <- ifelse((TP_vec + FN_vec) == 0, 0, TP_vec / (TP_vec + FN_vec))
  denom_all <- precision_all + recall_all
  f1_all_vec <- ifelse(denom_all == 0, 0, 2 * precision_all * recall_all / denom_all)

  fisher_p <- mapply(
    function(a, b, c, d) {
      suppressWarnings(stats::fisher.test(matrix(c(a, b, c, d), nrow = 2), alternative = "two.sided")$p.value)
    },
    TP_vec, FP_vec, FN_vec, TN_vec
  )

  qval_vec <- p.adjust(fisher_p)

  df <- data.frame(
    cluster1 = c1,
    cluster2 = c2,
    node = nodes,
    pct1 = pct1_vec,
    log2fc = log2fc_vec,
    f1_all = f1_all_vec,
    qval = qval_vec,
    stringsAsFactors = FALSE
  )

  df <- subset(df, qval < 0.01 & log2fc >= 1 & pct1 >= 0.5)

  if (nrow(df) == 0) {
    return(data.frame())
  }
  df[, c("cluster1", "cluster2", "node", "pct1", "log2fc", "f1_all")]
}

feature_index <- read_delim(index_file,
  delim = "\t", escape_double = FALSE,
  col_names = FALSE, trim_ws = TRUE
)
colnames(feature_index) <- c("index", "name")
map <- setNames(feature_index$name, feature_index$index)

shap_filtered <- read_csv(shap, show_col_types = F)
shap_filtered <- column_to_rownames(shap_filtered, var = colnames(shap_filtered)[1])

binary <- read_csv(binary, show_col_types = F)
binary <- column_to_rownames(binary, var = colnames(binary)[1])
nm <- names(binary)
key <- sub(">=.*$", "", nm)
val <- unname(map[key])

names(binary) <- paste0(ifelse(is.na(val), key, val), sub("^[^>]+", "", nm))

binary_filtered <- binary[rownames(shap_filtered), colnames(shap_filtered), drop = FALSE]
colnames(binary_filtered) <- gsub("_", "-", colnames(binary_filtered))
df_new <- 1 - binary_filtered
colnames(df_new) <- gsub(">=", "<", colnames(binary_filtered), fixed = TRUE)
full_mat <- cbind(binary_filtered, df_new)

labels <- read_delim(labels,
  delim = "\t", escape_double = FALSE,
  col_names = T, trim_ws = TRUE
)
labels <- na.omit(labels)
clusters <- as.vector(sort(unique(labels$cluster)))
nodes_df <- data.frame()
for (i in clusters) {
  js <- clusters[clusters != i]
  ij_nodes_list <- mclapply(
    js,
    function(j, i, dim_df, ng_matrix) {
      print(c(i, j))
      ij_nodes <- find_DFs(
        c1 = i, c2 = j, dim_df = dim_df, ng_matrix = ng_matrix
      )
    },
    i, labels, full_mat,
    mc.cores = cores
  )
  ij_nodes <- do.call(rbind, ij_nodes_list)
  ij_nodes$node_type <- ij_nodes$node %>%
    sub("([<>]=?)[0-9.]+$", "\\1", .)
  ij_nodes <- ij_nodes[!duplicated(ij_nodes$node), ]
  ij_nodes <- ij_nodes %>%
    group_by(node_type) %>%
    slice_max(order_by = abs(f1_all), n = 1, with_ties = TRUE) %>%
    ungroup() %>%
    as.data.frame()
  nodes_df <- rbind(nodes_df, ij_nodes)
}

rep_nodes <- data.frame()
for (c in unique(nodes_df$cluster1)) {
  df <- nodes_df[nodes_df$cluster1 == c, c("cluster1", "node")] %>%
    .[!duplicated(.), ]
  rep_nodes <- rbind(rep_nodes, df)
}

clean_nodes <- unique(nodes_df$node)
clean_mat <- full_mat[, clean_nodes, drop = F]
write.table(clean_mat, file = paste0(output, "/train_gene_DFs_mat.txt"), row.names = T, col.names = T, sep = "\t", quote = F)

colnames(rep_nodes) <- c("cluster", "nodes")
write.table(rep_nodes, file = paste0(output, "/differential_nodes.txt"), row.names = F, col.names = T, sep = "\t", quote = F)

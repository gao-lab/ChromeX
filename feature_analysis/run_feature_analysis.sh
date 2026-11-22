#!/usr/bin/env bash

python extract_decision_nodes.py \
    -D ../model_training/files/human_train_val_set.csv \
    -P train -m ../model_training/files/ChromeX_human.joblib \
    -n human -N human -o results -i hdf5_path/human -I hdf5_path/human -L results \
    -F files/xgboost_human_node_42.joblib \
    --gain_threshold 0.03 --node node_information_on_human_gain0.03.txt

python extract_decision_nodes.py \
    -D ../model_training/files/human_train_val_set.csv \
    -P predict -m ../model_training/files/ChromeX_human.joblib \
    -n human -N human -o results -i hdf5_path/human -I hdf5_path/human -L results \
    -F ./files/xgboost_h20_node_42.joblib \
    --gain_threshold 0.03 --node node_information_on_h20_model_gain0.03.txt

python extract_decision_nodes.py \
    -D ../model_training/files/human_train_val_set.csv \
    -P shap -m ../model_training/files/ChromeX_human.joblib \
    -n human -N human -o results -i hdf5_path/human -I hdf5_path/human \
    -L results -F ./files/xgboost_h20_node_42.joblib --gain_threshold 0.03 \
    --node node_information_on_h20_model_gain0.03.txt \
    --binary ./files/human_train_binary_mat.csv \
    --shap ./files/human_train_shap_mat.csv

python run_pca_snn_leiden.py \
    --input files/human_train_shap_mat.csv files/human_test_shap_mat.csv \
    --output results --index_file reference/feature_index_name_association.txt \
    --seed 100 --n_components 64 --k_query 100 --top_m 64 --min_weight 3.0

python cluster_by_coassociation_matrix.py \
    --assignments results/leiden_assignments_res0.?.csv \
    --output results --resolution 0.2

Rscript find_differential_features.R \
    --label_file results/human_train_cluster_label.txt \
    --shap files/train_shap_filtered.csv \
    --binary files/human_train_binary_mat.csv \
    --index_file reference/feature_index_name_association.txt \
    --output results -p 30

for cluster in $(awk 'NR>1 {print $1}' results/differential_nodes.txt | sort -nu); do
    python run_fpgrowth.py \
        --target_cluster ${cluster} \
        --gn_mat results/train_gene_DFs_mat.txt \
        --labels results/human_train_cluster_label.txt \
        --DF results/differential_nodes.txt \
        --exe /path/to/fpgrowth \
        --minsup 0.5 --maxsup 1 --top_k 10000 \
        --max_len 6 --output results
done

python cluster_label_model.py \
    --input files/human_train_shap_mat.csv \
    --output results \
    -L results/human_train_cluster_label.txt \
    --n_components 128 --seed 0 -P train \
    -m files/gene_cluster_prediction_model.joblib

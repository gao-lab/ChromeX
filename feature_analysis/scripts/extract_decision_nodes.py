import sys, h5py, os

import pandas as pd
import numpy as np
import joblib, glob, argparse

from sklearn.utils import resample
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import scipy.stats as stats
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import Lasso
import shap


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog,
            max_help_position=40,
            width=100,
        )
    )
    parser.add_argument(
        "-D",
        "--dataset",
        type=str,
        required=False,
        help="Train datasets list.",
        dest="datalist",
    )
    parser.add_argument(
        "-P",
        "--process",
        type=str,
        required=True,
        help="Build model or predict gene expression. [required]",
        dest="process",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=False,
        help="Model save path. [required]",
        dest="model",
    )
    parser.add_argument(
        "-n",
        "--modelname",
        type=str,
        required=False,
        help="The name of training dataset.",
        dest="model_name",
    )
    parser.add_argument(
        "-N",
        "--predictname",
        type=str,
        required=False,
        help="The name of the data used for predicting.",
        dest="predict_name",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        help="Result path. [required]",
        dest="output",
    )
    parser.add_argument(
        "-i",
        "--traindata",
        type=str,
        required=False,
        help="Train data path.",
        dest="train_path",
    )
    parser.add_argument(
        "-I",
        "--testdata",
        type=str,
        required=False,
        help="Test data path.",
        dest="test_path",
    )
    parser.add_argument(
        "-L",
        "--lasso",
        type=str,
        required=False,
        help="Lasso model save path. [required]",
        dest="lasso",
    )
    parser.add_argument(
        "-F",
        "--feature",
        type=str,
        required=False,
        help="Xgboost model for nodes save path. [required]",
        dest="feature",
    )
    parser.add_argument(
        "--gain_threshold",
        type=float,
        required=False,
        default=0.03,
        help="Proportion of top-ranked gain values for decision nodes obtained from the species-based XGBoost model to be preserved, between 0 and 1.",
        dest="gain_threshold",
    )
    parser.add_argument(
        "--node",
        type=str,
        required=False,
        help="Dataframe of nodes file.",
        dest="node",
    )
    parser.add_argument(
        "--binary",
        type=str,
        required=False,
        help="Gene-node binary matrix.",
        dest="binary",
    )
    parser.add_argument(
        "--shap",
        type=str,
        required=False,
        help="Gene-node shap matrix.",
        dest="shap",
    )
    return parser.parse_args()


def removeN(AB10, AB50):
    nonzero_indices10 = np.argwhere(np.all(AB10[:, 95:106] != 0, axis=1))
    nonzero_indices50 = np.argwhere(AB50[:, 20] != 0)
    nonzero_indices = np.intersect1d(
        nonzero_indices10.flatten(), nonzero_indices50.flatten()
    )
    return nonzero_indices


def transform_expression(Y):
    Y = np.log(Y + 1)
    Y_max = Y.max()
    Y = Y / Y_max
    return Y, Y_max


def load_dataset(data_file):
    f = h5py.File(data_file, "r")
    gene_order = f["gene_order"][:]
    Y = f["Y"][:]
    gene_len = f["X_gene_len"][:]
    AB = f["X_AB"][:]
    AB_pre = f["X_AB_pre"][:]
    genebodyAB = f["X_genebodyAB"][:]
    loop = f["X_loop"][:]
    boundary_level = f["X_boundary_level"][:]
    region_level = f["X_region_level"][:]
    interaction_frequency = f["X_interaction_frequency"][:]
    f.close()

    nonzero_indices = removeN(AB_pre, AB)
    gene_order = [x.decode("utf-8") for x in gene_order]
    gene_order = [gene_order[i] for i in nonzero_indices.flatten()]
    Y = Y[nonzero_indices]
    gene_len = gene_len[nonzero_indices]
    AB = AB[nonzero_indices]
    genebodyAB = genebodyAB[nonzero_indices]
    loop = loop[nonzero_indices]
    boundary_level = boundary_level[nonzero_indices]
    region_level = region_level[nonzero_indices]
    interaction_frequency = interaction_frequency[nonzero_indices]
    return (
        gene_order,
        Y,
        gene_len,
        AB,
        genebodyAB,
        boundary_level,
        interaction_frequency,
        region_level,
        loop,
    )


def stratified_sampling(Y, quantiles=[20, 40, 60, 80], test_size=0.1, random_state=42):
    quantile_values = np.percentile(Y, quantiles)
    Y_class = np.digitize(Y, bins=quantile_values, right=True)

    data = pd.DataFrame(Y, columns=["Y"])
    data["Y_class"] = Y_class
    train_data, _ = train_test_split(
        data, test_size=test_size, stratify=data["Y_class"], random_state=random_state
    )

    idx_train = train_data.index
    return idx_train


def split_dataset(
    gene_order,
    Y,
    gene_len,
    AB,
    genebodyAB,
    boundary_level,
    interaction_frequency,
    region_level,
    loop,
    upsampling=True,
    sample_seed=42,
    sample_size=1,
):
    test_size = 1 - sample_size
    idx_train = stratified_sampling(
        Y, quantiles=[20, 40, 60, 80], test_size=test_size, random_state=sample_seed
    )

    gene_train = [gene_order[pos] for pos in idx_train]

    Y_train = Y[idx_train]

    X_train_AB = AB[idx_train]
    X_train_genebodyAB = genebodyAB[idx_train]
    X_train_loop = loop[idx_train]
    X_train_boundary_level = boundary_level[idx_train]
    X_train_region_level = region_level[idx_train]
    X_train_interaction_frequency = interaction_frequency[idx_train]
    X_train_gene_len = gene_len[idx_train]
    X_train_gene_len = np.asarray(X_train_gene_len).reshape(-1, 1)

    Train_all = [
        X_train_AB,
        X_train_genebodyAB,
        X_train_boundary_level,
        X_train_interaction_frequency,
        X_train_region_level,
        X_train_loop,
        X_train_gene_len,
    ]
    X_train = np.concatenate(Train_all, axis=1)

    return X_train, Y_train, gene_train


def build_feature_matrix(
    AB, genebodyAB, boundary_level, interaction_frequency, region_level, loop, gene_len
):
    gene_len = np.asarray(gene_len).reshape(-1, 1)
    X = [
        AB,
        genebodyAB,
        boundary_level,
        interaction_frequency,
        region_level,
        loop,
        gene_len,
    ]
    return np.concatenate(X, axis=1)


def merge_cell_data(cell_list, data_path, sample_seed=42, sample_size=1):
    X_list = []
    Y_list = []
    gene_order_list = []
    cell_order_list = []

    for cell in cell_list:
        data_file = os.path.join(data_path, f"{cell}_data.hdf5")
        if not os.path.isfile(data_file):
            raise FileNotFoundError(f"Error: {data_file} does not exist.")

        (
            gene_order,
            Y,
            gene_len,
            AB,
            genebodyAB,
            boundary_level,
            interaction_frequency,
            region_level,
            loop,
        ) = load_dataset(data_file)
        if sample_size == 1:
            X_cell = build_feature_matrix(
                AB,
                genebodyAB,
                boundary_level,
                interaction_frequency,
                region_level,
                loop,
                gene_len,
            )
            Y_list.append(Y)
            X_list.append(X_cell)
            gene_order_list.append(gene_order)
            cell_order_list.append([cell] * len(gene_order))
        else:
            X_cell, Y_cell, gene_cell = split_dataset(
                gene_order,
                Y,
                gene_len,
                AB,
                genebodyAB,
                boundary_level,
                interaction_frequency,
                region_level,
                loop,
                upsampling=True,
                sample_seed=sample_seed,
                sample_size=sample_size,
            )
            Y_list.append(Y_cell)
            X_list.append(X_cell)
            gene_order_list.append(gene_cell)
            cell_order_list.append([cell] * len(gene_cell))

    X_all = np.concatenate(X_list, axis=0)
    Y_all = np.concatenate(Y_list, axis=0)
    gene_order_all = np.concatenate(gene_order_list, axis=0).tolist()
    cell_order_all = np.concatenate(cell_order_list, axis=0).tolist()

    return X_all, Y_all, gene_order_all, cell_order_all


def create_datasets(
    Train_cell, Val_cell, data_path, upsampling=True, sample_seed=42, sample_size=1
):
    Train_cell = sorted(Train_cell)
    X_train, Y_train, gene_order_train, cell_order_train = merge_cell_data(
        Train_cell, data_path, sample_seed=sample_seed, sample_size=sample_size
    )
    Y_train, Y_max = transform_expression(Y_train)

    if Val_cell is not None:
        Val_cell = sorted(Val_cell)
        X_val, Y_val, gene_order_val, cell_order_val = merge_cell_data(
            Val_cell, data_path, sample_seed=sample_seed, sample_size=sample_size
        )
        Y_val = np.log(Y_val + 1)
        Y_val = Y_val / Y_max
    else:
        X_val, Y_val, gene_order_val, cell_order_val = None, None, None, None

    over_threshold_indices = np.where(Y_train > 0.75)[0]
    upsample = min(int((len(Y_train) / len(over_threshold_indices)) / 4), 5)
    if upsample > 0 and upsampling:
        X_train_high = X_train[over_threshold_indices]
        Y_train_high = Y_train[over_threshold_indices]
        X_train_resampled, Y_train_resampled = resample(
            X_train_high,
            Y_train_high,
            replace=True,
            n_samples=len(X_train_high) * upsample,
            random_state=42,
        )
        X_train = np.concatenate([X_train, X_train_resampled], axis=0)
        Y_train = np.concatenate([Y_train, Y_train_resampled], axis=0)
    return (
        X_train,
        Y_train,
        X_val,
        Y_val,
        gene_order_train,
        gene_order_val,
        Y_max,
        cell_order_train,
        cell_order_val,
    )


def node_robust(
    cell_list, train_path, lasso_model_path, node_ids, result_path, sample_size=0.9
):
    seed = [67, 12, 45, 42, 89, 3, 76, 28, 94, 51]
    node_df = pd.DataFrame()
    for j in range(0, 100):
        for i in seed:
            lasso_model_file = os.path.join(
                lasso_model_path, f"lasso_{model_name}_{i}.joblib"
            )
            if os.path.isfile(lasso_model_file):
                lasso_model = joblib.load(lasso_model_file)["model"]
            else:
                lasso_model, _ = build_lasso_model(
                    cell_list,
                    train_path,
                    lasso_model_file,
                    node_ids,
                    random_state=i,
                    sample_seed=j,
                    sample_size=sample_size,
                )
            signif_features, signif_coefs = extract_significant_nodes(
                lasso_model, threshold_t=0
            )
            node_info = node_ids.iloc[signif_features].copy()
            node_info = node_list.merge(
                node_info[["Feature", "Split"]], on=["Feature", "Split"], how="inner"
            )
            node_info.to_csv(
                os.path.join(
                    result_path,
                    f"node_information_on_{model_name}_gain{gain_threshold}_{j}_{i}.txt",
                ),
                sep="\t",
                index=False,
            )
            node_df = pd.concat([node_df, node_info], ignore_index=True)
        # build node model
        node_df.drop_duplicates(inplace=True, keep="first")
        node_df.index = range(node_df.shape[0])
        node_df.to_csv(
            os.path.join(
                result_path,
                f"node_information_on_{model_name}_gain{gain_threshold}_{j}.txt",
            ),
            sep="\t",
            index=False,
        )


def extract_tree_nodes(xgb_model, gain_threshold):
    trees_df = xgb_model.get_booster().trees_to_dataframe()
    internal_nodes = trees_df[trees_df["Feature"] != "Leaf"]
    internal_nodes = internal_nodes[
        internal_nodes["Gain"] >= internal_nodes["Gain"].quantile(1 - gain_threshold)
    ]
    internal_nodes.index = range(internal_nodes.shape[0])
    return internal_nodes


def build_gene_node_mat(X, internal_nodes, gene=None):
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=feature_names)
    gene_node_mat = np.zeros((X.shape[0], internal_nodes.shape[0]), dtype=int)
    gene_node_col = []

    for col_idx in range(internal_nodes.shape[0]):
        feature = internal_nodes.loc[internal_nodes.index[col_idx], "Feature"]
        threshold = internal_nodes.loc[internal_nodes.index[col_idx], "Split"]
        feat_values = X[feature].values
        gene_node_mat[:, col_idx] = (feat_values >= threshold).astype(int)
        gene_node_col.append(feature + ">=" + str(threshold))

    if gene is not None:
        gene_node_df = pd.DataFrame(gene_node_mat, columns=gene_node_col, index=gene)
    else:
        gene_node_df = None
    return gene_node_mat, gene_node_df


def build_lasso_model(
    cell_list,
    train_path,
    lasso_model_file,
    node_ids,
    random_state=42,
    sample_seed=42,
    sample_size=1,
):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_val_mse = float("inf")
    # define the hyperparameter search space for grid search
    param_grid = {"alpha": [5e-4, 1e-4, 5e-5], "max_iter": [5000]}

    dataset_cache = {}
    data_set = {}
    fold = 0
    for train_index, val_index in kf.split(cell_list):
        fold = fold + 1
        train_data = [cell_list[i] for i in train_index]
        val_data = [cell_list[i] for i in val_index]
        X_train, Y_train, X_val, Y_val, _, _, _, _, _ = create_datasets(
            train_data,
            val_data,
            train_path,
            upsampling=True,
            sample_seed=sample_seed,
            sample_size=sample_size,
        )
        X_train_node, _ = build_gene_node_mat(X_train, node_ids)
        X_val_node, _ = build_gene_node_mat(X_val, node_ids)
        dataset_cache[fold] = [X_train_node, Y_train, X_val_node, Y_val]
        data_set[fold] = [train_data, val_data]
    # grid search
    for alpha in param_grid["alpha"]:
        for max_iter in param_grid["max_iter"]:
            model = Lasso(
                alpha=alpha,
                max_iter=max_iter,
                random_state=random_state,
                selection="random",
            )
            mse_all = []
            for i in range(len(dataset_cache)):
                X_train, Y_train, X_val, Y_val = dataset_cache[i + 1]
                model.fit(X_train, Y_train)
                Y_val_pred = model.predict(X_val)
                val_mse = mean_squared_error(Y_val, Y_val_pred)
                mse_all.append(val_mse)
            mse_mean = np.mean(mse_all)
            if mse_mean < best_val_mse:
                best_val_mse = mse_mean
                best_params = {
                    "alpha": alpha,
                    "max_iter": max_iter,
                }
    best_model = Lasso(
        alpha=best_params["alpha"],
        max_iter=best_params["max_iter"],
        random_state=random_state,
        selection="random",
    )
    X_train, Y_train, _, _, _, _, Y_max, _, _ = create_datasets(
        Train_cell=cell_list,
        Val_cell=None,
        data_path=train_path,
        upsampling=True,
        sample_seed=sample_seed,
        sample_size=sample_size,
    )
    X_train_node, _ = build_gene_node_mat(X_train, node_ids)
    best_model.fit(X_train_node, Y_train)
    # save the best-performing model
    if not os.path.exists(os.path.dirname(lasso_model_file)):
        os.makedirs(os.path.dirname(lasso_model_file))
    joblib.dump(
        {"model": best_model, "Y_max": Y_max, "node_ids": node_ids}, lasso_model_file
    )

    resultlist = {
        "Train_cell": [y[0] for y in data_set.values()],
        "Validation_cell": [y[1] for y in data_set.values()],
        "final_model_mse_mean": [best_val_mse],
    }
    return best_model, resultlist


def extract_significant_nodes(lasso_model, threshold_t=0):
    coefs = lasso_model.coef_
    signif_idx = np.where(np.abs(coefs) > threshold_t)[0]
    signif_coefs = coefs[signif_idx]
    signif_features = signif_idx.tolist()
    return signif_features, signif_coefs.tolist()


def build_xgb_model(cell_list, train_path, node_model_file, node_df, random_state=42):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_val_mse = float("inf")
    # define the hyperparameter search space for grid search
    param_grid = {
        "n_estimators": [500, 400, 300],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.3, 0.1, 0.05],
        "gamma": [0.4, 0.3, 0.2, 0.1, 0],
        "subsample": [0.7, 0.8, 0.9, 1],
    }
    dataset_cache = {}
    data_set = {}
    fold = 0
    for train_index, val_index in kf.split(cell_list):
        fold = fold + 1
        train_data = [cell_list[i] for i in train_index]
        val_data = [cell_list[i] for i in val_index]
        X_train, Y_train, X_val, Y_val, _, _, _, _, _ = create_datasets(
            train_data, val_data, train_path, upsampling=True
        )
        X_train_node, _ = build_gene_node_mat(X_train, node_df)
        X_val_node, _ = build_gene_node_mat(X_val, node_df)
        dataset_cache[fold] = [X_train_node, Y_train, X_val_node, Y_val]
        data_set[fold] = [train_data, val_data]
    # grid search
    for n_estimators in param_grid["n_estimators"]:
        for learning_rate in param_grid["learning_rate"]:
            for max_depth in param_grid["max_depth"]:
                for gamma in param_grid["gamma"]:
                    for subsample in param_grid["subsample"]:
                        model = xgb.XGBRegressor(
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                            n_estimators=n_estimators,
                            random_state=random_state,
                            gamma=gamma,
                            subsample=subsample,
                        )
                        mse_all = []
                        for i in range(len(dataset_cache)):
                            X_train, Y_train, X_val, Y_val = dataset_cache[i + 1]
                            model.fit(X_train, Y_train)
                            Y_val_pred = model.predict(X_val)
                            val_mse = mean_squared_error(Y_val, Y_val_pred)
                            mse_all.append(val_mse)
                        mse_mean = np.mean(mse_all)
                        print(mse_all)
                        if mse_mean < best_val_mse:
                            print(f"best_val_mse: {mse_all}")
                            best_val_mse = mse_mean
                            best_params = {
                                "max_depth": max_depth,
                                "learning_rate": learning_rate,
                                "n_estimators": n_estimators,
                                "gamma": gamma,
                                "subsample": subsample,
                            }
    best_model = xgb.XGBRegressor(
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        n_estimators=best_params["n_estimators"],
        random_state=random_state,
        gamma=best_params["gamma"],
        subsample=best_params["subsample"],
    )
    (
        X_train,
        Y_train,
        _,
        _,
        _,
        _,
        Y_max,
        _,
        _,
    ) = create_datasets(
        Train_cell=cell_list, Val_cell=None, data_path=train_path, upsampling=True
    )
    X_train_node, _ = build_gene_node_mat(X_train, node_df)
    best_model.fit(X_train_node, Y_train)
    # save the best-performing model
    if not os.path.exists(os.path.dirname(node_model_file)):
        os.makedirs(os.path.dirname(node_model_file))
    joblib.dump(
        {"model": best_model, "Y_max": Y_max, "node_ids": node_df}, node_model_file
    )

    resultlist = {
        "Train_cell": [y[0] for y in data_set.values()],
        "Validation_cell": [y[1] for y in data_set.values()],
        "final_model_mse_mean": [best_val_mse],
    }
    return resultlist


def gene_prediction(cell_list, model, file_path_list, model_name, Y_max, node_ids):
    resultlist = {
        "model": [],
        "predict_cell": [],
        "predict": [],
        "label": [],
        "isTrain": [],
        "gene_name": [],
    }
    for predict_cell in file_path_list:
        (
            gene_order,
            Y_real,
            gene_len,
            AB,
            genebodyAB,
            boundary_level,
            interaction_frequency,
            region_level,
            loop,
        ) = load_dataset(predict_cell)
        X = build_feature_matrix(
            AB,
            genebodyAB,
            boundary_level,
            interaction_frequency,
            region_level,
            loop,
            gene_len,
        )
        X_node, gene_node_df = build_gene_node_mat(X, node_ids, gene_order)
        Y_pred = np.exp(model.predict(X_node) * Y_max) - 1
        predict_cell = os.path.basename(predict_cell).rsplit("_", 1)[0]
        if predict_cell in cell_list:
            resultlist["isTrain"].extend([1] * len(Y_pred))
        else:
            resultlist["isTrain"].extend([0] * len(Y_pred))
        resultlist["gene_name"].extend(gene_order)
        resultlist["model"].extend([model_name] * len(Y_pred))
        resultlist["predict_cell"].extend([predict_cell] * len(Y_pred))
        resultlist["predict"].extend(list(Y_pred))
        resultlist["label"].extend(list(Y_real))
    return resultlist


def compute_shap(model, mat):
    explainer = shap.TreeExplainer(model)
    shap_mat = explainer.shap_values(mat)
    return shap_mat


def main():
    args = parse_args()
    train_path = args.train_path
    test_path = args.test_path
    result_path = args.output
    xgb_model_file = args.model
    lasso_model_path = args.lasso
    node_model_file = args.feature
    gain_threshold = args.gain_threshold
    process = args.process
    datalist = args.datalist
    model_name = args.model_name
    predict_name = args.predict_name
    node_file = args.node
    binary_file = args.binary
    shap_file = args.shap

    if datalist == None:
        cell_list = []
    else:
        datalist_df = pd.read_csv(datalist, header=None)
        cell_list = datalist_df.iloc[0].tolist()

    if process == "train":
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        xgb_model = joblib.load(xgb_model_file)["model"]
        node_list = extract_tree_nodes(xgb_model, gain_threshold=gain_threshold)
        node_ids = node_list[["Feature", "Split"]].drop_duplicates()
        node_ids.index = range(node_ids.shape[0])
        # get 10 random seed for LASSO regression
        seed = [67, 12, 45, 42, 89, 3, 76, 28, 94, 51]
        node_df = pd.DataFrame()
        for i in seed:
            lasso_model_file = os.path.join(
                lasso_model_path, f"lasso_{model_name}_{i}.joblib"
            )
            if os.path.isfile(lasso_model_file):
                lasso_model = joblib.load(lasso_model_file)["model"]
            else:
                lasso_model, _ = build_lasso_model(
                    cell_list,
                    train_path,
                    lasso_model_file,
                    node_ids,
                    random_state=i,
                    sample_seed=42,
                    sample_size=1,
                )
            signif_features, signif_coefs = extract_significant_nodes(
                lasso_model, threshold_t=0
            )
            node_info = node_ids.iloc[signif_features].copy()
            node_info = node_list.merge(
                node_info[["Feature", "Split"]], on=["Feature", "Split"], how="inner"
            )
            node_info.to_csv(
                os.path.join(
                    result_path,
                    f"node_information_on_{model_name}_model_gain{gain_threshold}_seed{i}.txt",
                ),
                sep="\t",
                index=False,
            )
            node_df = pd.concat([node_df, node_info], ignore_index=True)
        # build node model
        node_df.drop_duplicates(inplace=True, keep="first")
        node_df.index = range(node_df.shape[0])
        node_df.to_csv(
            os.path.join(
                result_path,
                f"node_information_on_{model_name}_model_gain{gain_threshold}.txt",
            ),
            sep="\t",
            index=False,
        )
        node_df = node_df[["Feature", "Split"]].drop_duplicates()
        node_df.index = range(node_df.shape[0])

        resultlist = build_xgb_model(cell_list, train_path, node_model_file, node_df)
        with open(
            os.path.join(result_path, f"{model_name}_node_model_performance.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            for key, value in resultlist.items():
                if isinstance(value, list):
                    value_str = "\n".join(map(str, value))
                else:
                    value_str = str(value)
                f.write(f"{key}:\n{value_str}\n\n")

    elif process == "predict":
        if test_path is None:
            raise ValueError("Test data path is required.")
        if not os.path.isfile(node_model_file):
            raise FileNotFoundError(f"Error: {node_model_file} does not exist.")
        test_path_list = glob.glob(os.path.join(test_path, "*.hdf5"))
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        model_obj = joblib.load(node_model_file)
        node_model = model_obj["model"]
        Y_max = model_obj["Y_max"]
        node_ids = model_obj["node_ids"]
        final_resultlist = gene_prediction(
            cell_list, node_model, test_path_list, model_name, Y_max, node_ids
        )
        final_resultlist = pd.DataFrame(final_resultlist)
        final_resultlist.to_csv(
            os.path.join(
                result_path,
                f"{predict_name}_gene_prediction_on_{model_name}_node_model.txt",
            ),
            sep="\t",
            index=False,
        )

    elif process == "shap":
        X, _, _, _, gene_order, _, _, cell_order, _ = create_datasets(
            Train_cell=cell_list, Val_cell=None, data_path=train_path, upsampling=False
        )
        node_df = pd.read_csv(node_file, header=0, sep="\t")
        node_df = node_df[["Feature", "Split"]].drop_duplicates()
        node_df.index = range(node_df.shape[0])
        name_list = [f"{cell}~{gene}" for gene, cell in zip(gene_order, cell_order)]
        _, gene_node_mat = build_gene_node_mat(X, node_df, name_list)
        gene_node_mat.to_csv(binary_file, index=True, header=True)

        node_model = joblib.load(node_model_file)["model"]
        feature_names = list(gene_node_mat.columns)
        shap_mat = compute_shap(node_model, gene_node_mat)
        shap_df = pd.DataFrame(
            shap_mat, index=gene_node_mat.index, columns=feature_names
        )
        shap_df.to_csv(shap_file, index=True, header=True)
    else:
        raise ValueError(
            "Invalid argument: The '-P/--process' parameter must be either 'train', 'predict' or 'shap'."
        )


if __name__ == "__main__":
    main()

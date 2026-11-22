#!/usr/bin/env python3
"""Train or apply the cluster-label classification model."""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog,
            max_help_position=40,
            width=100,
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Gene-Shap matrix.", 
        dest="input",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Result path. [required]",
        dest="output",
    )
    parser.add_argument(
        "-L",
        "--label",
        required=False,
        help="Cluster label file.",
        dest="label",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=128,
        required=False,
        help="",
        dest="n_components",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=0,
        help="",
        dest="seed",
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
        required=True,
        help="Model save path. [required]",
        dest="model",
    )
    return parser.parse_args()


def macro_auroc_ovr(y_true, y_proba):
    return roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")


def micro_auroc_ovr(y_true, y_proba):
    return roc_auc_score(y_true, y_proba, multi_class="ovr", average="micro")


def macro_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    total = cm.sum()

    specs = []
    for k in range(n_classes):
        tp = cm[k, k]
        fn = cm[k, :].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = total - tp - fn - fp
        denom = tn + fp
        specs.append(tn / denom if denom > 0 else 0.0)

    return float(np.mean(specs))


def export_macro_roc_pr_data(
    y_true, y_proba, n_classes, output, prefix="test", n_grid=1000
):
    y_proba = np.asarray(y_proba)
    classes = np.arange(n_classes)
    y_true_bin = label_binarize(y_true, classes=classes)

    # Binary classification compatibility.
    if n_classes == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    # Macro-averaged ROC data.
    fpr = {}
    tpr = {}
    valid_roc_classes = []

    for i in range(n_classes):
        # ROC requires both positive and negative samples.
        if len(np.unique(y_true_bin[:, i])) < 2:
            continue

        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        valid_roc_classes.append(i)

    if valid_roc_classes:
        all_fpr = np.unique(np.concatenate([fpr[i] for i in valid_roc_classes]))
        mean_tpr = np.zeros_like(all_fpr)

        for i in valid_roc_classes:
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= len(valid_roc_classes)
        macro_auroc = auc(all_fpr, mean_tpr)

        roc_df = pd.DataFrame(
            {
                "fpr": all_fpr,
                "tpr": mean_tpr,
                "macro_auroc": macro_auroc,
            }
        )
        roc_df.to_csv(
            os.path.join(output, f"{prefix}_macro_roc_curve.txt"),
            sep="\t",
            index=False,
        )

    # Macro-averaged PR data.
    precision = {}
    recall = {}
    average_precision = {}
    valid_pr_classes = []

    for i in range(n_classes):
        # PR curve requires at least one positive sample.
        if np.sum(y_true_bin[:, i]) == 0:
            continue

        precision[i], recall[i], _ = precision_recall_curve(
            y_true_bin[:, i], y_proba[:, i]
        )

        # recall returned by precision_recall_curve is descending.
        recall_i = recall[i][::-1]
        precision_i = precision[i][::-1]

        # Remove duplicated recall values for stable interpolation.
        tmp_df = pd.DataFrame({"recall": recall_i, "precision": precision_i})
        tmp_df = (
            tmp_df.groupby("recall", as_index=False)
            .agg({"precision": "max"})
            .sort_values("recall")
        )

        recall[i] = tmp_df["recall"].values
        precision[i] = tmp_df["precision"].values
        average_precision[i] = average_precision_score(y_true_bin[:, i], y_proba[:, i])
        valid_pr_classes.append(i)

    if valid_pr_classes:
        mean_recall = np.linspace(0, 1, n_grid)
        mean_precision = np.zeros_like(mean_recall)

        for i in valid_pr_classes:
            mean_precision += np.interp(mean_recall, recall[i], precision[i])

        mean_precision /= len(valid_pr_classes)
        macro_auprc = auc(mean_recall, mean_precision)
        macro_average_precision = np.mean(
            [average_precision[i] for i in valid_pr_classes]
        )

        pr_df = pd.DataFrame(
            {
                "recall": mean_recall,
                "precision": mean_precision,
                "macro_auprc": macro_auprc,
                "macro_average_precision": macro_average_precision,
            }
        )
        pr_df.to_csv(
            os.path.join(output, f"{prefix}_macro_pr_curve.txt"),
            sep="\t",
            index=False,
        )


def model_evaluation(
    X,
    y,
    n_classes,
    test_size=0.2,
    seed=0,
    random_state=42,
    output=None,
    n_components=128,
):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    best_estimator, cleaned_best_params, _, cv_metrics = grid_search(
        X_train,
        y_train,
        n_classes,
        n_components=n_components,
        seed=seed,
    )
    y_pred = best_estimator.predict(X_test)
    y_proba = best_estimator.predict_proba(X_test)

    if output is not None:
        export_macro_roc_pr_data(
            y_true=y_test,
            y_proba=y_proba,
            n_classes=n_classes,
            output=output,
            prefix="test",
        )

    test_metrics = {
        "ACC": accuracy_score(y_test, y_pred),
        "MacroF1": f1_score(y_test, y_pred, average="macro"),
        "MacroAUROC": roc_auc_score(
            y_test, y_proba, multi_class="ovr", average="macro"
        ),
        "MicroAUROC": roc_auc_score(
            y_test, y_proba, multi_class="ovr", average="micro"
        ),
        "MacroSEN": recall_score(y_test, y_pred, average="macro"),
        "MacroSPE": macro_specificity(y_test, y_pred),
        "MacroPRE": precision_score(y_test, y_pred, average="macro", zero_division=0),
    }
    return cleaned_best_params, cv_metrics, test_metrics


def grid_search(X_train, y_train, n_classes, random_state=42, n_components=128, seed=0):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    pipe = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("pca", PCA(n_components=n_components, random_state=seed)),
            (
                "clf",
                xgb.XGBClassifier(
                    objective="multi:softprob",
                    num_class=n_classes,
                    tree_method="hist",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )

    param_grid = {
        "clf__n_estimators": [300, 400, 500],
        "clf__max_depth": [3, 4, 5],
        "clf__learning_rate": [0.3, 0.1],
        "clf__gamma": [0.4, 0.3, 0.2, 0.1, 0.0],
        "clf__subsample": [0.6, 0.7, 0.8, 0.9, 1],
        "clf__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1],
    }

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=skf,
        n_jobs=-1,
        refit=True,
        verbose=2,
    )
    grid.fit(X_train, y_train)

    best_estimator = grid.best_estimator_
    best_params = grid.best_params_
    scoring = {
        "f1_macro": make_scorer(f1_score, average="macro"),
        "acc": make_scorer(accuracy_score),
        "macro_pre": make_scorer(precision_score, average="macro", zero_division=0),
        "macro_sen": make_scorer(recall_score, average="macro", zero_division=0),
        "macro_spe": make_scorer(macro_specificity),
        "macro_auroc": make_scorer(macro_auroc_ovr, needs_proba=True),
        "micro_auroc": make_scorer(micro_auroc_ovr, needs_proba=True),
    }

    cv2 = cross_validate(
        grid.best_estimator_,
        X_train,
        y_train,
        cv=skf,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    cv_metrics = {
        k: f"{np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}"
        for k, v in {
            "MacroF1": cv2["test_f1_macro"],
            "ACC": cv2["test_acc"],
            "MacroAUROC": cv2["test_macro_auroc"],
            "MicroAUROC": cv2["test_micro_auroc"],
            "MacroSEN": cv2["test_macro_sen"],
            "MacroSPE": cv2["test_macro_spe"],
            "MacroPRE": cv2["test_macro_pre"],
        }.items()
    }

    cleaned_best_params = {k.replace("clf__", ""): v for k, v in best_params.items()}
    print(f"Final optimal params: {cleaned_best_params}, Best metrics: {cv_metrics}")
    return best_estimator, cleaned_best_params, best_params, cv_metrics


def build_model(
    df,
    model_file,
    output=None,
    n_components=128,
    test_size=0.2,
    seed=0,
    random_state=42,
):
    X = df.loc[:, ~df.columns.isin(["gene", "cluster"])].to_numpy()
    y_raw = df["cluster"].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)

    best_params, cv_metrics, test_metrics = model_evaluation(
        X,
        y,
        n_classes,
        test_size=test_size,
        seed=seed,
        output=output,
        n_components=n_components,
    )
    pipe = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("pca", PCA(n_components=n_components, random_state=seed)),
            (
                "clf",
                xgb.XGBClassifier(
                    objective="multi:softprob",
                    num_class=n_classes,
                    tree_method="hist",
                    n_jobs=-1,
                    random_state=random_state,
                    **best_params,
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    model_bundle = {
        "pipeline": pipe,
        "label_encoder": le,
        "best_params": best_params,
        "PCs": n_components,
    }
    joblib.dump(model_bundle, model_file)
    resultlist = {
        "Final_optimal_params": best_params,
        "cv_metrics": cv_metrics,
        "testset_metrics": test_metrics,
    }
    return resultlist


def label_prediction(df_test, model_obj):
    pip = model_obj["pipeline"]
    le = model_obj["label_encoder"]
    X_test = df_test.loc[:, ~df_test.columns.isin(["gene", "cluster"])].to_numpy()
    class_names = le.classes_

    y_pred = pip.predict(X_test)
    y_proba = pip.predict_proba(X_test)
    y_pred_label = le.inverse_transform(y_pred)
    pred_df = pd.DataFrame(
        {
            "gene": df_test["gene"].values,
            "cluster": y_pred_label,
        }
    )
    pred_df["cluster"] = pred_df["cluster"].astype(int)
    prob_cols = [str(cls) for cls in class_names]
    proba_df = pd.DataFrame(y_proba, columns=prob_cols)
    proba_df.insert(0, "gene", df_test["gene"].values)
    return proba_df, pred_df


def write_resultlist(resultlist, output_file):
    with open(output_file, "w", encoding="utf-8") as handle:
        for key, value in resultlist.items():
            value_str = (
                "\n".join(map(str, value)) if isinstance(value, list) else str(value)
            )
            handle.write(f"{key}:\n{value_str}\n\n")


def run_train(args):
    os.makedirs(args.output, exist_ok=True)
    df_raw = pd.read_csv(args.input, index_col=0).reset_index(names="gene")
    df_label = pd.read_csv(args.label, sep="\t")
    df_label = df_label.dropna().reset_index(drop=True)
    df = df_raw.merge(df_label, how="inner", on="gene", validate="1:1")
    resultlist = build_model(
        df,
        args.model,
        output=args.output,
        n_components=args.n_components,
        test_size=0.2,
        seed=args.seed,
        random_state=42,
    )
    write_resultlist(
        resultlist,
        os.path.join(args.output, "cluster_label_model_performance.txt"),
    )


def run_predict(args):
    os.makedirs(args.output, exist_ok=True)
    df_test = pd.read_csv(args.input, index_col=0).reset_index(names="gene")
    model_obj = joblib.load(args.model)
    proba_df, pred_df = label_prediction(df_test, model_obj)

    proba_filtered = proba_df[proba_df.drop(columns="gene").max(axis=1) >= 0.8]
    pred_filtered = pred_df[pred_df["gene"].isin(proba_filtered["gene"])]

    proba_df.to_csv(
        os.path.join(args.output, "predict_cluster_probability.txt"),
        index=False,
        sep="\t",
    )
    pred_df.to_csv(
        os.path.join(args.output, "predict_cluster_label.txt"), index=False, sep="\t"
    )
    pred_filtered.to_csv(
        os.path.join(args.output, "predict_cluster_label_filtered.txt"),
        index=False,
        sep="\t",
    )


def main():
    args = parse_args()
    if args.process == "train":
        run_train(args)
    elif args.process == "predict":
        run_predict(args)
    else:
        raise ValueError(
            "Invalid argument: The '-P/--process' parameter must be either 'train' or 'predict'."
        )


if __name__ == "__main__":
    main()

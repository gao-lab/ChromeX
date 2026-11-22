#!/usr/bin/env python3

"""Train and apply ChromeX model.

ChromeX model trains on multiple celltype/dataset HDF5 files and uses
celltype-level KFold splits for hyperparameter selection.

Examples
--------
Train:
    python ChromeX.py \
        -P train \
        -D references/train_val_set.csv \
        -i data/human \
        -m results/ChromeX_human.joblib \
        -n human \
        -o results

Predict:
    python ChromeX.py \
        -P predict \
        -D references/train_val_set.csv \
        -I data/human \
        -m results/ChromeX_human.joblib \
        -n human \
        -N human \
        -o results
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from src.common import RANDOM_SEED
from src.io import (
    configure_logging,
    dump_model_bundle,
    ensure_dir,
    load_dataset,
    load_model_bundle,
    read_name_list,
    require_dir,
    require_file,
    write_table,
)
from src.modeling import make_xgb_regressor
from src.preprocessing import (
    ChromeXDataset,
    inverse_normalize_expression,
    normalize_expression,
    normalize_expression_with_ymax,
    upsample_high_expression,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class MatrixDataset:
    """Feature matrix, expression values and gene names merged across datasets."""

    x: np.ndarray
    y: np.ndarray
    gene_order: List[str]


@dataclass
class SpeciesSplit:
    """Train/validation arrays for one celltype-level split."""
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: Optional[np.ndarray]
    y_val: Optional[np.ndarray]
    train_cells: List[str]
    val_cells: Optional[List[str]]
    y_max: float


def dataset_file_for_cell(data_path: str, cell_name: str) -> Path:
    """Return the expected HDF5 path for a celltype/dataset name."""
    return Path(data_path) / f"{cell_name}_data.hdf5"


def merge_cell_data(cell_list: Sequence[str], data_path: str) -> MatrixDataset:
    """Load and concatenate all HDF5 datasets for the given celltypes."""
    if not cell_list:
        raise ValueError("cell list is empty.")

    require_dir(data_path, "data directory")

    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    gene_list: List[str] = []

    for cell in cell_list:
        dataset = load_dataset(dataset_file_for_cell(data_path, cell))
        x_list.append(dataset.feature_matrix())
        y_list.append(dataset.y)
        gene_list.extend(dataset.gene_order)

    return MatrixDataset(
        x=np.concatenate(x_list, axis=0),
        y=np.nan_to_num(
            np.concatenate(y_list, axis=0), nan=0.0, posinf=0.0, neginf=0.0
        ),
        gene_order=gene_list,
    )


def create_datasets(
    train_cells: Sequence[str],
    val_cells: Optional[Sequence[str]],
    data_path: str,
    *,
    upsampling: bool = True,
    random_state: int = RANDOM_SEED,
) -> SpeciesSplit:
    """Create train/validation matrices for ChromeX training."""
    train_cell_names = sorted(train_cells)
    train = merge_cell_data(train_cell_names, data_path)
    y_train, y_max = normalize_expression(train.y)

    x_train = train.x
    if upsampling:
        x_train, y_train, factor = upsample_high_expression(
            x_train,
            y_train,
            random_state=random_state,
        )

    if val_cells is None:
        return SpeciesSplit(
            x_train=x_train,
            y_train=y_train,
            x_val=None,
            y_val=None,
            train_cells=train_cell_names,
            val_cells=None,
            y_max=y_max,
        )

    val_cell_names = sorted(val_cells)
    val = merge_cell_data(val_cell_names, data_path)
    y_val = normalize_expression_with_ymax(val.y, y_max)

    return SpeciesSplit(
        x_train=x_train,
        y_train=y_train,
        x_val=val.x,
        y_val=y_val,
        train_cells=train_cell_names,
        val_cells=val_cell_names,
        y_max=y_max,
    )


def train_one_xgb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    params: Dict[str, float],
    random_state: int = RANDOM_SEED,
):
    """Instantiate and fit one XGBRegressor."""
    model = make_xgb_regressor(
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        n_estimators=int(params["n_estimators"]),
        gamma=float(params["gamma"]),
        subsample=float(params["subsample"]),
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    return model


def find_best_model(
    cell_list: Sequence[str],
    train_path: str,
    model_file: str,
    *,
    random_state: int = RANDOM_SEED,
) -> Dict[str, object]:
    """Run celltype-level KFold grid search, fit final model and save it."""
    if Path(model_file).is_file():
        raise FileExistsError(
            f"Model file already exists: {model_file}. "
            "Remove it or choose a different -m/--model path."
        )

    cells = list(cell_list)
    if len(cells) < 2:
        raise ValueError("ChromeX requires at least 2 training cells/datasets.")

    n_splits = min(5, len(cells))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_cache: Dict[int, SpeciesSplit] = {}
    fold_cells: Dict[int, Tuple[List[str], List[str]]] = {}

    for fold_id, (train_index, val_index) in enumerate(kf.split(cells), start=1):
        train_cells = [cells[i] for i in train_index]
        val_cells = [cells[i] for i in val_index]
        split = create_datasets(
            train_cells,
            val_cells,
            train_path,
            upsampling=True,
            random_state=random_state,
        )
        fold_cache[fold_id] = split
        fold_cells[fold_id] = (train_cells, val_cells)

    LOGGER.info("Dataset splitting is done. Starting grid search...")

    param_grid = {
        "n_estimators": [400, 500],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.3, 0.1],
        "gamma": [0.4, 0.3, 0.2, 0.1, 0.0],
        "subsample": [0.7, 0.8, 0.9, 1.0],
    }

    best_params: Optional[Dict[str, float]] = None
    best_val_mse = float("inf")

    for n_estimators, learning_rate, max_depth, gamma, subsample in product(
        param_grid["n_estimators"],
        param_grid["learning_rate"],
        param_grid["max_depth"],
        param_grid["gamma"],
        param_grid["subsample"],
    ):
        params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "gamma": gamma,
            "subsample": subsample,
        }

        fold_mse: List[float] = []
        for split in fold_cache.values():
            if split.x_val is None or split.y_val is None:
                raise RuntimeError("Validation split is missing.")
            model = train_one_xgb(
                split.x_train,
                split.y_train,
                params=params,
                random_state=random_state,
            )
            y_val_pred = model.predict(split.x_val)
            fold_mse.append(float(mean_squared_error(split.y_val, y_val_pred)))

        mse_mean = float(np.mean(fold_mse))
        LOGGER.info("params=%s fold_mse=%s mse_mean=%.6g", params, fold_mse, mse_mean)

        if mse_mean < best_val_mse:
            best_val_mse = mse_mean
            best_params = params

    if best_params is None:
        raise RuntimeError("Grid search did not produce best parameters.")

    final_split = create_datasets(
        cells,
        None,
        train_path,
        upsampling=True,
        random_state=random_state,
    )
    final_model = train_one_xgb(
        final_split.x_train,
        final_split.y_train,
        params=best_params,
        random_state=random_state,
    )

    dump_model_bundle(
        model_file,
        model=final_model,
        y_max=final_split.y_max,
        metadata={
            "model_type": "species-level",
            "best_params": best_params,
            "fold_mse_mean": best_val_mse,
            "random_state": random_state,
            "train_cells": cells,
        },
    )

    return {
        "Train_cell": [pair[0] for pair in fold_cells.values()],
        "Validation_cell": [pair[1] for pair in fold_cells.values()],
        "final_model_mse_mean": best_val_mse,
        "best_params": best_params,
    }


def predict_cell_name(hdf5_path: Path) -> str:
    """Derive celltype/dataset name from '<name>_data.hdf5'."""
    stem = hdf5_path.stem
    return stem.rsplit("_", 1)[0] if stem.endswith("_data") else stem


def predict_genes(
    *,
    train_cells: Sequence[str],
    model,
    hdf5_files: Sequence[Path],
    y_max: float,
    model_name: str,
) -> pd.DataFrame:
    """Predict gene expression for all HDF5 files in a test directory."""
    result = {
        "model": [],
        "predict_cell": [],
        "predict": [],
        "label": [],
        "isTrain": [],
        "gene_name": [],
    }
    train_cell_set = set(train_cells)

    for hdf5_file in sorted(hdf5_files):
        dataset: ChromeXDataset = load_dataset(hdf5_file)
        x = dataset.feature_matrix()
        y_pred = inverse_normalize_expression(model.predict(x), y_max)

        cell_name = predict_cell_name(hdf5_file)
        is_train = int(cell_name in train_cell_set)

        result["model"].extend([model_name] * len(y_pred))
        result["predict_cell"].extend([cell_name] * len(y_pred))
        result["predict"].extend(list(y_pred))
        result["label"].extend(list(dataset.y))
        result["isTrain"].extend([is_train] * len(y_pred))
        result["gene_name"].extend(dataset.gene_order)

    return pd.DataFrame(result)


def write_performance_text(result: Dict[str, object], output_file: Path) -> None:
    """Write ChromeX grid-search summary in the original text style."""
    if output_file.parent and str(output_file.parent) != ".":
        output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as handle:
        for key, value in result.items():
            if isinstance(value, list):
                value_str = "\n".join(map(str, value))
            else:
                value_str = str(value)
            handle.write(f"{key}:\n{value_str}\n\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or apply ChromeX model.",
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog,
            max_help_position=40,
            width=100,
        ),
    )
    parser.add_argument(
        "-D",
        "--dataset",
        required=True,
        dest="datalist",
        help="Dataset/celltype name list. [required]",
    )
    parser.add_argument(
        "-P",
        "--process",
        required=True,
        choices=("train", "predict"),
        dest="process",
        help="Run mode: train or predict. [required]",
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        dest="model",
        help="Model joblib path. [required]",
    )
    parser.add_argument(
        "-n",
        "--modelname",
        required=True,
        dest="model_name",
        help="Name of the training dataset/model. [required]",
    )
    parser.add_argument(
        "-N",
        "--predictname",
        dest="predict_name",
        help="Name of the prediction dataset.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        dest="output",
        help="Result directory. [required]",
    )
    parser.add_argument(
        "-i",
        "--traindata",
        dest="train_path",
        help="The directory of HDF5 files for training.",
    )
    parser.add_argument(
        "-I",
        "--testdata",
        dest="test_path",
        help="The directory of HDF5 files for prediction.",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED, help="Random seed. Default: 42"
    )
    return parser


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()

    cell_list = read_name_list(require_file(args.datalist, "dataset list"))
    result_dir = ensure_dir(args.output)

    if args.process == "train":
        if not args.train_path:
            raise ValueError("Training requires -i/--traindata.")
        result = find_best_model(
            cell_list,
            args.train_path,
            args.model,
            random_state=args.seed,
        )
        write_performance_text(
            result, result_dir / f"{args.model_name}_model_performance.txt"
        )

    elif args.process == "predict":
        if not args.test_path:
            raise ValueError("Prediction requires -I/--testdata.")
        if not args.predict_name:
            raise ValueError("Prediction requires -N/--predictname.")
        test_dir = require_dir(args.test_path, "prediction HDF5 directory")
        hdf5_files = sorted(test_dir.glob("*.hdf5"))
        if not hdf5_files:
            raise FileNotFoundError(f"No .hdf5 files found in {test_dir}")

        model, y_max, _ = load_model_bundle(require_file(args.model, "model file"))
        if y_max is None:
            raise ValueError(
                "ChromeX model file does not contain Y_max. "
                "Please retrain or provide a new-style ChromeX model bundle."
            )

        out = predict_genes(
            train_cells=cell_list,
            model=model,
            hdf5_files=hdf5_files,
            y_max=y_max,
            model_name=args.model_name,
        )
        write_table(
            out,
            result_dir
            / f"{args.predict_name}_gene_prediction_on_{args.model_name}_model.txt",
        )


if __name__ == "__main__":
    main()

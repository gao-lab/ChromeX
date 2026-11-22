#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train and apply the celltype-level XGBoost model.

This script keeps the original command-line interface, while reusable code is
split across:
- src/io.py: file/path/model/table I/O
- src/preprocessing.py: feature construction and expression transforms
- src/modeling.py: XGBoost helpers and metrics
- src/common.py: shared constants

Examples
--------
Train:
    python celltype_model.py \
        -P train \
        -c GM12878 \
        -i data/GM12878_data.hdf5 \
        -m results/GM12878_celltype.joblib \
        -o results/celltype

Predict:
    python celltype_model.py \
        -P predict \
        -c K562 \
        -C K562 \
        -I data/K562_data.hdf5 \
        -m results/K562_celltype.joblib \
        -o results/celltype
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src.common import RANDOM_SEED
from src.io import (
    configure_logging,
    dump_model_bundle,
    ensure_dir,
    load_dataset,
    load_model_bundle,
    require_file,
    write_table,
)
from src.modeling import (
    fit_with_optional_early_stopping,
    make_xgb_regressor,
    safe_pearsonr,
)
from src.preprocessing import (
    ChromeXDataset,
    inverse_normalize_expression,
    normalize_expression,
    normalize_expression_with_ymax,
    upsample_high_expression,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class CelltypeSplit:
    """Container for train/validation/test arrays of one celltype dataset."""
    x_train: np.ndarray
    y_train: np.ndarray
    y_train_raw: np.ndarray
    genes_train: List[str]

    x_val: np.ndarray
    y_val: np.ndarray
    y_val_raw: np.ndarray
    genes_val: List[str]

    x_test: np.ndarray
    y_test: np.ndarray
    y_test_raw: np.ndarray
    genes_test: List[str]

    y_max: float

    @property
    def x_all(self) -> np.ndarray:
        return np.concatenate([self.x_train, self.x_val, self.x_test], axis=0)

    @property
    def y_all_raw(self) -> np.ndarray:
        return np.concatenate(
            [self.y_train_raw, self.y_val_raw, self.y_test_raw], axis=0
        )

    @property
    def genes_all(self) -> List[str]:
        return list(self.genes_train) + list(self.genes_val) + list(self.genes_test)

    @property
    def is_train_like(self) -> np.ndarray:
        """Match the original output: train + validation = 1, test = 0."""
        return np.concatenate(
            [
                np.ones(len(self.y_train) + len(self.y_val), dtype=int),
                np.zeros(len(self.y_test), dtype=int),
            ]
        )


def stratified_sampling(
    y: np.ndarray,
    *,
    quantiles: Tuple[int, ...] = (20, 40, 60, 80),
    test_size: float = 0.4,
    valid_size: float = 0.5,
    random_state: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split genes into train/validation/test while preserving expression bins."""
    y_clean = np.nan_to_num(np.asarray(y, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    quantile_values = np.percentile(y_clean, quantiles)
    y_class = np.digitize(y_clean, bins=quantile_values, right=True)

    data = pd.DataFrame({"y": y_clean, "y_class": y_class})
    stratify = data["y_class"] if data["y_class"].value_counts().min() >= 2 else None

    if stratify is None:
        LOGGER.warning(
            "Some expression bins have <2 genes; falling back to non-stratified split."
        )

    train_data, temp_data = train_test_split(
        data,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state,
    )

    temp_stratify = (
        temp_data["y_class"] if temp_data["y_class"].value_counts().min() >= 2 else None
    )
    valid_data, test_data = train_test_split(
        temp_data,
        test_size=valid_size,
        stratify=temp_stratify,
        random_state=random_state,
    )
    return (
        train_data.index.to_numpy(),
        valid_data.index.to_numpy(),
        test_data.index.to_numpy(),
    )


def split_dataset(
    dataset: ChromeXDataset,
    *,
    upsampling: bool = True,
    random_state: int = RANDOM_SEED,
) -> CelltypeSplit:
    """Create train/validation/test matrices for a celltype model."""
    train_idx, val_idx, test_idx = stratified_sampling(
        dataset.y, random_state=random_state
    )

    x_all = dataset.feature_matrix()
    y_all = np.nan_to_num(dataset.y, nan=0.0, posinf=0.0, neginf=0.0)

    x_train, x_val, x_test = x_all[train_idx], x_all[val_idx], x_all[test_idx]
    y_train_raw, y_val_raw, y_test_raw = (
        y_all[train_idx],
        y_all[val_idx],
        y_all[test_idx],
    )

    y_train, y_max = normalize_expression(y_train_raw)
    y_val = normalize_expression_with_ymax(y_val_raw, y_max)
    y_test = normalize_expression_with_ymax(y_test_raw, y_max)

    genes_train = [dataset.gene_order[i] for i in train_idx]
    genes_val = [dataset.gene_order[i] for i in val_idx]
    genes_test = [dataset.gene_order[i] for i in test_idx]

    if upsampling:
        x_train, y_train, factor = upsample_high_expression(
            x_train,
            y_train,
            random_state=random_state,
        )

    return CelltypeSplit(
        x_train=x_train,
        y_train=y_train,
        y_train_raw=y_train_raw,
        genes_train=genes_train,
        x_val=x_val,
        y_val=y_val,
        y_val_raw=y_val_raw,
        genes_val=genes_val,
        x_test=x_test,
        y_test=y_test,
        y_test_raw=y_test_raw,
        genes_test=genes_test,
        y_max=y_max,
    )


def grid_search(
    split: CelltypeSplit,
    model_file: str,
    *,
    random_state: int = RANDOM_SEED,
) -> Tuple[object, Dict[str, float], np.ndarray, np.ndarray, float]:
    """Find the best celltype model by validation MSE."""
    model_path = Path(model_file)
    if model_path.is_file():
        LOGGER.info("Model exists, loading: %s", model_path)
        model, _, metadata = load_model_bundle(model_path)
        val_pred = model.predict(split.x_val)
        test_pred = model.predict(split.x_test)
        test_mse = mean_squared_error(split.y_test, test_pred)
        return (
            model,
            metadata.get("best_params", {}),
            val_pred,
            test_pred,
            float(test_mse),
        )

    param_grid = {
        "max_depth": [3, 4, 5],
        "learning_rate": [0.1, 0.3],
        "gamma": [0, 0.2, 0.4, 0.6, 0.8, 1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
    }

    best_model: Optional[object] = None
    best_params: Dict[str, float] = {}
    best_val_pred: Optional[np.ndarray] = None
    best_val_mse = float("inf")

    for learning_rate, max_depth, gamma, subsample in product(
        param_grid["learning_rate"],
        param_grid["max_depth"],
        param_grid["gamma"],
        param_grid["subsample"],
    ):
        model = make_xgb_regressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=500,
            gamma=gamma,
            subsample=subsample,
            random_state=random_state,
        )
        fit_with_optional_early_stopping(
            model,
            split.x_train,
            split.y_train,
            split.x_val,
            split.y_val,
            early_stopping_rounds=30,
        )
        val_pred = model.predict(split.x_val)
        val_mse = mean_squared_error(split.y_val, val_pred)

        if val_mse < best_val_mse:
            best_val_mse = float(val_mse)
            best_model = model
            best_val_pred = val_pred
            best_iteration = getattr(model, "best_iteration", None)
            best_params = {
                "n_estimators": (
                    int(best_iteration + 1) if best_iteration is not None else 500
                ),
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "gamma": gamma,
                "subsample": subsample,
                "validation_mse": best_val_mse,
            }

    if best_model is None or best_val_pred is None:
        raise RuntimeError("Grid search did not produce a model.")

    test_pred = best_model.predict(split.x_test)
    test_mse = float(mean_squared_error(split.y_test, test_pred))
    LOGGER.info("Best parameters: %s", best_params)
    LOGGER.info("Test MSE: %.6g", test_mse)
    return best_model, best_params, best_val_pred, test_pred, test_mse


def train_model(
    dataset: ChromeXDataset,
    *,
    model_file: str,
    random_state: int = RANDOM_SEED,
) -> Tuple[float, float, Dict[str, float]]:
    """Train/evaluate one celltype model and save a model bundle."""
    split = split_dataset(dataset, upsampling=True, random_state=random_state)
    model, best_params, val_pred_norm, test_pred_norm, test_mse = grid_search(
        split,
        model_file,
        random_state=random_state,
    )

    val_pred = inverse_normalize_expression(val_pred_norm, split.y_max)
    test_pred = inverse_normalize_expression(test_pred_norm, split.y_max)

    pearson_val = safe_pearsonr(val_pred, split.y_val_raw)
    pearson_test = safe_pearsonr(test_pred, split.y_test_raw)

    dump_model_bundle(
        model_file,
        model=model,
        y_max=split.y_max,
        metadata={
            "model_type": "celltype-level",
            "best_params": best_params,
            "test_mse_normalized": test_mse,
            "random_state": random_state,
        },
    )

    LOGGER.info("Validation Pearson r: %.6g", pearson_val)
    LOGGER.info("Test Pearson r: %.6g", pearson_test)
    return pearson_val, pearson_test, best_params


def predict_genes(
    *,
    model_cell: str,
    predict_cell: str,
    dataset: ChromeXDataset,
    model_file: str,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Predict gene expression for one celltype dataset."""
    split = split_dataset(dataset, upsampling=False, random_state=random_state)
    model, model_y_max, _ = load_model_bundle(model_file)

    # New model bundles store y_max. Old bare celltype models do not; for those
    # files we fall back to the prediction-data y_max so old files remain usable.
    y_max = model_y_max if model_y_max is not None else split.y_max
    if model_y_max is None:
        LOGGER.warning(
            "Model file does not contain Y_max. Falling back to prediction-data Y_max; "
            "retrain with this refactored script for reproducible scaling."
        )

    pred_raw = inverse_normalize_expression(model.predict(split.x_all), y_max)

    return pd.DataFrame(
        {
            "model": model_cell,
            "cell": predict_cell,
            "predict": pred_raw,
            "label": split.y_all_raw,
            "isTrain": split.is_train_like,
            "gene_name": split.genes_all,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or apply the celltype-level XGBoost model.",
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog,
            max_help_position=40,
            width=100,
        ),
    )
    parser.add_argument(
        "-c",
        "--modelcell",
        required=True,
        dest="model_cell",
        help="Cell used for training. [required]",
    )
    parser.add_argument(
        "-C", "--predictcell", dest="predict_cell", help="Cell used for prediction."
    )
    parser.add_argument(
        "-i", "--modelinput", dest="model_input", help="Input HDF5 file for training."
    )
    parser.add_argument(
        "-I",
        "--predictinput",
        dest="predict_input",
        help="Input HDF5 file for prediction.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        dest="output",
        help="Result directory. [required]",
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        dest="model",
        help="Model joblib path. [required]",
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
        "--seed", type=int, default=RANDOM_SEED, help="Random seed. Default: 42"
    )
    return parser


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()

    result_dir = ensure_dir(args.output)

    if args.process == "train":
        if not args.model_input:
            raise ValueError("Training requires -i/--modelinput.")
        dataset = load_dataset(
            require_file(args.model_input, "training HDF5 data file")
        )
        pearson_val, pearson_test, _ = train_model(
            dataset,
            model_file=args.model,
            random_state=args.seed,
        )
        out = pd.DataFrame(
            {
                "Cell": [args.model_cell],
                "Validationset_performance": [pearson_val],
                "Testset_performance": [pearson_test],
            }
        )
        write_table(out, result_dir / f"{args.model_cell}_model_performance.txt")

    elif args.process == "predict":
        if not args.predict_input:
            raise ValueError("Prediction requires -I/--predictinput.")
        if not args.predict_cell:
            raise ValueError("Prediction requires -C/--predictcell.")
        require_file(args.model, "model file")
        dataset = load_dataset(
            require_file(args.predict_input, "prediction HDF5 data file")
        )
        out = predict_genes(
            model_cell=args.model_cell,
            predict_cell=args.predict_cell,
            dataset=dataset,
            model_file=args.model,
            random_state=args.seed,
        )
        write_table(
            out,
            result_dir
            / f"{args.predict_cell}_gene_prediction_on_{args.model_cell}_model.txt",
        )


if __name__ == "__main__":
    main()

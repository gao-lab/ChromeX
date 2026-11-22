#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reusable modeling helpers for ChromeX XGBoost models."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np
import xgboost as xgb

from .common import RANDOM_SEED


def safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """Return Pearson correlation, or NaN when it is undefined."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if len(x_arr) < 2 or len(y_arr) < 2:
        return float("nan")
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def make_xgb_regressor(
    *,
    max_depth: int,
    learning_rate: float,
    n_estimators: int,
    gamma: float,
    subsample: float,
    random_state: int = RANDOM_SEED,
    extra_params: Optional[Mapping[str, object]] = None,
) -> xgb.XGBRegressor:
    """Create a consistently configured XGBRegressor."""
    params: Dict[str, object] = {
        "max_depth": int(max_depth),
        "learning_rate": float(learning_rate),
        "n_estimators": int(n_estimators),
        "random_state": random_state,
        "gamma": float(gamma),
        "subsample": float(subsample),
        "objective": "reg:squarederror",
        "n_jobs": -1,
    }
    if extra_params:
        params.update(dict(extra_params))
    return xgb.XGBRegressor(**params)


def fit_with_optional_early_stopping(
    model: xgb.XGBRegressor,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    early_stopping_rounds: int = 30,
) -> xgb.XGBRegressor:
    try:
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            verbose=False,
            early_stopping_rounds=early_stopping_rounds,
        )
    except TypeError:
        model.set_params(early_stopping_rounds=early_stopping_rounds)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    return model

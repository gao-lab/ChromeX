#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preprocessing utilities for ChromeX XGBoost models.

This module owns row filtering, feature construction, expression scaling and
training-set upsampling. It intentionally does not perform file I/O.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import List, Sequence, Tuple

import numpy as np
from sklearn.utils import resample

from .common import RANDOM_SEED

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChromeXDataset:
    """In-memory representation of one filtered ChromeX HDF5 dataset."""

    gene_order: List[str]
    y: np.ndarray
    gene_len: np.ndarray
    ab: np.ndarray
    genebody_ab: np.ndarray
    boundary_level: np.ndarray
    interaction_frequency: np.ndarray
    region_level: np.ndarray
    loop: np.ndarray

    def select(self, indices: Sequence[int]) -> "ChromeXDataset":
        """Return a row-subset dataset while preserving field names."""
        idx = np.asarray(indices, dtype=int)
        return replace(
            self,
            gene_order=[self.gene_order[i] for i in idx],
            y=self.y[idx],
            gene_len=self.gene_len[idx],
            ab=self.ab[idx],
            genebody_ab=self.genebody_ab[idx],
            boundary_level=self.boundary_level[idx],
            interaction_frequency=self.interaction_frequency[idx],
            region_level=self.region_level[idx],
            loop=self.loop[idx],
        )

    def feature_matrix(self) -> np.ndarray:
        """Build the model feature matrix using the canonical feature order."""
        return build_feature_matrix(
            gene_len=self.gene_len,
            ab=self.ab,
            genebody_ab=self.genebody_ab,
            boundary_level=self.boundary_level,
            interaction_frequency=self.interaction_frequency,
            region_level=self.region_level,
            loop=self.loop,
        )


def valid_ab_indices(ab_pre: np.ndarray, ab: np.ndarray) -> np.ndarray:
    """Return indices that pass the original AB/non-zero filtering rule.

    Rule inherited from the source scripts:
    - all columns 95:106 of X_AB_pre are non-zero;
    - column 20 of X_AB is non-zero.
    """
    if ab_pre.ndim != 2 or ab.ndim != 2:
        raise ValueError("X_AB_pre and X_AB must be 2-D arrays.")
    if ab_pre.shape[0] != ab.shape[0]:
        raise ValueError("X_AB_pre and X_AB must have the same row count.")
    if ab_pre.shape[1] < 106:
        raise ValueError("X_AB_pre must contain at least 106 columns.")
    if ab.shape[1] < 21:
        raise ValueError("X_AB must contain at least 21 columns.")

    idx_pre = np.argwhere(np.all(ab_pre[:, 95:106] != 0, axis=1)).ravel()
    idx_ab = np.argwhere(ab[:, 20] != 0).ravel()
    return np.intersect1d(idx_pre, idx_ab)


def build_feature_matrix(
    *,
    gene_len: np.ndarray,
    ab: np.ndarray,
    genebody_ab: np.ndarray,
    boundary_level: np.ndarray,
    interaction_frequency: np.ndarray,
    region_level: np.ndarray,
    loop: np.ndarray,
) -> np.ndarray:
    """Concatenate all model features into one 2-D matrix."""
    gene_len_2d = np.asarray(gene_len).reshape(-1, 1)
    matrices = [
        np.asarray(ab),
        np.asarray(genebody_ab),
        np.asarray(boundary_level),
        np.asarray(interaction_frequency),
        np.asarray(region_level),
        np.asarray(loop),
        gene_len_2d,
    ]

    row_counts = {matrix.shape[0] for matrix in matrices}
    if len(row_counts) != 1:
        raise ValueError(f"Feature arrays have inconsistent row counts: {row_counts}")

    return np.concatenate(matrices, axis=1)


def normalize_expression(y: np.ndarray) -> Tuple[np.ndarray, float]:
    """Apply log1p expression transform and divide by the train-set max."""
    y_clean = np.nan_to_num(np.asarray(y, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    y_log = np.log1p(y_clean)
    y_max = float(np.max(y_log))
    if not np.isfinite(y_max) or y_max <= 0:
        LOGGER.warning(
            "Expression max is non-positive; using y_max=1.0 to avoid division by zero."
        )
        y_max = 1.0
    return y_log / y_max, y_max


def normalize_expression_with_ymax(y: np.ndarray, y_max: float) -> np.ndarray:
    """Apply log1p expression transform using a previously learned y_max."""
    if y_max <= 0:
        raise ValueError(f"y_max must be positive, got {y_max}")
    y_clean = np.nan_to_num(np.asarray(y, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return np.log1p(y_clean) / float(y_max)


def inverse_normalize_expression(y_norm: np.ndarray, y_max: float) -> np.ndarray:
    """Invert normalize_expression/normalize_expression_with_ymax."""
    if y_max <= 0:
        raise ValueError(f"y_max must be positive, got {y_max}")
    return np.expm1(np.asarray(y_norm, dtype=float) * float(y_max))


def upsample_high_expression(
    x: np.ndarray,
    y: np.ndarray,
    *,
    threshold: float = 0.75,
    divisor: int = 4,
    max_factor: int = 5,
    random_state: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Upsample high-expression samples using the rule from the original scripts.

    Returns ``(x_resampled, y_resampled, factor)``. ``factor`` is 0 when no
    upsampling was performed.
    """
    high_idx = np.where(np.asarray(y) > threshold)[0]
    if len(high_idx) == 0:
        LOGGER.warning("No samples above threshold %.3f; skip upsampling.", threshold)
        return x, y, 0

    factor = min(int((len(y) / len(high_idx)) / divisor), max_factor)
    if factor <= 0:
        return x, y, 0

    x_high = x[high_idx]
    y_high = y[high_idx]
    x_extra, y_extra = resample(
        x_high,
        y_high,
        replace=True,
        n_samples=len(high_idx) * factor,
        random_state=random_state,
    )
    return (
        np.concatenate([x, x_extra], axis=0),
        np.concatenate([y, y_extra], axis=0),
        factor,
    )

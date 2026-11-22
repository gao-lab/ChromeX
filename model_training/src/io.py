#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""File I/O helpers for ChromeX XGBoost models.

Responsibilities:
- path validation
- HDF5 loading
- datalist reading
- model bundle loading/saving
- TSV/table output
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import h5py
import joblib
import numpy as np
import pandas as pd

from .common import FEATURE_ORDER, HDF5_KEYS
from .preprocessing import ChromeXDataset, valid_ab_indices

PathLike = Any
LOGGER = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a compact console logger if the application has not done so."""
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_dir(path: PathLike) -> Path:
    """Create a directory if needed and return it as a Path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def require_file(path: PathLike, description: str = "file") -> Path:
    """Validate that a path exists and is a regular file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{description} does not exist: {p}")
    return p


def require_dir(path: PathLike, description: str = "directory") -> Path:
    """Validate that a path exists and is a directory."""
    p = Path(path)
    if not p.is_dir():
        raise NotADirectoryError(f"{description} does not exist: {p}")
    return p


def decode_gene_names(raw_gene_order: np.ndarray) -> List[str]:
    """Decode HDF5 gene names into Python strings."""
    gene_names: List[str] = []
    for item in raw_gene_order:
        if isinstance(item, bytes):
            gene_names.append(item.decode("utf-8"))
        else:
            gene_names.append(str(item))
    return gene_names


def read_name_list(path: PathLike) -> List[str]:
    """Read dataset/cell names from a CSV-like file.

    This accepts both formats commonly seen in these scripts:
    one-row-many-columns and one-column-many-rows. Empty cells are ignored.
    """
    p = require_file(path, "dataset list")
    table = pd.read_csv(p, header=None)
    names: List[str] = []
    for value in table.values.ravel():
        if pd.isna(value):
            continue
        name = str(value).strip()
        if name:
            names.append(name)

    deduped = list(dict.fromkeys(names))
    if not deduped:
        raise ValueError(f"No dataset/cell names found in {p}")
    return deduped


def load_dataset(data_file: PathLike) -> ChromeXDataset:
    """Load and filter one ChromeX HDF5 dataset."""
    p = require_file(data_file, "HDF5 data file")

    with h5py.File(p, "r") as h5:
        missing = [key for key in HDF5_KEYS.values() if key not in h5]
        if missing:
            raise KeyError(f"{p} is missing required HDF5 keys: {missing}")

        gene_order_raw = h5[HDF5_KEYS["gene_order"]][:]
        y = h5[HDF5_KEYS["y"]][:]
        gene_len = h5[HDF5_KEYS["gene_len"]][:]
        ab_pre = h5[HDF5_KEYS["ab_pre"]][:]
        ab = h5[HDF5_KEYS["ab"]][:]
        genebody_ab = h5[HDF5_KEYS["genebody_ab"]][:]
        loop = h5[HDF5_KEYS["loop"]][:]
        boundary_level = h5[HDF5_KEYS["boundary_level"]][:]
        region_level = h5[HDF5_KEYS["region_level"]][:]
        interaction_frequency = h5[HDF5_KEYS["interaction_frequency"]][:]

    keep = valid_ab_indices(ab_pre, ab)
    if len(keep) == 0:
        raise ValueError(f"No valid rows remain after AB filtering: {p}")

    gene_order = decode_gene_names(gene_order_raw)
    return ChromeXDataset(
        gene_order=[gene_order[i] for i in keep],
        y=np.nan_to_num(y[keep], nan=0.0, posinf=0.0, neginf=0.0),
        gene_len=gene_len[keep],
        ab=ab[keep],
        genebody_ab=genebody_ab[keep],
        loop=loop[keep],
        boundary_level=boundary_level[keep],
        region_level=region_level[keep],
        interaction_frequency=interaction_frequency[keep],
    )


def dump_model_bundle(
    model_file: PathLike,
    *,
    model: Any,
    y_max: float,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    """Save a model with metadata needed for reproducible prediction."""
    p = Path(model_file)
    if p.parent and str(p.parent) != ".":
        p.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "model": model,
        "Y_max": float(y_max),
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    joblib.dump(payload, p)


def load_model_bundle(
    model_file: PathLike,
) -> Tuple[Any, Optional[float], Dict[str, Any]]:
    """Load either a new model bundle or an old bare XGBRegressor.

    Returns ``(model, y_max, metadata)``.
    """
    p = require_file(model_file, "model file")
    obj = joblib.load(p)

    if isinstance(obj, dict) and "model" in obj:
        y_max = obj.get("Y_max", None)
        metadata = dict(obj.get("metadata", {}))
        return obj["model"], None if y_max is None else float(y_max), metadata

    return obj, None, {}


def write_table(df: pd.DataFrame, output_file: PathLike) -> None:
    """Write a tab-separated table, creating parent directories."""
    p = Path(output_file)
    if p.parent and str(p.parent) != ".":
        p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, sep="\t", index=False)

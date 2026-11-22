"""ChromeX XGBoost model utilities."""

from .common import FEATURE_ORDER, HDF5_KEYS, RANDOM_SEED
from .io import (
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
from .modeling import (
    fit_with_optional_early_stopping,
    make_xgb_regressor,
    safe_pearsonr,
)
from .preprocessing import (
    ChromeXDataset,
    build_feature_matrix,
    inverse_normalize_expression,
    normalize_expression,
    normalize_expression_with_ymax,
    upsample_high_expression,
    valid_ab_indices,
)

__all__ = [
    "ChromeXDataset",
    "FEATURE_ORDER",
    "HDF5_KEYS",
    "RANDOM_SEED",
    "build_feature_matrix",
    "configure_logging",
    "dump_model_bundle",
    "ensure_dir",
    "fit_with_optional_early_stopping",
    "inverse_normalize_expression",
    "load_dataset",
    "load_model_bundle",
    "make_xgb_regressor",
    "normalize_expression",
    "normalize_expression_with_ymax",
    "read_name_list",
    "require_dir",
    "require_file",
    "safe_pearsonr",
    "upsample_high_expression",
    "valid_ab_indices",
    "write_table",
]

__version__ = "0.2.0"

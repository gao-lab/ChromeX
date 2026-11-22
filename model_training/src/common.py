#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small shared constants for ChromeX XGBoost models."""

from __future__ import annotations

RANDOM_SEED = 42

FEATURE_ORDER = (
    "X_AB",
    "X_genebodyAB",
    "X_boundary_level",
    "X_interaction_frequency",
    "X_region_level",
    "X_loop",
    "X_gene_len",
)

HDF5_KEYS = {
    "gene_order": "gene_order",
    "y": "Y",
    "gene_len": "X_gene_len",
    "ab_pre": "X_AB_pre",
    "ab": "X_AB",
    "genebody_ab": "X_genebodyAB",
    "loop": "X_loop",
    "boundary_level": "X_boundary_level",
    "region_level": "X_region_level",
    "interaction_frequency": "X_interaction_frequency",
}

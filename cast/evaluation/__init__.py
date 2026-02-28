# -*- coding: utf-8 -*-
"""
Evaluation module for CAST-ACF.

Implements the ACF (Alignment/Coverage/Factuality) evaluation framework
that addresses the overestimation issue in DTELS-style metrics.
"""
from .acf import (
    # Configuration
    DEFAULT_CONFIG,
    # Text processing
    normalize_text,
    tokenize_zh,
    date_ordinal,
    # Entailment
    entail_rouge_recall,
    entail_batch,
    # Time kernel
    compute_tau,
    gaussian_kernel,
    normalize_score,
    # Atoms
    extract_atoms,
    join_atoms,
    # Metrics
    compute_alignment,
    compute_coverage,
    compute_factuality,
    evaluate_acf,
)

__all__ = [
    # config
    "DEFAULT_CONFIG",
    # text
    "normalize_text",
    "tokenize_zh", 
    "date_ordinal",
    # entailment
    "entail_rouge_recall",
    "entail_batch",
    # time
    "compute_tau",
    "gaussian_kernel",
    "normalize_score",
    # atoms
    "extract_atoms",
    "join_atoms",
    # metrics
    "compute_alignment",
    "compute_coverage",
    "compute_factuality",
    "evaluate_acf",
]

# -*- coding: utf-8 -*-
"""
CAST-ACF: Robust Generation and Evaluation for Multi-Granularity Timeline Summarization

This package provides:
- CAST: Chronology-windowed Abstractive Summarization for multi-granularity Timelines
- ACF: Alignment/Coverage/Factuality evaluation metrics

Reference:
    Ai, Y. & Kong, F. (2026). CAST-ACF: Robust Generation and Evaluation for 
    Multi-Granularity Timeline Summarization. ICASSP 2026.
"""

__version__ = "1.0.0"
__author__ = "Yuming Ai"

from . import utils
from . import data
from . import models
from . import evaluation

__all__ = [
    "utils",
    "data", 
    "models",
    "evaluation",
]

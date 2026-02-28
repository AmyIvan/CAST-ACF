# -*- coding: utf-8 -*-
"""
Model components for CAST-ACF.

Includes:
- Evidence Selector: Binary classifier for selecting relevant articles
- Summarizer utilities: LoRA fine-tuning helpers
"""
from .selector_feats import (
    build_feats_for_window,
    window_embeddings,
    SHELL_HINTS,
    VERB_HINTS,
)
from .selector import (
    SelectorConfig,
    train_selector,
    load_selector,
    predict_selector,
)

__all__ = [
    # selector features
    "build_feats_for_window",
    "window_embeddings",
    "SHELL_HINTS",
    "VERB_HINTS",
    # selector
    "SelectorConfig",
    "train_selector",
    "load_selector",
    "predict_selector",
]

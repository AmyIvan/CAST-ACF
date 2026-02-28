# -*- coding: utf-8 -*-
"""
Data processing module for CAST-ACF.

Includes evidence pair mining, prompt construction, and dataset utilities.
"""
from .mining import (
    MiningConfig,
    extract_gold_timelines,
    normalize_gold_list,
    clean_candidate,
    compute_scores,
    mine_pairs_for_topic,
)
from .prompts import (
    PromptConfig,
    build_evidence_items,
    assemble_prompt,
    apply_chat_template,
    render_training_example,
)
from .dataset import (
    PairsDataset,
    collate_fn,
)

__all__ = [
    # mining
    "MiningConfig",
    "extract_gold_timelines",
    "normalize_gold_list",
    "clean_candidate",
    "compute_scores",
    "mine_pairs_for_topic",
    # prompts
    "PromptConfig",
    "build_evidence_items",
    "assemble_prompt",
    "apply_chat_template",
    "render_training_example",
    # dataset
    "PairsDataset",
    "collate_fn",
]

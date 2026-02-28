# -*- coding: utf-8 -*-
"""
Utility functions for CAST-ACF.
"""
from .io import (
    read_jsonl,
    read_json_file,
    iter_jsonl,
    write_jsonl,
    index_cand_tar,
    stream_cand_jsonl_from_tar,
    TarJsonlIndex,
)
from .timebin import (
    parse_time_any,
    build_windows,
    in_window,
    daydiff,
    same_day,
)
from .text import (
    zh_tokenize,
    safe_join,
    clean_for_eval,
)
from .embedding import (
    EmbeddingEncoder,
    cosine_sim,
)
from .bm25 import BM25Scorer
from .aligner import min_redundancy_pick

__all__ = [
    # io
    "read_jsonl",
    "read_json_file", 
    "iter_jsonl",
    "write_jsonl",
    "index_cand_tar",
    "stream_cand_jsonl_from_tar",
    "TarJsonlIndex",
    # timebin
    "parse_time_any",
    "build_windows",
    "in_window",
    "daydiff",
    "same_day",
    # text
    "zh_tokenize",
    "safe_join",
    "clean_for_eval",
    # embedding
    "EmbeddingEncoder",
    "cosine_sim",
    # bm25
    "BM25Scorer",
    # aligner
    "min_redundancy_pick",
]

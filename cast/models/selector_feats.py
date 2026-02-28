# -*- coding: utf-8 -*-
"""
Feature extraction for Evidence Selector.

Implements the 9-dimensional feature vector for each candidate article,
used to train a binary classifier for evidence selection.
"""
from __future__ import annotations
import re
from typing import List, Dict, Any, Tuple
from collections import Counter
import numpy as np

from ..utils.embedding import EmbeddingEncoder


# Heuristic vocabulary for feature extraction
SHELL_HINTS = ["快讯", "直播", "视频", "图集", "漫画", "整点新闻", "VLOG", "专访"]
VERB_HINTS = [
    "宣布", "通过", "批准", "发布", "上涨", "下跌", "确诊", "遇难", 
    "开庭", "判决", "登陆", "签署", "中标", "上市", "下架", "撤离", 
    "停火", "约谈", "处罚", "起火", "地震", "降雨", "暴雨", "洪水", 
    "失联", "复航", "复工", "投产", "提价", "降价"
]

_num_pat = re.compile(r"\d+([.,]\d+)?")


def _has_number(s: str) -> int:
    """Check if text contains a number."""
    return 1 if _num_pat.search(s or "") else 0


def _contains_any(s: str, vocab: List[str]) -> int:
    """Check if text contains any word from vocabulary."""
    s = s or ""
    return 1 if any(v in s for v in vocab) else 0


def _ensure_2d(a: np.ndarray) -> np.ndarray:
    """Ensure array is 2D."""
    return a.reshape(-1, 1) if a.ndim == 1 else a


def _row_topk(mat: np.ndarray, k: int) -> np.ndarray:
    """
    Get top-k values for each row of a matrix.
    
    Returns empty array if k <= 0 or matrix has no columns.
    """
    mat = _ensure_2d(mat)
    n_cols = mat.shape[1]
    
    if n_cols == 0 or k <= 0:
        return np.empty((mat.shape[0], 0), dtype=mat.dtype)
    
    k = min(k, n_cols)
    kth = n_cols - k
    
    if kth <= 0:
        return np.sort(mat, axis=1)[:, -k:]
    
    idx = np.argpartition(mat, kth, axis=1)[:, -k:]
    return np.take_along_axis(mat, idx, axis=1)


def window_embeddings(
    encoder: EmbeddingEncoder, 
    titles: List[str], 
    texts: List[str]
) -> np.ndarray:
    """
    Compute embeddings for candidates in a window.
    
    Concatenates title and content for each candidate.
    """
    texts_for_emb = [
        ((titles[i] or "") + " " + (texts[i] or "")) 
        for i in range(len(titles))
    ]
    return encoder.encode(texts_for_emb, batch_size=64, normalize=True)


def build_feats_for_window(
    encoder: EmbeddingEncoder,
    titles: List[str],
    texts: List[str],
    ordinals: List[int],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build feature vectors for candidates in a chronological window.
    
    Features (9-dimensional):
        1. sims: Cosine similarity to window centroid
        2. dens: Local density (average of top-3 similarities)
        3. same_day: Count of articles on the same day
        4. day_off: Absolute day offset from window median
        5. title_len: Title length (characters)
        6. content_len: Content length (characters)
        7. has_number: Binary - contains numeric content
        8. has_shell: Binary - contains shallow content indicators
        9. has_verb_hint: Binary - contains event verb indicators
    
    Args:
        encoder: Embedding encoder
        titles: List of article titles
        texts: List of article contents
        ordinals: List of day ordinals (timestamps)
        
    Returns:
        X: Feature matrix of shape (M, 9)
        stats: Dict containing intermediate values (embeddings, etc.)
    """
    M = len(titles)
    
    # Compute embeddings
    emb = window_embeddings(encoder, titles, texts)  # [M, H]
    
    # Window centroid
    centroid = emb.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
    
    # Feature 1: Similarity to centroid
    sims = emb @ centroid  # [M]
    
    # Feature 2: Local density (top-3 similarity mean)
    mat = emb @ emb.T  # [M, M]
    np.fill_diagonal(mat, -1.0)  # Exclude self
    mat = _ensure_2d(mat)
    
    topk_vals = _row_topk(mat, k=3)
    if topk_vals.shape[1] > 0:
        dens = topk_vals.mean(axis=1)
        top1_max = topk_vals.max(axis=1)
    else:
        dens = np.zeros((mat.shape[0],), dtype=mat.dtype)
        top1_max = np.zeros((mat.shape[0],), dtype=mat.dtype)
    
    # Feature 3: Same day count
    cnt = Counter(ordinals)
    same_day = np.array([cnt[o] for o in ordinals], dtype=float)
    
    # Feature 4: Day offset from median
    mid = int(np.median(ordinals)) if len(ordinals) else 0
    day_off = np.abs(
        np.array(ordinals, dtype=int) - mid
    ).astype(float) if len(ordinals) else np.zeros((M,), float)
    
    # Feature 5-6: Length features
    title_len = np.array([len(t or "") for t in titles], dtype=float)
    content_len = np.array([len(x or "") for x in texts], dtype=float)
    
    # Feature 7-9: Heuristic features
    has_num = np.array([
        _has_number((titles[i] or "") + " " + (texts[i] or "")) 
        for i in range(M)
    ], dtype=float)
    
    has_shell = np.array([
        _contains_any(titles[i] or "", SHELL_HINTS) 
        for i in range(M)
    ], dtype=float)
    
    has_verb = np.array([
        _contains_any(titles[i] or "", VERB_HINTS) 
        for i in range(M)
    ], dtype=float)
    
    # Stack features
    X = np.stack([
        sims, dens, same_day, day_off, 
        title_len, content_len, 
        has_num, has_shell, has_verb
    ], axis=1)
    
    stats = {
        "emb": emb,
        "centroid": centroid,
        "sim_centroid": sims,
        "density": dens,
        "top1": top1_max,
    }
    
    return X, stats

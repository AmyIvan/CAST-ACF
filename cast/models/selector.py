# -*- coding: utf-8 -*-
"""
Evidence Selector training module.

Implements the binary classifier for evidence selection:
    p_i = g_ψ(X_i) = σ(w^T X_i + b)
    
Trained with cross-entropy loss (Equation 2 in paper).
"""
from __future__ import annotations
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import random

import numpy as np
import joblib
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from ..utils.io import index_cand_tar, stream_cand_jsonl_from_tar
from ..utils.timebin import parse_time_any
from ..utils.embedding import EmbeddingEncoder
from .selector_feats import build_feats_for_window


@dataclass
class SelectorConfig:
    """Configuration for selector training."""
    embed_model: str = "moka-ai/m3e-large"
    neg_downsample: float = 0.5   # Negative sample ratio
    topk_near: int = 80           # Max candidates per window
    days_radius: Optional[float] = None  # Optional time radius filter
    seed: int = 42


def clean_candidate(c: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Clean and validate a candidate article."""
    t = parse_time_any(c.get("time"))
    if t is None:
        return None
    return {
        "time": t,
        "title": (c.get("title") or "").strip(),
        "content": (c.get("content") or c.get("abstract") or c.get("desc") or "").strip(),
    }


def pick_pool_indices(
    times_ord: np.ndarray, 
    gold_ord: int, 
    topk_near: int, 
    days_radius: Optional[float]
) -> np.ndarray:
    """
    Select candidate pool based on time proximity.
    
    Args:
        times_ord: Ordinal timestamps of all candidates
        gold_ord: Ordinal timestamp of gold node
        topk_near: Maximum candidates to select
        days_radius: Optional radius filter (days)
        
    Returns:
        Array of selected candidate indices
    """
    dt = np.abs(times_ord - gold_ord)
    
    if days_radius is not None and days_radius > 0:
        mask = dt <= float(days_radius)
        idx = np.where(mask)[0]
        
        if idx.size == 0:
            # Fallback: take nearest regardless of radius
            order = np.argsort(dt)
            idx = order[:topk_near]
        else:
            # Filter by radius, then take nearest
            idx = idx[np.argsort(dt[idx])][:topk_near]
    else:
        order = np.argsort(dt)
        idx = order[:topk_near]
    
    return idx


def train_selector(
    cand_tar_path: str,
    pairs_paths: List[str],
    K: str,
    config: SelectorConfig,
    output_path: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train the evidence selector.
    
    Args:
        cand_tar_path: Path to candidate tar archive
        pairs_paths: Paths to pairs JSONL files
        K: Granularity ("N", "10", "5")
        config: Training configuration
        output_path: Path to save trained model
        verbose: Print progress
        
    Returns:
        Training statistics
    """
    # Set random seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    # Load pairs records
    try:
        import ujson
    except ImportError:
        import json as ujson
    
    pairs_rows: List[Dict[str, Any]] = []
    for p in pairs_paths:
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = ujson.loads(line)
                if str(r.get("K")) != str(K):
                    continue
                pairs_rows.append(r)
    
    if not pairs_rows:
        raise RuntimeError(f"No pairs found for K={K} in {pairs_paths}")
    
    # Index candidates
    tar_idx = index_cand_tar(cand_tar_path)
    
    # Initialize encoder
    encoder = EmbeddingEncoder(config.embed_model)
    
    # Group by topic
    from collections import defaultdict
    rows_by_topic: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in pairs_rows:
        tid = str(r.get("topic_id"))
        rows_by_topic[tid].append(r)
    
    # Collect training data
    X_all, y_all = [], []
    stats = {"windows": 0, "pos": 0, "neg": 0}
    
    topic_iter = tqdm(rows_by_topic.items(), desc=f"Training selector [{K}]") if verbose else rows_by_topic.items()
    
    for tid, rows in topic_iter:
        # Load candidates for this topic
        cands_raw = list(stream_cand_jsonl_from_tar(tar_idx, tid))
        cands = [z for z in (clean_candidate(c) for c in cands_raw) if z is not None]
        
        if not cands:
            continue
        
        titles = [c["title"] for c in cands]
        texts = [c["content"] for c in cands]
        ords = np.array([c["time"].toordinal() for c in cands], dtype=np.int64)
        
        for r in rows:
            node_id = int(r.get("node_id"))
            gold_time = r.get("gold_time")
            pos_ids: List[int] = list(map(int, r.get("cand_ids", [])))
            
            if len(pos_ids) == 0:
                continue
            
            gdt = parse_time_any(gold_time)
            if gdt is None:
                continue
            g_ord = gdt.toordinal()
            
            # Check bounds
            n_all = len(cands)
            bad = [i for i in pos_ids if i < 0 or i >= n_all]
            if bad:
                continue
            
            # Select candidate pool
            pool_idx = pick_pool_indices(
                ords, g_ord, 
                topk_near=config.topk_near, 
                days_radius=config.days_radius
            )
            pool_set = set(pool_idx.tolist()) | set(pos_ids)
            pool_idx = np.array(sorted(pool_set), dtype=np.int64)
            
            # Build features
            titles_in = [titles[i] for i in pool_idx]
            texts_in = [texts[i] for i in pool_idx]
            ord_in = [int(ords[i]) for i in pool_idx]
            
            X_win, _ = build_feats_for_window(encoder, titles_in, texts_in, ord_in)
            y_win = np.array([
                1 if int(pool_idx[i]) in pos_ids else 0 
                for i in range(len(pool_idx))
            ], dtype=np.int64)
            
            # Negative downsampling
            if config.neg_downsample < 1.0:
                pos_i = np.where(y_win == 1)[0]
                neg_i = np.where(y_win == 0)[0]
                
                if neg_i.size > 0:
                    keep_mask = np.random.rand(neg_i.size) < float(config.neg_downsample)
                    neg_i = neg_i[keep_mask]
                
                keep_idx = np.concatenate([pos_i, neg_i], axis=0)
                
                if keep_idx.size == 0:
                    continue
                
                X_win = X_win[keep_idx]
                y_win = y_win[keep_idx]
            
            X_all.append(X_win)
            y_all.append(y_win)
            
            stats["windows"] += 1
            stats["pos"] += int(y_win.sum())
            stats["neg"] += int((y_win == 0).sum())
    
    if not X_all:
        raise RuntimeError("No training samples collected")
    
    X_all = np.vstack(X_all)
    y_all = np.hstack(y_all)
    
    # Check class balance
    uniq, counts = np.unique(y_all, return_counts=True)
    if len(uniq) < 2:
        raise ValueError("Only one class found - check pairs data")
    
    if verbose:
        print(f"[Selector] samples={len(y_all)}, pos_rate={y_all.mean():.3f}")
    
    # Train classifier
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    
    clf = LogisticRegression(
        max_iter=200,
        class_weight="balanced",
        n_jobs=1,
        solver="lbfgs",
    )
    clf.fit(X_scaled, y_all)
    
    # Save model
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    joblib.dump({"scaler": scaler, "clf": clf}, output_path)
    
    if verbose:
        print(f"[Selector] saved to {output_path}")
    
    stats["X_shape"] = X_all.shape
    return stats


def load_selector(path: str) -> Dict[str, Any]:
    """Load a trained selector model."""
    return joblib.load(path)


def predict_selector(
    selector: Dict[str, Any],
    encoder: EmbeddingEncoder,
    titles: List[str],
    texts: List[str],
    ordinals: List[int],
) -> np.ndarray:
    """
    Predict evidence scores for candidates.
    
    Args:
        selector: Loaded selector model (dict with 'scaler' and 'clf')
        encoder: Embedding encoder
        titles: Candidate titles
        texts: Candidate contents
        ordinals: Day ordinals
        
    Returns:
        Probability scores for each candidate
    """
    X, _ = build_feats_for_window(encoder, titles, texts, ordinals)
    
    scaler = selector["scaler"]
    clf = selector["clf"]
    
    X_scaled = scaler.transform(X)
    probs = clf.predict_proba(X_scaled)[:, 1]
    
    return probs

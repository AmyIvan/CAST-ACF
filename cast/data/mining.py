# -*- coding: utf-8 -*-
"""
Evidence Pair Mining for CAST training.

This module implements the scoring function from Equation (1) in the paper:
    S(i,j) = α·cos(f(a_i+b_i), f(y_j)) + β·BM25(a_i+b_i, y_j) 
             - γ·|ord(t_i) - ord(τ_j)| + δ·1{date(t_i) = τ_j}

Where:
    - a_i, b_i: title and body of article i
    - y_j: gold summary for node j
    - f(·): sentence encoder (e.g., m3e-large)
    - τ_j: timestamp of gold node j
"""
from __future__ import annotations
import os
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np

from ..utils.io import (
    index_cand_tar, 
    stream_cand_jsonl_from_tar, 
    iter_jsonl, 
    read_json_file
)
from ..utils.timebin import parse_time_any, build_windows, in_window
from ..utils.text import safe_join
from ..utils.embedding import EmbeddingEncoder
from ..utils.bm25 import BM25Scorer
from ..utils.aligner import min_redundancy_pick


@dataclass
class MiningConfig:
    """Configuration for evidence pair mining."""
    # Scoring weights (Equation 1)
    alpha: float = 1.0    # Semantic similarity weight
    beta: float = 0.5     # BM25 weight
    gamma: float = 0.05   # Time penalty weight
    delta: float = 0.5    # Same-day bonus weight
    
    # Evidence selection
    L_N: int = 6          # Max evidence per node for N-granularity
    L_10: int = 4         # Max evidence per node for 10-granularity
    L_5: int = 3          # Max evidence per node for 5-granularity
    
    # Deduplication
    dedup_thr: float = 0.85  # MMR similarity threshold
    
    # Models
    embed_model: str = "moka-ai/m3e-large"
    
    # BM25 options
    bm25_use_title: bool = True
    bm25_use_content: bool = True
    
    # Output
    max_evidence_chars: int = 2000


def extract_gold_timelines(rec: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract timelines at three granularities from a reference record.
    
    Handles various annotation formats:
    - timeline_N, timeline_10, timeline_5 keys
    - meta_timeline dict with N/10/5 keys
    - timelines dict with N/10/5 keys
    - Top-level timeline (mapped to N)
    
    Returns:
        Dict with keys "N", "10", "5", each mapping to list of {time, summary}
    """
    out = {"N": [], "10": [], "5": []}

    def _push(arr, k):
        if isinstance(arr, list):
            for x in arr:
                t = x.get("time")
                s = x.get("summary") or x.get("title") or ""
                out[k].append({"time": t, "summary": s})

    tld = rec.get("timelines") if isinstance(rec.get("timelines"), dict) else None
    mtl = rec.get("meta_timeline") if isinstance(rec.get("meta_timeline"), dict) else None

    # N-granularity
    for key in ["timeline_N", "timeline_n", "timelineN"]:
        if isinstance(rec.get(key), list):
            _push(rec[key], "N")
            break
    
    if not out["N"] and mtl:
        node = mtl.get("N")
        if isinstance(node, dict) and isinstance(node.get("timeline"), list):
            _push(node["timeline"], "N")
        elif isinstance(node, list):
            _push(node, "N")
    
    if not out["N"] and tld and isinstance(tld.get("N"), list):
        _push(tld["N"], "N")
    
    if not out["N"]:
        tl = rec.get("timeline")
        if isinstance(tl, list) and tl:
            _push(tl, "N")

    # 10-granularity
    if mtl:
        node = mtl.get("10")
        if isinstance(node, dict) and isinstance(node.get("timeline"), list):
            _push(node["timeline"], "10")
        elif isinstance(node, list):
            _push(node, "10")
    
    if not out["10"] and tld and isinstance(tld.get("10"), list):
        _push(tld["10"], "10")
    
    if not out["10"]:
        for key in ["timeline_10", "timeline10", "tl_10", "tl10"]:
            if isinstance(rec.get(key), list):
                _push(rec[key], "10")
                break

    # 5-granularity
    if mtl:
        node = mtl.get("5")
        if isinstance(node, dict) and isinstance(node.get("timeline"), list):
            _push(node["timeline"], "5")
        elif isinstance(node, list):
            _push(node, "5")
    
    if not out["5"] and tld and isinstance(tld.get("5"), list):
        _push(tld["5"], "5")
    
    if not out["5"]:
        for key in ["timeline_5", "timeline5", "tl_5", "tl5"]:
            if isinstance(rec.get(key), list):
                _push(rec[key], "5")
                break

    return out


def normalize_gold_list(lst: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize and sort gold timeline nodes by time."""
    buf = []
    for x in lst:
        dt = parse_time_any(x.get("time"))
        if dt is None:
            continue
        s = (x.get("summary") or "").strip()
        if not s:
            continue
        buf.append({"time": dt, "summary": s})
    buf.sort(key=lambda z: z["time"])
    return buf


def clean_candidate(c: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Clean and validate a candidate article."""
    t = parse_time_any(c.get("time"))
    if t is None:
        return None
    title = (c.get("title") or "").strip()
    content = (c.get("content") or c.get("abstract") or c.get("desc") or "").strip()
    return {"time": t, "title": title, "content": content, "raw": c}


def compute_scores(
    gold_summ: str,
    gold_time,
    cand_times: List,
    emb_cand: np.ndarray,
    bm25_scores: List[float],
    emb_encoder: EmbeddingEncoder,
    config: MiningConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute scoring function S(i,j) from Equation (1).
    
    Returns:
        total_scores: Combined scores for each candidate
        sem_scores: Semantic similarity scores
    """
    # Gold embedding
    emb_gold = emb_encoder.encode([gold_summ], batch_size=8, normalize=True)[0]
    sem = emb_cand @ emb_gold  # Cosine similarity (embeddings are normalized)
    
    # BM25 scores (already computed)
    bm25 = np.array(bm25_scores, dtype=float)
    
    # Time penalty
    dt_days = np.array([
        abs(int((ct - gold_time).total_seconds() // 86400)) 
        for ct in cand_times
    ], dtype=float)
    
    # Same-day bonus
    sameday = np.array([
        1.0 if (ct.year, ct.month, ct.day) == (gold_time.year, gold_time.month, gold_time.day) 
        else 0.0
        for ct in cand_times
    ], dtype=float)
    
    # Combined score (Equation 1)
    total = (
        config.alpha * sem 
        + config.beta * bm25 
        - config.gamma * dt_days 
        + config.delta * sameday
    )
    
    return total, sem


def mine_pairs_for_topic(
    topic_id: str,
    gold_timelines: Dict[str, List[Dict[str, Any]]],
    candidates: List[Dict[str, Any]],
    emb_encoder: EmbeddingEncoder,
    config: MiningConfig,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Mine evidence pairs for all granularities of a single topic.
    
    Returns:
        Dict mapping granularity ("N", "10", "5") to list of pair records
    """
    if not candidates:
        return {"N": [], "10": [], "5": []}
    
    # Prepare candidate data
    titles = [c["title"] for c in candidates]
    texts = [c["content"] for c in candidates]
    times = [c["time"] for c in candidates]
    
    # Build BM25 corpus
    corpus_texts = []
    for i in range(len(candidates)):
        fields = []
        if config.bm25_use_title:
            fields.append(titles[i])
        if config.bm25_use_content:
            fields.append(texts[i])
        corpus_texts.append(" ".join([f for f in fields if f]))
    bm25 = BM25Scorer(corpus_texts)
    
    # Precompute embeddings for all candidates
    texts_for_sem = [safe_join([titles[i], texts[i]]) for i in range(len(candidates))]
    emb_all_sem = emb_encoder.encode(texts_for_sem, batch_size=64, normalize=True)
    
    texts_for_red = [" ".join([titles[i], texts[i]]) for i in range(len(candidates))]
    emb_all_red = emb_encoder.encode(texts_for_red, batch_size=64, normalize=True)
    
    results = {"N": [], "10": [], "5": []}
    
    for K, gold_list in [("N", gold_timelines.get("N", [])), 
                         ("10", gold_timelines.get("10", [])),
                         ("5", gold_timelines.get("5", []))]:
        if not gold_list:
            continue
        
        L_keep = {"N": config.L_N, "10": config.L_10, "5": config.L_5}[K]
        
        # Build time windows
        sorted_times = [g["time"] for g in gold_list]
        windows = build_windows(sorted_times)
        
        for node_id, (gold_node, window) in enumerate(zip(gold_list, windows)):
            gsum = gold_node["summary"]
            gtime = gold_node["time"]
            
            # BM25 scores for this gold summary
            bm25_full = bm25.scores(gsum)
            
            # Find candidates in this window
            idx_in = [i for i, ct in enumerate(times) if in_window(ct, window)]
            
            if len(idx_in) == 0:
                # Fallback: use all candidates, pick top-1
                totals, _ = compute_scores(
                    gsum, gtime, times, emb_all_sem, bm25_full, 
                    emb_encoder, config
                )
                best = int(np.argmax(totals))
                chosen = [best]
            else:
                # Score candidates in window
                emb_sub = emb_all_sem[idx_in]
                bm25_sub = [bm25_full[i] for i in idx_in]
                times_sub = [times[i] for i in idx_in]
                
                totals, _ = compute_scores(
                    gsum, gtime, times_sub, emb_sub, bm25_sub,
                    emb_encoder, config
                )
                
                # Rank and select with MMR
                order = np.argsort(-totals).tolist()
                emb_red_sub = emb_all_red[idx_in]
                picked_local = min_redundancy_pick(
                    order, emb_red_sub, topk=L_keep, sim_thr=config.dedup_thr
                )
                chosen = [idx_in[i] for i in picked_local]
            
            # Build output record
            cand_pack = []
            for i in chosen:
                content_clip = candidates[i]["content"]
                if config.max_evidence_chars > 0:
                    content_clip = content_clip[:config.max_evidence_chars]
                cand_pack.append({
                    "title": candidates[i]["title"],
                    "content": content_clip,
                    "time": candidates[i]["time"].isoformat(),
                })
            
            rec = {
                "topic_id": topic_id,
                "K": K,
                "node_id": node_id,
                "gold_time": gtime.isoformat(),
                "gold_summary": gsum,
                "cand_ids": chosen,
                "cand_pack": cand_pack,
            }
            results[K].append(rec)
    
    return results

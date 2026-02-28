# -*- coding: utf-8 -*-
"""
Alignment and redundancy removal utilities.
Implements Maximal Marginal Relevance (MMR) for diverse selection.
"""
from __future__ import annotations
from typing import List
import numpy as np

from .embedding import cosine_sim


def min_redundancy_pick(
    cand_idx: List[int], 
    emb: np.ndarray, 
    topk: int, 
    sim_thr: float
) -> List[int]:
    """
    Select up to topk candidates while avoiding redundancy (MMR-style).
    
    Implements Maximal Marginal Relevance selection:
    - Candidates are selected in order of their ranking (cand_idx)
    - A candidate is added only if its similarity to all previously
      selected candidates is below sim_thr
    
    Args:
        cand_idx: List of candidate indices, ordered by relevance (best first)
        emb: Embedding matrix of shape (n_candidates, dim)
        topk: Maximum number of candidates to select
        sim_thr: Similarity threshold for redundancy filtering
        
    Returns:
        List of selected candidate indices
        
    Reference:
        Carbonell & Goldstein (1998). "The Use of MMR, Diversity-Based 
        Reranking for Reordering Documents and Producing Summaries"
    """
    chosen: List[int] = []
    
    for i in cand_idx:
        if len(chosen) >= topk:
            break
        
        # Check redundancy against all previously selected
        is_redundant = False
        for j in chosen:
            s = cosine_sim(emb[i], emb[j])
            if s >= sim_thr:
                is_redundant = True
                break
        
        if not is_redundant:
            chosen.append(i)
    
    return chosen

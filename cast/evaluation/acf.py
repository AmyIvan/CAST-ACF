# -*- coding: utf-8 -*-
"""
ACF (Alignment/Coverage/Factuality) Evaluation Metrics.

This module implements the three-dimensional evaluation framework from the paper:
- Alignment (A): Overall quality considering semantic similarity and temporal proximity
- Coverage (C): Fraction of reference events that are adequately covered
- Factuality (F): Faithfulness of generated summaries to reference

Key improvement over DTELS-style evaluation:
- Removes Hungarian algorithm bias ("main diagonal bright band" effect)
- Uses entailment-based scoring with Gaussian time decay
"""
from __future__ import annotations
import re
import math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

# Optional dependencies
try:
    from rouge_chinese import Rouge
    _HAS_ROUGE = True
except ImportError:
    _HAS_ROUGE = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

try:
    import jieba
    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False


# ==================== Configuration ====================

# Default hyperparameters (from paper Section 5.1)
DEFAULT_CONFIG = {
    "alpha_tau": 0.1,           # τ = α_τ × span
    "tau_min": 4.0,
    "tau_max": 21.0,
    "clip_lo": 0.15,            # [L_φ, H_φ] for normalization
    "clip_hi": 0.85,
    "topk_frac": 0.6,           # α_topk for top-k averaging
    "theta": 0.50,              # Coverage threshold θ
    "theta_c": 0.38,            # Content threshold θ_c
    "theta_t": 0.60,            # Time threshold θ_t
}


# ==================== Text Processing ====================

_ALIAS = [
    ("川普", "特朗普"), ("特朗普总统", "特朗普"),
    ("Facebook母公司", "Meta"), ("Facebook 母公司", "Meta"),
]
_FILLER = [
    "感谢大家", "感谢支持", "谢谢大家", "热搜", "网友", 
    "据称", "传言", "有消息称", "媒体报道", "报道称",
    "目前", "近日", "随后", "当日", "今日", "昨日", "刚刚",
]
_REPORTING_VERBS = ["表示", "称", "回应", "发文", "指出", "透露", "认为", "强调"]


def normalize_text(s: str) -> str:
    """Light text normalization for evaluation."""
    if not s:
        return ""
    s = str(s).strip()
    
    # Alias normalization
    for a, b in _ALIAS:
        s = s.replace(a, b)
    
    # Remove filler words
    for w in _FILLER:
        s = s.replace(w, "")
    for w in _REPORTING_VERBS:
        s = re.sub(rf"{w}(?:称|道)?", "", s)
    
    # Normalize whitespace
    s = re.sub(r"[，,\s]{2,}", "，", s).strip("，, ")
    
    return s


def tokenize_zh(s: str) -> List[str]:
    """Tokenize Chinese text."""
    s = (s or "").strip()
    if not s:
        return []
    if _HAS_JIEBA:
        return list(jieba.cut(s))
    return re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", s)


def date_ordinal(date_str: str) -> int:
    """Convert date string to ordinal number."""
    if not date_str:
        return 0
    try:
        date_part = date_str.split("T")[0] if "T" in date_str else date_str
        y, m, d = map(int, date_part.split("-"))
        return y * 366 + m * 31 + d
    except Exception:
        return 0


# ==================== Entailment Scoring ====================

def _lcs_length(a: str, b: str) -> int:
    """Compute LCS length between two strings."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[n][m]


def entail_rouge_recall(premise: str, hypothesis: str) -> float:
    """
    Compute entailment score using ROUGE-L recall.
    
    Uses hypothesis as reference, returns LCS / len(hypothesis).
    """
    premise = normalize_text(premise)
    hypothesis = normalize_text(hypothesis)
    
    if not hypothesis:
        return 0.0
    
    if _HAS_ROUGE:
        rouge = Rouge(metrics=['rouge-l'])
        tokens1 = ' '.join(tokenize_zh(premise))
        tokens2 = ' '.join(tokenize_zh(hypothesis))
        if not tokens1 or not tokens2:
            return 0.0
        try:
            score = rouge.get_scores(tokens1, tokens2)[0]['rouge-l']['r']
            return float(score)
        except Exception:
            pass
    
    # Fallback: character-level LCS
    L = _lcs_length(premise, hypothesis)
    return L / max(1, len(hypothesis))


def entail_batch(
    text_pairs: List[Tuple[str, str]], 
    backend: str = "rouge"
) -> List[float]:
    """
    Compute entailment scores for a batch of text pairs.
    
    Args:
        text_pairs: List of (premise, hypothesis) tuples
        backend: "rouge" or "nli"
        
    Returns:
        List of entailment scores
    """
    if not text_pairs:
        return []
    
    # Currently only ROUGE backend is implemented
    # NLI backend can be added if needed
    return [entail_rouge_recall(p, h) for p, h in text_pairs]


# ==================== Time Kernel ====================

def compute_tau(ref_times_ord: np.ndarray, config: Dict = None) -> float:
    """
    Compute time scale τ from reference timeline span.
    
    τ = clip(α_τ × span, τ_min, τ_max)
    """
    config = config or DEFAULT_CONFIG
    
    if len(ref_times_ord) == 0:
        return config.get("tau_min", 4.0)
    
    span = float(np.max(ref_times_ord) - np.min(ref_times_ord))
    alpha = config.get("alpha_tau", 0.1)
    tau_min = config.get("tau_min", 4.0)
    tau_max = config.get("tau_max", 21.0)
    
    if span <= 0:
        return tau_min
    
    return float(np.clip(alpha * span, tau_min, tau_max))


def gaussian_kernel(delta_days: float, tau: float) -> float:
    """
    Gaussian time decay kernel.
    
    w_ij = exp(-(Δ_ij / τ)²)
    """
    if tau is None or tau <= 0:
        return 1.0
    x = float(delta_days) / float(tau)
    return float(np.exp(-(x * x)))


def normalize_score(x: float, lo: float, hi: float) -> float:
    """
    Normalize score to [0, 1] with clipping.
    
    φ(x) = clip(x, L, H) - L) / (H - L)
    """
    x = float(np.clip(x, lo, hi))
    return (x - lo) / max(hi - lo, 1e-6)


# ==================== Atom Extraction ====================

def extract_atoms(node: Dict[str, Any]) -> List[str]:
    """
    Extract atoms from a timeline node.
    
    Atoms are clause-like parts with complete sentence structure.
    Falls back to summary/title if no atoms field.
    """
    atoms = node.get("atoms", [])
    if isinstance(atoms, list) and atoms:
        return [a.strip() for a in atoms if isinstance(a, str) and a.strip()]
    
    # Fallback
    s = (node.get("summary") or node.get("title") or "").strip()
    return [s] if s else []


def join_atoms(atoms: List[str]) -> str:
    """Join atoms into a single text."""
    atoms = [a.strip() for a in (atoms or []) if isinstance(a, str) and a.strip()]
    return "。".join(atoms)


# ==================== ACF Metrics ====================

def compute_alignment(
    gen_nodes: List[Dict[str, Any]],
    ref_nodes: List[Dict[str, Any]],
    config: Dict = None,
) -> Tuple[float, List[Dict], Dict]:
    """
    Compute Alignment (A) metric.
    
    For each reference bin j, finds best matching generated node i* that
    maximizes S_ij = φ(entF1_ij) × w_ij
    
    A = (1/K) Σ_j S_{i*(j),j}
    
    Returns:
        alignment_score: A metric value
        matches: List of match details per bin
        details: Additional statistics
    """
    config = config or DEFAULT_CONFIG
    K = len(ref_nodes)
    
    if K == 0:
        return 0.0, [], {"reason": "empty reference"}
    if not gen_nodes:
        return 0.0, [], {"K": K, "reason": "empty generated"}
    
    # Compute τ from reference timeline
    ref_times_ord = np.array([date_ordinal(n.get("time", "")) for n in ref_nodes])
    tau = compute_tau(ref_times_ord, config)
    
    # Extract atoms
    ref_atoms = [extract_atoms(n) for n in ref_nodes]
    ref_texts = [join_atoms(a) for a in ref_atoms]
    
    G = len(gen_nodes)
    gen_times_ord = np.array([date_ordinal(n.get("time", "")) for n in gen_nodes])
    gen_atoms = [extract_atoms(n) for n in gen_nodes]
    gen_texts = [join_atoms(a) for a in gen_atoms]
    
    # Time window for candidate filtering
    win_days = int(max(21.0, 2.0 * tau))
    
    # Hyperparameters
    clip_lo = config.get("clip_lo", 0.15)
    clip_hi = config.get("clip_hi", 0.85)
    topk_frac = config.get("topk_frac", 0.6)
    
    matches = []
    Sj_list = []
    
    for j in range(K):
        # Find candidates within time window
        cand_idx = [
            i for i in range(G) 
            if abs(int(gen_times_ord[i]) - int(ref_times_ord[j])) <= win_days
        ]
        if not cand_idx:
            cand_idx = list(range(G))
        
        best = None
        
        for i in cand_idx:
            # Compute entailment scores
            Ra = ref_atoms[j]
            Ga = gen_atoms[i]
            g_text = gen_texts[i]
            r_text = ref_texts[j]
            
            # entR: gen -> ref atoms (recall)
            if Ra:
                entR_scores = [entail_rouge_recall(g_text, ra) for ra in Ra]
                # Top-k mean
                k = max(1, int(math.ceil(len(entR_scores) * topk_frac)))
                entR = float(np.mean(sorted(entR_scores, reverse=True)[:k]))
            else:
                entR = 0.0
            
            # entP: ref -> gen atoms (precision)
            if Ga:
                entP_scores = [entail_rouge_recall(r_text, ga) for ga in Ga]
                k = max(1, int(math.ceil(len(entP_scores) * topk_frac)))
                entP = float(np.mean(sorted(entP_scores, reverse=True)[:k]))
            else:
                entP = 0.0
            
            # F1
            entF1 = (2.0 * entR * entP) / (entR + entP + 1e-8)
            entF1_scaled = normalize_score(entF1, clip_lo, clip_hi)
            
            # Time weight
            delta_days = abs(int(gen_times_ord[i]) - int(ref_times_ord[j]))
            time_w = gaussian_kernel(delta_days, tau)
            
            # Combined score
            Sij = entF1_scaled * time_w
            
            if best is None or Sij > best["score"]:
                best = {
                    "i": i,
                    "entR": entR,
                    "entP": entP,
                    "entF1": entF1,
                    "entF1_scaled": entF1_scaled,
                    "delta_days": delta_days,
                    "time_weight": time_w,
                    "score": Sij,
                }
        
        if best is None:
            matches.append({"j": j, "covered": False, "score": 0.0})
            Sj_list.append(0.0)
        else:
            matches.append({
                "j": j,
                "i": best["i"],
                "covered": True,
                **best,
            })
            Sj_list.append(best["score"])
    
    A = float(np.mean(Sj_list)) if Sj_list else 0.0
    
    details = {
        "A": A,
        "K": K,
        "G": G,
        "tau": tau,
        "win_days": win_days,
    }
    
    return A, matches, details


def compute_coverage(
    matches: List[Dict],
    config: Dict = None,
) -> Tuple[float, Dict]:
    """
    Compute Coverage (C) metric.
    
    A reference event j is covered if:
        S_{i*(j),j} >= θ  OR  (φ(entF1) >= θ_c AND w >= θ_t)
    
    C = (1/K) Σ_j 1{covered(j)}
    """
    config = config or DEFAULT_CONFIG
    K = len(matches)
    
    if K == 0:
        return 0.0, {"reason": "no matches"}
    
    theta = config.get("theta", 0.50)
    theta_c = config.get("theta_c", 0.38)
    theta_t = config.get("theta_t", 0.60)
    
    covered_count = 0
    bin_status = []
    
    for m in matches:
        if not m.get("covered", False):
            bin_status.append(False)
            continue
        
        score = m.get("score", 0.0)
        entF1_scaled = m.get("entF1_scaled", 0.0)
        time_w = m.get("time_weight", 0.0)
        
        # Coverage condition
        is_covered = (score >= theta) or (entF1_scaled >= theta_c and time_w >= theta_t)
        bin_status.append(is_covered)
        
        if is_covered:
            covered_count += 1
    
    C = float(covered_count) / float(K)
    
    details = {
        "C": C,
        "K": K,
        "covered": covered_count,
        "theta": theta,
        "theta_c": theta_c,
        "theta_t": theta_t,
        "bin_status": bin_status,
    }
    
    return C, details


def compute_factuality(
    matches: List[Dict],
    config: Dict = None,
) -> Tuple[float, Dict]:
    """
    Compute Factuality (F) metric.
    
    F = (1/K) Σ_j (entR_scaled_{i*(j),j} × w_{i*(j),j})
    
    Measures faithfulness of generated content to reference.
    """
    config = config or DEFAULT_CONFIG
    K = len(matches)
    
    if K == 0:
        return 0.0, {"reason": "no matches"}
    
    clip_lo = config.get("clip_lo", 0.15)
    clip_hi = config.get("clip_hi", 0.85)
    
    F_scores = []
    
    for m in matches:
        if not m.get("covered", False):
            F_scores.append(0.0)
            continue
        
        entR = m.get("entR", 0.0)
        time_w = m.get("time_weight", 1.0)
        
        entR_scaled = normalize_score(entR, clip_lo, clip_hi)
        F_scores.append(entR_scaled * time_w)
    
    F = float(np.mean(F_scores)) if F_scores else 0.0
    
    details = {
        "F": F,
        "K": K,
        "per_bin": F_scores,
    }
    
    return F, details


def evaluate_acf(
    gen_timeline: List[Dict[str, Any]],
    ref_timeline: List[Dict[str, Any]],
    config: Dict = None,
) -> Dict[str, Any]:
    """
    Evaluate a generated timeline against reference using ACF metrics.
    
    Args:
        gen_timeline: Generated timeline nodes
        ref_timeline: Reference timeline nodes
        config: Evaluation configuration
        
    Returns:
        Dict with A, C, F scores and average
    """
    config = config or DEFAULT_CONFIG
    
    # Alignment
    A, matches, align_details = compute_alignment(gen_timeline, ref_timeline, config)
    
    # Coverage
    C, cov_details = compute_coverage(matches, config)
    
    # Factuality
    F, fact_details = compute_factuality(matches, config)
    
    # Average (simple mean)
    average = (A + C + F) / 3.0
    
    return {
        "alignment": A,
        "coverage": C,
        "factuality": F,
        "average": average,
        "details": {
            "alignment": align_details,
            "coverage": cov_details,
            "factuality": fact_details,
            "matches": matches,
        },
    }

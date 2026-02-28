#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate generated timelines using ACF metrics.

Usage:
    python scripts/evaluate.py \
        --pred outputs/predictions.jsonl \
        --gold data/gold_reference.jsonl \
        --K N \
        --outfile outputs/eval_results.json
"""
import os
import argparse
import json
from typing import Dict, Any, List
from tqdm import tqdm
import numpy as np

from cast.utils import iter_jsonl
from cast.evaluation import evaluate_acf, DEFAULT_CONFIG


def load_reference(path: str) -> Dict[str, Dict[str, Any]]:
    """Load reference timelines indexed by topic_id."""
    ref_data = {}
    
    for rec in iter_jsonl(path):
        topic_id = rec.get("id") or rec.get("topic_id")
        if topic_id is None:
            continue
        topic_id = str(topic_id)
        
        ref_data[topic_id] = {
            "timeline": rec.get("timeline", []),
            "meta_timeline": rec.get("meta_timeline", {}),
        }
    
    return ref_data


def get_timeline_for_K(ref: Dict[str, Any], K: str) -> List[Dict[str, Any]]:
    """Extract timeline for specific granularity."""
    if K == "N":
        return ref.get("timeline", [])
    
    meta = ref.get("meta_timeline", {})
    if K in meta:
        node = meta[K]
        if isinstance(node, dict) and "timeline" in node:
            return node["timeline"]
        if isinstance(node, list):
            return node
    
    # Fallback to main timeline
    return ref.get("timeline", [])


def main():
    parser = argparse.ArgumentParser(description="Evaluate timelines with ACF metrics")
    parser.add_argument("--pred", required=True, help="Predictions JSONL")
    parser.add_argument("--gold", required=True, help="Gold reference JSONL")
    parser.add_argument("--K", required=True, choices=["N", "10", "5"])
    parser.add_argument("--outfile", required=True, help="Output JSON path")
    
    # ACF hyperparameters
    parser.add_argument("--alpha_tau", type=float, default=0.1)
    parser.add_argument("--clip_lo", type=float, default=0.15)
    parser.add_argument("--clip_hi", type=float, default=0.85)
    parser.add_argument("--theta", type=float, default=0.50)
    parser.add_argument("--theta_c", type=float, default=0.38)
    parser.add_argument("--theta_t", type=float, default=0.60)
    
    args = parser.parse_args()
    
    # Build config
    config = DEFAULT_CONFIG.copy()
    config.update({
        "alpha_tau": args.alpha_tau,
        "clip_lo": args.clip_lo,
        "clip_hi": args.clip_hi,
        "theta": args.theta,
        "theta_c": args.theta_c,
        "theta_t": args.theta_t,
    })
    
    # Load reference
    print(f"Loading reference from {args.gold}...")
    ref_data = load_reference(args.gold)
    print(f"Loaded {len(ref_data)} reference topics")
    
    # Evaluate predictions
    results = {}
    all_A, all_C, all_F = [], [], []
    
    for rec in tqdm(list(iter_jsonl(args.pred)), desc="Evaluating"):
        topic_id = rec.get("id") or rec.get("topic_id")
        if topic_id is None:
            continue
        topic_id = str(topic_id)
        
        if topic_id not in ref_data:
            print(f"Warning: {topic_id} not in reference, skipping")
            continue
        
        gen_timeline = rec.get("timeline", [])
        ref_timeline = get_timeline_for_K(ref_data[topic_id], args.K)
        
        if not ref_timeline:
            print(f"Warning: {topic_id} has no reference for K={args.K}, skipping")
            continue
        
        # Evaluate
        eval_result = evaluate_acf(gen_timeline, ref_timeline, config)
        
        results[topic_id] = {
            "alignment": eval_result["alignment"],
            "coverage": eval_result["coverage"],
            "factuality": eval_result["factuality"],
            "average": eval_result["average"],
        }
        
        all_A.append(eval_result["alignment"])
        all_C.append(eval_result["coverage"])
        all_F.append(eval_result["factuality"])
    
    # Compute overall
    if all_A:
        overall = {
            "alignment": float(np.mean(all_A)),
            "coverage": float(np.mean(all_C)),
            "factuality": float(np.mean(all_F)),
            "average": float(np.mean([np.mean(all_A), np.mean(all_C), np.mean(all_F)])),
            "n_topics": len(all_A),
        }
        results["overall"] = overall
        
        print("\n=== Overall Results ===")
        print(f"Alignment (A):   {overall['alignment']:.4f}")
        print(f"Coverage (C):    {overall['coverage']:.4f}")
        print(f"Factuality (F):  {overall['factuality']:.4f}")
        print(f"Average:         {overall['average']:.4f}")
        print(f"Topics:          {overall['n_topics']}")
    
    # Save results
    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
    with open(args.outfile, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {args.outfile}")


if __name__ == "__main__":
    main()

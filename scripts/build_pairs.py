#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build alignment pairs for CAST training.

This script mines evidence-summary pairs from the dataset:
1. For each gold summary, scores candidates using Equation (1)
2. Selects top-k diverse evidence using MMR
3. Outputs pairs in JSONL format for training

Usage:
    python scripts/build_pairs.py \
        --cand data/cand.tar.gz \
        --gold data/gold_reference.jsonl \
        --lengths data/lengths.json \
        --outdir outputs/pairs \
        --embed_model moka-ai/m3e-large
"""
import os
import argparse
from tqdm import tqdm

try:
    import ujson
except ImportError:
    import json as ujson

from cast.utils import (
    index_cand_tar,
    stream_cand_jsonl_from_tar,
    iter_jsonl,
    read_json_file,
)
from cast.utils.embedding import EmbeddingEncoder
from cast.data.mining import (
    MiningConfig,
    extract_gold_timelines,
    normalize_gold_list,
    clean_candidate,
    mine_pairs_for_topic,
)


def count_jsonl_records(path: str) -> int:
    """Count non-empty lines in JSONL file."""
    n = 0
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def load_lengths(path: str) -> dict:
    """Load topic lengths from JSON file."""
    if not path or not os.path.exists(path):
        return {}
    obj = read_json_file(path)
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            try:
                out[str(k)] = int(v)
            except Exception:
                continue
    elif isinstance(obj, list):
        for it in obj:
            try:
                out[str(it["id"])] = int(it["N"])
            except Exception:
                continue
    return out


def main():
    parser = argparse.ArgumentParser(description="Build alignment pairs for CAST training")
    parser.add_argument("--cand", required=True, help="Path to candidate tar.gz archive")
    parser.add_argument("--gold", required=True, help="Path to gold reference JSONL")
    parser.add_argument("--lengths", default=None, help="Path to lengths.json (optional)")
    parser.add_argument("--outdir", required=True, help="Output directory for pairs")
    
    # Scoring weights (Equation 1)
    parser.add_argument("--alpha", type=float, default=1.0, help="Semantic similarity weight")
    parser.add_argument("--beta", type=float, default=0.5, help="BM25 weight")
    parser.add_argument("--gamma", type=float, default=0.05, help="Time penalty weight")
    parser.add_argument("--delta", type=float, default=0.5, help="Same-day bonus weight")
    
    # Evidence selection
    parser.add_argument("--L_N", type=int, default=6, help="Max evidence for N-granularity")
    parser.add_argument("--L_10", type=int, default=4, help="Max evidence for 10-granularity")
    parser.add_argument("--L_5", type=int, default=3, help="Max evidence for 5-granularity")
    parser.add_argument("--dedup_thr", type=float, default=0.85, help="MMR dedup threshold")
    
    # Models
    parser.add_argument("--embed_model", type=str, default="moka-ai/m3e-large")
    
    # BM25
    parser.add_argument("--bm25_use_title", type=bool, default=True)
    parser.add_argument("--bm25_use_content", type=bool, default=True)
    
    parser.add_argument("--max_evidence_chars", type=int, default=2000)
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # Create config
    config = MiningConfig(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        delta=args.delta,
        L_N=args.L_N,
        L_10=args.L_10,
        L_5=args.L_5,
        dedup_thr=args.dedup_thr,
        embed_model=args.embed_model,
        bm25_use_title=args.bm25_use_title,
        bm25_use_content=args.bm25_use_content,
        max_evidence_chars=args.max_evidence_chars,
    )
    
    # Index candidates
    print(f"Indexing candidates from {args.cand}...")
    tar_idx = index_cand_tar(args.cand)
    
    # Load lengths
    id2N = load_lengths(args.lengths)
    
    # Initialize encoder
    print(f"Loading embedding model: {args.embed_model}...")
    encoder = EmbeddingEncoder(args.embed_model)
    
    # Output files
    out_N = open(os.path.join(args.outdir, "pairs_N.jsonl"), "w", encoding="utf-8")
    out_10 = open(os.path.join(args.outdir, "pairs_10.jsonl"), "w", encoding="utf-8")
    out_5 = open(os.path.join(args.outdir, "pairs_5.jsonl"), "w", encoding="utf-8")
    
    out_files = {"N": out_N, "10": out_10, "5": out_5}
    
    try:
        total = count_jsonl_records(args.gold)
        
        for rec in tqdm(iter_jsonl(args.gold), desc="Mining pairs", total=total):
            topic_id = rec.get("id") or rec.get("topic_id") or rec.get("topicId")
            if topic_id is None:
                continue
            topic_id = str(topic_id)
            
            # Extract gold timelines
            gold_tl = extract_gold_timelines(rec)
            gold_N = normalize_gold_list(gold_tl.get("N", []))
            gold_10 = normalize_gold_list(gold_tl.get("10", []))
            gold_5 = normalize_gold_list(gold_tl.get("5", []))
            
            if not any([gold_N, gold_10, gold_5]):
                continue
            
            # Load candidates
            cands_raw = list(stream_cand_jsonl_from_tar(tar_idx, topic_id))
            cands = [z for z in (clean_candidate(c) for c in cands_raw) if z is not None]
            
            if not cands:
                continue
            
            # Mine pairs
            gold_timelines = {"N": gold_N, "10": gold_10, "5": gold_5}
            pairs = mine_pairs_for_topic(topic_id, gold_timelines, cands, encoder, config)
            
            # Write output
            for K in ["N", "10", "5"]:
                for rec_out in pairs.get(K, []):
                    out_files[K].write(ujson.dumps(rec_out, ensure_ascii=False) + "\n")
        
        print(f"[Done] Pairs written to {args.outdir}")
        
    finally:
        for f in out_files.values():
            f.close()


if __name__ == "__main__":
    main()

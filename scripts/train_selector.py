#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the Evidence Selector for CAST.

This script trains a logistic regression classifier that predicts
which candidate articles are good evidence for a timeline node.

Usage:
    python scripts/train_selector.py \
        --cand data/cand.tar.gz \
        --pairs outputs/pairs/pairs_N.jsonl \
        --K N \
        --embed_model moka-ai/m3e-large \
        --out_model outputs/selector/selector_N.pkl
"""
import argparse

from cast.models.selector import SelectorConfig, train_selector


def main():
    parser = argparse.ArgumentParser(description="Train CAST Evidence Selector")
    parser.add_argument("--cand", required=True, help="Path to candidate tar.gz archive")
    parser.add_argument("--pairs", required=True, nargs="+", help="Paths to pairs JSONL files")
    parser.add_argument("--K", required=True, choices=["N", "10", "5"], help="Granularity")
    parser.add_argument("--embed_model", type=str, default="moka-ai/m3e-large")
    parser.add_argument("--out_model", type=str, default="./outputs/selector/selector.pkl")
    
    # Training parameters
    parser.add_argument("--neg_downsample", type=float, default=0.5,
                        help="Negative sample downsampling ratio")
    parser.add_argument("--topk_near", type=int, default=80,
                        help="Max candidates per window")
    parser.add_argument("--days_radius", type=float, default=None,
                        help="Optional time radius filter (days)")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Create config
    config = SelectorConfig(
        embed_model=args.embed_model,
        neg_downsample=args.neg_downsample,
        topk_near=args.topk_near,
        days_radius=args.days_radius,
        seed=args.seed,
    )
    
    # Train
    stats = train_selector(
        cand_tar_path=args.cand,
        pairs_paths=args.pairs,
        K=args.K,
        config=config,
        output_path=args.out_model,
        verbose=True,
    )
    
    print(f"Training complete: {stats}")


if __name__ == "__main__":
    main()

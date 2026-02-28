#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the CAST Summarizer with LoRA.

This script fine-tunes a language model (e.g., Qwen2.5-7B-Instruct) using LoRA
to generate timeline summaries from evidence.

Usage:
    python scripts/train_summarizer.py \
        --base_model Qwen/Qwen2.5-7B-Instruct \
        --pairs_train outputs/pairs/pairs_N.train.jsonl \
        --pairs_val outputs/pairs/pairs_N.val.jsonl \
        --K N \
        --outdir outputs/summarizer/N
"""
import os
import sys
import math
import random
import argparse
from typing import Optional, Dict

# Disable tokenizer parallelism to avoid fork issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType

from cast.data.prompts import PromptConfig
from cast.data.dataset import PairsDataset, collate_fn


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_model_device(model: torch.nn.Module) -> torch.device:
    """Get device of model's trainable parameters."""
    for p in model.parameters():
        if p.requires_grad:
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(model, tokenizer, out_dir: str, metrics: Optional[Dict] = None):
    """Save model checkpoint."""
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    if metrics is not None:
        import json
        with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train CAST Summarizer with LoRA")
    parser.add_argument("--base_model", required=True, help="Base model path or name")
    parser.add_argument("--pairs_train", required=True, help="Training pairs JSONL")
    parser.add_argument("--pairs_val", required=True, help="Validation pairs JSONL")
    parser.add_argument("--K", required=True, choices=["N", "10", "5"])
    parser.add_argument("--outdir", required=True, help="Output directory")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--bsz", type=int, default=2)
    parser.add_argument("--grad_acc", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    
    # Sequence lengths
    parser.add_argument("--max_src", type=int, default=1400)
    parser.add_argument("--max_tgt", type=int, default=64)
    parser.add_argument("--max_evidence", type=int, default=4)
    
    # LoRA configuration
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    
    # Prompt configuration
    parser.add_argument("--evidence_mode", type=str, default="title",
                        choices=["title", "title+lead", "atoms", "atoms+title"])
    parser.add_argument("--use_chat_template", action="store_true")
    
    # Data
    parser.add_argument("--subset_ratio", type=float, default=1.0,
                        help="Fraction of data to use (for debugging)")
    
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=False,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"Loading model from {args.base_model}...")
    quant_config = None
    device_map = None
    
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        device_map = "auto"
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
        quantization_config=quant_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    
    # Apply LoRA
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "w1", "w2", "w3"  # For some model architectures
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Move to GPU if not using 4bit
    if torch.cuda.is_available() and device_map is None:
        model.to("cuda")
    
    device = get_model_device(model)
    print(f"Model device: {device}")
    
    # Create prompt config
    prompt_config = PromptConfig(
        max_src=args.max_src,
        max_tgt=args.max_tgt,
        max_evidence=args.max_evidence,
        evidence_mode=args.evidence_mode,
        use_chat_template=args.use_chat_template,
    )
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = PairsDataset(
        args.pairs_train,
        tokenizer,
        prompt_config,
        K=args.K,
        subset_ratio=args.subset_ratio,
    )
    val_dataset = PairsDataset(
        args.pairs_val,
        tokenizer,
        prompt_config,
        K=args.K,
        subset_ratio=args.subset_ratio,
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bsz,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id or 0),
        pin_memory=True,
    )
    
    # Training setup
    steps_per_epoch = max(1, math.ceil(len(train_dataset) / (args.bsz * args.grad_acc)))
    total_steps = steps_per_epoch * args.epochs
    
    print(f"Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}")
    
    # Optimizer
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    bf16_ok = (device.type == "cuda") and torch.cuda.is_available()
    global_step = 0
    best_loss = float("inf")
    
    model.train()
    
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)
        
        step_in_epoch = 0
        running_loss = 0.0
        
        for batch in train_loader:
            # Move batch to device
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            # Forward pass
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=bf16_ok):
                outputs = model(**batch, output_hidden_states=False, use_cache=False)
                logits = outputs.logits[:, :-1, :].contiguous()
                labels = batch["labels"][:, 1:].contiguous()
                
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                    label_smoothing=0.05,
                    reduction="mean",
                ) / args.grad_acc
            
            loss.backward()
            running_loss += loss.item()
            
            # Gradient accumulation step
            if ((global_step + 1) % args.grad_acc) == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                
                step_in_epoch += 1
                pbar.update(1)
                pbar.set_postfix({
                    "loss": f"{running_loss:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}"
                })
                running_loss = 0.0
                
                if step_in_epoch >= steps_per_epoch:
                    break
            
            global_step += 1
        
        pbar.close()
        
        # Save checkpoint at end of epoch
        epoch_dir = os.path.join(args.outdir, f"checkpoint-epoch{epoch}")
        save_checkpoint(model, tokenizer, epoch_dir, {"epoch": epoch})
        print(f"Saved checkpoint to {epoch_dir}")
    
    # Save final model
    final_dir = os.path.join(args.outdir, "checkpoint-final")
    save_checkpoint(model, tokenizer, final_dir, {"epochs": args.epochs})
    print(f"Training complete. Final model saved to {final_dir}")


if __name__ == "__main__":
    main()

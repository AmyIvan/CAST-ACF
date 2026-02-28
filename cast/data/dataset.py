# -*- coding: utf-8 -*-
"""
PyTorch Dataset classes for CAST training.
"""
from __future__ import annotations
import os
from typing import List, Dict, Any, Optional
from tqdm import tqdm

try:
    import ujson
except ImportError:
    import json as ujson

from torch.utils.data import Dataset

from .prompts import (
    PromptConfig,
    build_evidence_items,
    assemble_prompt,
    apply_chat_template,
)


class PairsDataset(Dataset):
    """
    Dataset for CAST summarizer training.
    
    Loads pairs from JSONL files and converts them to (prompt, target) pairs.
    """
    
    def __init__(
        self,
        pairs_path: str,
        tokenizer,
        config: PromptConfig,
        K: str = "N",
        subset_ratio: float = 1.0,
        subset_seed: int = 42,
    ):
        """
        Initialize the dataset.
        
        Args:
            pairs_path: Path to pairs JSONL file
            tokenizer: Tokenizer for encoding
            config: Prompt configuration
            K: Granularity filter ("N", "10", "5")
            subset_ratio: Fraction of data to use (0, 1]
            subset_seed: Random seed for subsetting
        """
        import random
        
        self.tokenizer = tokenizer
        self.config = config
        self.K = K
        
        # Load all rows
        all_rows: List[Dict[str, Any]] = []
        
        with open(pairs_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Loading {os.path.basename(pairs_path)}"):
                if not line.strip():
                    continue
                rec = ujson.loads(line)
                
                # Filter by granularity
                if str(rec.get("K")) != str(K):
                    continue
                
                # Skip if no gold summary
                gold = (rec.get("gold_summary") or "").strip()
                if not gold:
                    continue
                
                all_rows.append(rec)
        
        # Subsample if needed
        ratio = max(0.0, min(1.0, float(subset_ratio)))
        if 0.0 < ratio < 1.0 and len(all_rows) > 0:
            k = max(1, int(round(len(all_rows) * ratio)))
            rng = random.Random(int(subset_seed))
            idxs = rng.sample(range(len(all_rows)), k)
            idxs.sort()
            self.rows = [all_rows[i] for i in idxs]
        else:
            self.rows = all_rows
        
        # Tokenize
        self.inputs: List[Dict[str, Any]] = []
        
        for row in tqdm(self.rows, desc="Tokenizing"):
            cand_pack = row.get("cand_pack") or []
            evid_items = build_evidence_items(cand_pack, config)
            topic_title = (row.get("topic_title") or row.get("title") or "").strip()
            
            # Build prompt
            user_content = assemble_prompt(
                tokenizer, config, K, topic_title, evid_items
            )
            user_content += "\n（仅用中文输出简短结论）"
            
            # Apply chat template if needed
            if config.use_chat_template:
                messages = [
                    {"role": "system", "content": config.sys_prompt},
                    {"role": "user", "content": user_content}
                ]
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt_text = user_content
            
            # Tokenize
            gold = row["gold_summary"].strip()
            
            enc = tokenizer(
                prompt_text, truncation=False, add_special_tokens=True
            )
            
            # Truncate from left if too long
            if len(enc["input_ids"]) > config.max_src:
                old_side = getattr(tokenizer, "truncation_side", "right")
                tokenizer.truncation_side = "left"
                enc = tokenizer(
                    prompt_text, 
                    truncation=True, 
                    max_length=config.max_src, 
                    add_special_tokens=True
                )
                tokenizer.truncation_side = old_side
            
            # Target
            tgt = gold + (tokenizer.eos_token or "")
            lab = tokenizer(
                tgt, 
                truncation=True, 
                max_length=config.max_tgt, 
                add_special_tokens=False
            )
            
            # Combine input and labels
            input_ids = enc["input_ids"] + lab["input_ids"]
            attention_mask = [1] * len(enc["input_ids"]) + [1] * len(lab["input_ids"])
            labels = [-100] * len(enc["input_ids"]) + lab["input_ids"]
            
            self.inputs.append({
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            })
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx]


def collate_fn(features: List[Dict[str, Any]], pad_token_id: int):
    """
    Collate function for DataLoader.
    
    Pads sequences to the same length.
    """
    import torch
    
    max_len = max(len(f["input_ids"]) for f in features)
    
    input_ids, labels, attention_mask = [], [], []
    
    for f in features:
        L = len(f["input_ids"])
        pad = max_len - L
        
        input_ids.append(f["input_ids"] + [pad_token_id] * pad)
        attention_mask.append(f["attention_mask"] + [0] * pad)
        labels.append(f["labels"] + [-100] * pad)
    
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

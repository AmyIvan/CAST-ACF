#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAST Inference Pipeline.

Generate multi-granularity timelines from candidate articles using:
1. Time window construction
2. Evidence selection (trained selector)
3. Summary generation (LoRA fine-tuned LLM)

Usage:
    python scripts/run_inference.py \
        --cand data/cand.tar.gz \
        --gold data/gold_reference.jsonl \
        --selector outputs/selector/selector_N.pkl \
        --summarizer outputs/summarizer/N/checkpoint-final \
        --K N \
        --outfile outputs/predictions.jsonl
"""
import os
import argparse
from typing import List, Dict, Any, Optional
from collections import Counter
from datetime import datetime, timezone
from tqdm import tqdm
import numpy as np

try:
    import ujson
except ImportError:
    import json as ujson

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from cast.utils import (
    index_cand_tar,
    stream_cand_jsonl_from_tar,
    iter_jsonl,
    parse_time_any,
    build_windows,
    in_window,
)
from cast.utils.embedding import EmbeddingEncoder
from cast.models.selector import load_selector, predict_selector
from cast.data.prompts import PromptConfig, build_evidence_items, assemble_prompt


def clean_candidate(c: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Clean and validate a candidate article."""
    t = parse_time_any(c.get("time"))
    if t is None:
        return None
    return {
        "time": t,
        "title": (c.get("title") or "").strip(),
        "content": (c.get("content") or c.get("abstract") or c.get("desc") or "").strip(),
        "raw": c,
    }


def filter_time_outliers(times: List[datetime]) -> List[int]:
    """
    Filter outlier timestamps using IQR method.
    Returns indices of valid timestamps.
    """
    if len(times) < 4:
        return list(range(len(times)))
    
    ords = np.array([t.toordinal() for t in times])
    q1, q3 = np.percentile(ords, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    
    valid_idx = [i for i, o in enumerate(ords) if lo <= o <= hi]
    return valid_idx if valid_idx else list(range(len(times)))


def build_time_windows_dynamic(
    cand_times: List[datetime],
    K: str,
    target_nodes: Optional[int] = None,
) -> List[tuple]:
    """
    Build time windows for timeline generation.
    
    For K=N: Dynamic based on unique days (0.8 * unique_days)
    For K=10/5: Fixed number of windows
    """
    if not cand_times:
        return []
    
    # Get unique days
    day_set = set()
    for t in cand_times:
        day_set.add((t.year, t.month, t.day))
    unique_days = len(day_set)
    
    # Determine number of windows
    if K == "N":
        if target_nodes:
            n_windows = target_nodes
        else:
            n_windows = max(1, int(0.8 * unique_days))
    elif K == "10":
        n_windows = 10
    elif K == "5":
        n_windows = 5
    else:
        n_windows = max(1, int(0.8 * unique_days))
    
    # Sort times
    sorted_times = sorted(cand_times)
    
    if n_windows >= len(sorted_times):
        # One window per time point
        return [(t, t) for t in sorted_times]
    
    # Evenly spaced window centers
    indices = np.linspace(0, len(sorted_times) - 1, n_windows, dtype=int)
    center_times = [sorted_times[i] for i in indices]
    
    # Build windows using midpoints
    windows = build_windows(center_times)
    
    return list(zip(center_times, windows))


def select_evidence(
    selector: Dict[str, Any],
    encoder: EmbeddingEncoder,
    cands: List[Dict[str, Any]],
    window: tuple,
    max_evidence: int = 4,
    used_idx: Optional[set] = None,
) -> List[int]:
    """
    Select evidence for a time window using the trained selector.
    
    Returns indices of selected candidates.
    """
    used_idx = used_idx or set()
    
    # Filter candidates in window
    in_window_idx = []
    for i, c in enumerate(cands):
        if i in used_idx:
            continue
        if in_window(c["time"], window):
            in_window_idx.append(i)
    
    if not in_window_idx:
        # Fallback: use nearest candidates
        center = window[0] if isinstance(window[0], datetime) else datetime.now(timezone.utc)
        dists = [(i, abs((c["time"] - center).total_seconds())) 
                 for i, c in enumerate(cands) if i not in used_idx]
        dists.sort(key=lambda x: x[1])
        in_window_idx = [x[0] for x in dists[:max_evidence * 2]]
    
    if not in_window_idx:
        return []
    
    # Prepare features
    titles = [cands[i]["title"] for i in in_window_idx]
    texts = [cands[i]["content"] for i in in_window_idx]
    ords = [cands[i]["time"].toordinal() for i in in_window_idx]
    
    # Predict scores
    scores = predict_selector(selector, encoder, titles, texts, ords)
    
    # Select top-k
    ranked = sorted(zip(in_window_idx, scores), key=lambda x: -x[1])
    selected = [idx for idx, _ in ranked[:max_evidence]]
    
    return selected


def generate_summary(
    model,
    tokenizer,
    evidence: List[Dict[str, Any]],
    K: str,
    prompt_config: PromptConfig,
    device: torch.device,
    max_new_tokens: int = 72,
    num_beams: int = 4,
) -> str:
    """Generate summary for selected evidence."""
    # Build prompt
    cand_pack = []
    for ev in evidence:
        cand_pack.append({
            "title": ev.get("title", ""),
            "content": ev.get("content", ""),
            "time": ev["time"].isoformat() if isinstance(ev["time"], datetime) else str(ev.get("time", "")),
        })
    
    evid_items = build_evidence_items(cand_pack, prompt_config)
    prompt = assemble_prompt(tokenizer, prompt_config, K, "", evid_items)
    
    # Apply chat template if needed
    if prompt_config.use_chat_template:
        messages = [
            {"role": "system", "content": prompt_config.sys_prompt},
            {"role": "user", "content": prompt}
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=prompt_config.max_src)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=16,
            num_beams=num_beams,
            length_penalty=0.9,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    summary = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    
    # Post-process
    summary = post_process_summary(summary, prompt_config.headline_char_limit)
    
    return summary


def post_process_summary(text: str, char_limit: int = 56) -> str:
    """Clean up generated summary."""
    import re
    
    if not text:
        return ""
    
    # Take first line
    text = text.split("\n")[0].strip()
    
    # Remove leading markers
    text = re.sub(r'^[\d.、\-\*]+\s*', '', text)
    text = re.sub(r'^(摘要|标题|总结|结论|答案)[：:]\s*', '', text)
    
    # Remove filler phrases
    fillers = ["根据以上信息", "综上所述", "总的来说", "简而言之", "总之"]
    for f in fillers:
        text = text.replace(f, "")
    
    # Truncate if too long
    if len(text) > char_limit:
        # Try to cut at sentence boundary
        for sep in ["。", "，", "；", " "]:
            idx = text.rfind(sep, 0, char_limit)
            if idx > char_limit // 2:
                text = text[:idx]
                break
        else:
            text = text[:char_limit]
    
    return text.strip()


def infer_topic(
    topic_id: str,
    cands: List[Dict[str, Any]],
    K: str,
    selector: Dict[str, Any],
    encoder: EmbeddingEncoder,
    model,
    tokenizer,
    prompt_config: PromptConfig,
    device: torch.device,
    target_nodes: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Generate timeline for a single topic."""
    if not cands:
        return []
    
    # Filter time outliers
    valid_idx = filter_time_outliers([c["time"] for c in cands])
    cands = [cands[i] for i in valid_idx]
    
    if not cands:
        return []
    
    # Build time windows
    cand_times = [c["time"] for c in cands]
    window_specs = build_time_windows_dynamic(cand_times, K, target_nodes)
    
    if not window_specs:
        return []
    
    # Generate timeline
    timeline = []
    used_idx = set()
    
    max_evidence = {"N": 6, "10": 4, "5": 3}.get(K, 4)
    
    for node_id, (center_time, window) in enumerate(window_specs):
        # Select evidence
        selected_idx = select_evidence(
            selector, encoder, cands, window, 
            max_evidence=max_evidence, used_idx=used_idx
        )
        
        if not selected_idx:
            continue
        
        # Update used indices (per-day deduplication)
        center_day = (center_time.year, center_time.month, center_time.day)
        for idx in selected_idx:
            c_day = (cands[idx]["time"].year, cands[idx]["time"].month, cands[idx]["time"].day)
            if c_day == center_day:
                used_idx.add(idx)
        
        # Generate summary
        evidence = [cands[i] for i in selected_idx]
        summary = generate_summary(
            model, tokenizer, evidence, K, prompt_config, device
        )
        
        if not summary:
            continue
        
        # Create timeline node
        node = {
            "id": f"{topic_id}_{K}_{node_id}",
            "time": center_time.isoformat(),
            "summary": summary,
            "cand_ids": selected_idx,
        }
        timeline.append(node)
    
    return timeline


def main():
    parser = argparse.ArgumentParser(description="CAST Inference Pipeline")
    parser.add_argument("--cand", required=True, help="Path to candidate tar.gz")
    parser.add_argument("--gold", required=True, help="Path to gold reference JSONL")
    parser.add_argument("--selector", required=True, help="Path to selector model")
    parser.add_argument("--summarizer", required=True, help="Path to summarizer checkpoint")
    parser.add_argument("--base_model", default=None, help="Base model (if not in checkpoint)")
    parser.add_argument("--K", required=True, choices=["N", "10", "5"])
    parser.add_argument("--outfile", required=True, help="Output JSONL path")
    
    # Model options
    parser.add_argument("--embed_model", default="moka-ai/m3e-large")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--use_chat_template", action="store_true")
    
    # Generation options
    parser.add_argument("--max_new_tokens", type=int, default=72)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--headline_limit", type=int, default=56)
    
    # Prompt options
    parser.add_argument("--evidence_mode", default="title",
                        choices=["title", "title+lead", "atoms", "atoms+title"])
    parser.add_argument("--max_evidence", type=int, default=4)
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load selector
    print(f"Loading selector from {args.selector}...")
    selector = load_selector(args.selector)
    
    # Load embedding encoder
    print(f"Loading embedding model: {args.embed_model}...")
    encoder = EmbeddingEncoder(args.embed_model)
    
    # Load summarizer
    print(f"Loading summarizer from {args.summarizer}...")
    
    # Determine base model
    base_model = args.base_model
    if base_model is None:
        # Try to find adapter_config.json
        adapter_config_path = os.path.join(args.summarizer, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            import json
            with open(adapter_config_path) as f:
                adapter_config = json.load(f)
                base_model = adapter_config.get("base_model_name_or_path")
    
    if base_model is None:
        raise ValueError("Could not determine base model. Please specify --base_model")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True,
        )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(model, args.summarizer)
    model.eval()
    
    # Prompt config
    prompt_config = PromptConfig(
        max_src=1400,
        max_tgt=64,
        max_evidence=args.max_evidence,
        evidence_mode=args.evidence_mode,
        headline_char_limit=args.headline_limit,
        use_chat_template=args.use_chat_template,
    )
    
    # Index candidates
    print(f"Indexing candidates from {args.cand}...")
    tar_idx = index_cand_tar(args.cand)
    
    # Process topics
    print("Processing topics...")
    
    with open(args.outfile, "w", encoding="utf-8") as out_f:
        for rec in tqdm(list(iter_jsonl(args.gold)), desc="Inference"):
            topic_id = rec.get("id") or rec.get("topic_id")
            if topic_id is None:
                continue
            topic_id = str(topic_id)
            
            # Load candidates
            cands_raw = list(stream_cand_jsonl_from_tar(tar_idx, topic_id))
            cands = [z for z in (clean_candidate(c) for c in cands_raw) if z is not None]
            
            if not cands:
                continue
            
            # Get target nodes if available
            target_nodes = None
            if args.K == "N":
                tl = rec.get("timeline")
                if isinstance(tl, list):
                    target_nodes = len(tl)
            
            # Generate timeline
            timeline = infer_topic(
                topic_id, cands, args.K,
                selector, encoder,
                model, tokenizer, prompt_config, device,
                target_nodes=target_nodes,
            )
            
            # Write output
            output_rec = {
                "id": topic_id,
                "K": args.K,
                "timeline": timeline,
            }
            out_f.write(ujson.dumps(output_rec, ensure_ascii=False) + "\n")
    
    print(f"Done! Output written to {args.outfile}")


if __name__ == "__main__":
    main()

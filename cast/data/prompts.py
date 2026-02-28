# -*- coding: utf-8 -*-
"""
Prompt templates for CAST summarizer training and inference.
"""
from __future__ import annotations
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class PromptConfig:
    """Configuration for prompt construction."""
    max_src: int = 1400           # Max source tokens
    max_tgt: int = 64             # Max target tokens
    max_evidence: int = 4         # Max evidence items
    evidence_mode: str = "title"  # "title", "title+lead", "atoms", "atoms+title"
    lead_chars: int = 120         # Max chars for lead sentence
    headline_char_limit: int = 56 # Target headline length
    use_chat_template: bool = False
    sys_prompt: str = "你是一个遵循指令的中文新闻节点摘要助手。"


# Prompt templates
PROMPT_HEAD = """你是一个多粒度时间线的节点摘要器。请根据提供的证据，为粒度 {K} 的时间线节点生成**中文短标题**（像新闻提要一样，极短、信息密集）。

# 证据（按时间）:
"""

PROMPT_TAIL = """
# 输出要求：
- 使用简体中文
- 只输出**1-2句**中文短标题
- **不超过{headline_limit}个汉字**
- 不要出现无根据的推断
- 要输出陈述句式的短标题
- 不要出现换行符
- 不要添加任何解释、背景或客套话
- 直接给出短标题本身

请给出短标题：
"""


def first_sentence(text: str, max_chars: int) -> str:
    """Extract first sentence up to max_chars."""
    text = (text or "").strip()
    if not text:
        return ""
    for sep in ["。", "！", "!", "？", "?", "\n"]:
        p = text.find(sep)
        if 0 <= p <= max_chars:
            text = text[:p+1]
            break
    return text[:max_chars]


def build_evidence_items(
    cand_pack: List[Dict[str, str]], 
    config: PromptConfig
) -> List[str]:
    """
    Build evidence items from candidate pack.
    
    Args:
        cand_pack: List of dicts with 'title', 'content', 'time', optionally 'atoms'
        config: Prompt configuration
        
    Returns:
        List of formatted evidence strings
    """
    items = []
    
    for x in cand_pack[:config.max_evidence]:
        t = str(x.get("time") or "")
        title = (x.get("title") or "").strip()
        content = (x.get("content") or "").strip()
        atoms = x.get("atoms") or []
        
        if config.evidence_mode == "title":
            body = f"[TITLE] {title}" if title else ""
            
        elif config.evidence_mode == "title+lead":
            lead = first_sentence(content, config.lead_chars)
            body = f"[TITLE] {title}"
            if lead:
                body += f"\n[LEAD] {lead}"
                
        elif config.evidence_mode == "atoms":
            if atoms:
                body = f"[ATOMS] {'；'.join(atoms)}"
            else:
                body = f"[TITLE] {title}" if title else ""
                
        elif config.evidence_mode == "atoms+title":
            if atoms:
                body = f"[ATOMS] {'；'.join(atoms)}"
                if title:
                    body += f"\n[TITLE] {title}"
            else:
                body = f"[TITLE] {title}" if title else ""
        else:
            body = f"[TITLE] {title}" if title else ""
        
        body = body.strip()
        if body:
            items.append(f"[TIME] {t}\n{body}")
    
    return items


def apply_chat_template(
    tokenizer, 
    content: str, 
    config: PromptConfig
) -> str:
    """Apply chat template if enabled."""
    if config.use_chat_template:
        messages = [
            {"role": "system", "content": config.sys_prompt},
            {"role": "user", "content": content}
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return content


def assemble_prompt(
    tokenizer,
    config: PromptConfig,
    K: str,
    topic_title: str,
    evid_items: List[str],
) -> str:
    """
    Assemble full prompt with budget constraint.
    
    Iteratively adds evidence items until token limit is reached.
    
    Args:
        tokenizer: Tokenizer for length checking
        config: Prompt configuration
        K: Granularity ("N", "10", "5")
        topic_title: Topic title (optional)
        evid_items: List of evidence strings
        
    Returns:
        Assembled prompt string
    """
    def _tok_len(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
    
    prefix = f"[TOPIC] {topic_title}\n\n" if topic_title else ""
    head = PROMPT_HEAD.format(K=K)
    tail = PROMPT_TAIL.format(headline_limit=config.headline_char_limit)
    
    # Iteratively add evidence items
    chosen: List[str] = []
    base = prefix + head
    
    for item in evid_items:
        trial = base + ("\n\n".join(chosen + [item])) + tail
        trial_chat = apply_chat_template(tokenizer, trial, config)
        if _tok_len(trial_chat) <= config.max_src:
            chosen.append(item)
        else:
            break
    
    # Assemble final prompt
    final_text = prefix + head + ("\n\n".join(chosen)) + tail
    final_chat = apply_chat_template(tokenizer, final_text, config)
    
    # If still too long, try to shorten
    if _tok_len(final_chat) > config.max_src:
        prefix2 = ""  # Remove topic
        tail2 = "\n请给出摘要：\n"
        final_text = prefix2 + head + ("\n\n".join(chosen)) + tail2
        final_chat = apply_chat_template(tokenizer, final_text, config)
        
        while chosen and _tok_len(final_chat) > config.max_src:
            chosen.pop()
            final_text = prefix2 + head + ("\n\n".join(chosen)) + tail2
            final_chat = apply_chat_template(tokenizer, final_text, config)
        
        if _tok_len(final_chat) > config.max_src:
            head2 = f"为粒度 {K} 的节点做中文摘要。证据：\n"
            final_text = head2 + ("\n\n".join(chosen)) + "\n请给出摘要：\n"
    
    return final_text


def render_training_example(
    cand_pack: List[Dict[str, Any]],
    gold_time: str,
    K: str,
    config: PromptConfig,
) -> str:
    """
    Render a training example prompt.
    
    Args:
        cand_pack: List of evidence items
        gold_time: Gold node timestamp
        K: Granularity
        config: Prompt configuration
        
    Returns:
        Formatted prompt string
    """
    evid_items = build_evidence_items(cand_pack, config)
    
    lines = [
        f"[SYSTEM]\n{config.sys_prompt}",
        f"[TASK]\n根据给定多条新闻证据，写出 {K} 粒度风格的一句话摘要（避免主观评价、使用事实）。",
        f"[TIME]\n{gold_time}",
        "[EVIDENCE]",
    ]
    
    for i, item in enumerate(evid_items[:config.max_evidence]):
        lines.append(f"({i+1}) {item}")
    
    lines.append(f"[STYLE]\n长度约{config.headline_char_limit}字；时间表达与给定时间一致；避免修饰与评价词。")
    lines.append("[OUTPUT]")
    
    return "\n".join(lines)

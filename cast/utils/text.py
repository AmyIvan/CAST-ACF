# -*- coding: utf-8 -*-
"""
Text processing utilities for Chinese NLP.
"""
from __future__ import annotations
import re
from typing import List, Iterable

try:
    import jieba
    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False

_WS = re.compile(r"\s+")


def zh_tokenize(s: str) -> List[str]:
    """
    Tokenize Chinese text using jieba.
    Falls back to character-level tokenization if jieba is not available.
    """
    s = s or ""
    s = s.strip()
    if not s:
        return []
    
    # Clean HTML tags and special characters
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"[^\w\u4e00-\u9fff]+", " ", s)
    s = _WS.sub(" ", s)
    
    if _HAS_JIEBA:
        return [x for x in jieba.cut(s) if x.strip()]
    else:
        # Fallback: character-level for Chinese, word-level for others
        tokens = []
        for match in re.finditer(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", s):
            tokens.append(match.group())
        return tokens


def safe_join(parts: Iterable[str], sep: str = " ", max_chars: int = None) -> str:
    """
    Safely join string parts, filtering out empty strings.
    
    Args:
        parts: Iterable of strings to join
        sep: Separator between parts
        max_chars: Maximum length of output (truncate if exceeded)
        
    Returns:
        Joined string
    """
    buf = []
    for p in parts:
        if p and isinstance(p, str):
            p = p.strip()
            if p:
                buf.append(p)
    
    out = sep.join(buf)
    
    if max_chars and len(out) > max_chars:
        return out[:max_chars]
    
    return out


def clean_for_eval(text: str, char_limit: int = None) -> str:
    """
    Light cleaning for evaluation (e.g., ROUGE scoring).
    
    - Keep only the first line
    - Remove answer-style prefixes
    - Remove leading enumeration/bullets
    - Filter unusual characters
    - Collapse whitespace
    - Strip trailing punctuation
    """
    if not text:
        return ""
    
    s = str(text).strip()
    
    # Keep only first line
    s = s.splitlines()[0]
    
    # Remove answer-style prefixes
    s = re.sub(r'^\s*(以下是|答案|正确答案|参考答案|结论|总结)\s*[:：\-]*\s*', '', s)
    
    # Remove leading enumeration/bullets
    s = re.sub(
        r'^\s*(?:[（(]?[第]?\d+[）)]、?|[①-⑩]|[A-Za-z]\)|[A-Za-z]\.|[0-9]+[.)、-]|[-*•]+)\s*',
        '', s
    )
    
    # Filter unusual characters (keep common Chinese/English/punctuation)
    allowed_re = re.compile(
        r"[^0-9A-Za-z_\-\u4e00-\u9fff，。！？、：；（）""''—《》〈〉·%\.，、：；！？""''\-·\s]"
    )
    s = allowed_re.sub('', s)
    
    # Collapse whitespace
    s = re.sub(r'\s+', ' ', s)
    
    # Strip trailing punctuation
    s = re.sub(r'[。！？!?\.\,，、；;：:]+$', '', s).strip()
    
    # Optional truncation
    if char_limit is not None and len(s) > char_limit:
        s = s[:char_limit].rstrip()
    
    return s

# -*- coding: utf-8 -*-
"""
BM25 scoring utilities for lexical similarity.
"""
from __future__ import annotations
from typing import List

from rank_bm25 import BM25Okapi
from .text import zh_tokenize


class BM25Scorer:
    """BM25 scorer for Chinese text."""
    
    def __init__(self, corpus_texts: List[str]):
        """
        Initialize BM25 scorer with a corpus.
        
        Args:
            corpus_texts: List of documents to index
        """
        self._tok = [zh_tokenize(t or "") for t in corpus_texts]
        self._bm25 = BM25Okapi(self._tok)
        self.corpus = corpus_texts

    def score_single(self, query: str, idx: int) -> float:
        """
        Get BM25 score for a single document.
        
        Args:
            query: Query text
            idx: Document index
            
        Returns:
            BM25 score
        """
        qtok = zh_tokenize(query or "")
        if not qtok:
            return 0.0
        scores = self._bm25.get_scores(qtok)
        if idx < 0 or idx >= len(scores):
            return 0.0
        return float(scores[idx])

    def scores(self, query: str) -> List[float]:
        """
        Get BM25 scores for all documents.
        
        Args:
            query: Query text
            
        Returns:
            List of BM25 scores for each document
        """
        qtok = zh_tokenize(query or "")
        if not qtok:
            return [0.0] * len(self.corpus)
        return list(map(float, self._bm25.get_scores(qtok)))

    def get_scores(self, query: str) -> List[float]:
        """Alias for scores() for compatibility."""
        return self.scores(query)

# -*- coding: utf-8 -*-
"""
Embedding encoder with SentenceTransformer and HuggingFace fallback.
"""
from __future__ import annotations
from typing import List, Optional
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class EmbeddingEncoder:
    """
    Embedding encoder supporting two loading modes:
      1) SentenceTransformer (if available and model is ST-compatible)
      2) HuggingFace AutoModel with mean pooling (fallback)
    """
    
    def __init__(self, model_name_or_path: str, device: Optional[str] = None):
        """
        Initialize the embedding encoder.
        
        Args:
            model_name_or_path: Path to model or HuggingFace model name
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
        """
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for EmbeddingEncoder")
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._st = None
        self._hf_model = None
        self._hf_tok = None
        self.dim = None

        # Try SentenceTransformer first
        try:
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer(model_name_or_path, device=self.device)
            v = self._st.encode(["test"], convert_to_numpy=True)[0]
            self.dim = int(v.shape[-1])
            return
        except Exception:
            self._st = None

        # Fallback to HuggingFace AutoModel
        from transformers import AutoModel, AutoTokenizer
        self._hf_tok = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self._hf_model = AutoModel.from_pretrained(
            model_name_or_path, trust_remote_code=True
        ).to(self.device)
        self._hf_model.eval()
        
        # Get embedding dimension
        with torch.no_grad():
            t = self._hf_tok(
                ["test"], padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            h = self._hf_model(**t).last_hidden_state
            attn = t["attention_mask"].unsqueeze(-1)
            num = (h * attn).sum(dim=1)
            den = attn.sum(dim=1).clamp(min=1e-6)
            v = (num / den).detach().cpu().numpy()[0]
            self.dim = int(v.shape[-1])

    def encode(
        self, 
        texts: List[str], 
        batch_size: int = 64, 
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            numpy array of shape (len(texts), dim)
        """
        if self._st is not None:
            return self._st.encode(
                texts, 
                batch_size=batch_size, 
                convert_to_numpy=True, 
                show_progress_bar=False, 
                normalize_embeddings=normalize
            )
        
        # HuggingFace path
        outs: List[np.ndarray] = []
        N = len(texts)
        B = max(1, batch_size)
        
        with torch.no_grad():
            for i in range(0, N, B):
                chunk = texts[i:i+B]
                t = self._hf_tok(
                    chunk, 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt", 
                    max_length=512
                ).to(self.device)
                
                h = self._hf_model(**t).last_hidden_state
                attn = t["attention_mask"].unsqueeze(-1)
                num = (h * attn).sum(dim=1)
                den = attn.sum(dim=1).clamp(min=1e-6)
                v = (num / den).detach().cpu().numpy()
                
                if normalize:
                    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
                
                outs.append(v)
        
        return np.vstack(outs)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float((a @ b) / (na * nb))

"""Sparse lexical retrieval over code descriptions."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

log = logging.getLogger(__name__)


@dataclass
class Hit:
    row: int
    score: float


class LexicalIndex:
    """TF-IDF index over a list of strings, searched by sparse dot product.

    `analyzer="char_wb"` gives robustness to misspellings and run-together
    words; `analyzer="word"` rewards exact token matches and rare words.
    """

    def __init__(self, analyzer: str = "char_wb", ngram_range=(3, 4), max_df: float = 0.2, min_df: int = 1):
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            sublinear_tf=True,
            dtype=np.float32,
            token_pattern=r"(?u)\b\w+\b" if analyzer == "word" else None,
        )
        self.matrix_t: sp.csr_matrix | None = None

    def fit(self, texts: list[str]) -> "LexicalIndex":
        m = self.vectorizer.fit_transform(texts)
        self.matrix_t = m.T.tocsr()
        self.size = m.shape[0]
        return self

    def search(self, queries: list[str], k: int, batch: int = 256) -> list[list[Hit]]:
        out: list[list[Hit]] = []
        q = self.vectorizer.transform(queries)
        for start in range(0, q.shape[0], batch):
            scores = (q[start : start + batch] @ self.matrix_t).tocsr()
            for i in range(scores.shape[0]):
                lo, hi = scores.indptr[i], scores.indptr[i + 1]
                if lo == hi:
                    out.append([])
                    continue
                data, idx = scores.data[lo:hi], scores.indices[lo:hi]
                if len(data) > k:
                    top = np.argpartition(-data, k)[:k]
                    data, idx = data[top], idx[top]
                order = np.argsort(-data, kind="stable")
                out.append([Hit(int(idx[j]), float(data[j])) for j in order])
        return out

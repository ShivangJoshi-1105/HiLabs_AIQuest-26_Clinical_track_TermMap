"""Dense (embedding) retrieval over code descriptions.

Two ways to supply dense results:

1. Precomputed hits: `artifacts/dense_hits_<system>.parquet` with columns
   query, rank, code, score. Produced by notebooks/build_dense_index.ipynb from
   the query list written by `python -m termmap export-queries`.
2. Live search: `artifacts/dense_<system>.emb.npy` and `.codes.npy` plus a
   local encoder (requirements-dense.txt). Used for queries that have no
   precomputed hits.

When neither is present the engine runs on lexical retrieval alone.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

log = logging.getLogger(__name__)


def _paths(cfg: Config, system: str, prefix: str = "dense") -> tuple[Path, Path, Path]:
    base = Path(cfg.artifact_dir)
    hits = "dense_hits" + prefix[len("dense"):]  # dense -> dense_hits, dense_biolord -> dense_hits_biolord
    return (
        base / f"{prefix}_{system}.emb.npy",
        base / f"{prefix}_{system}.codes.npy",
        base / f"{hits}_{system}.parquet",
    )


def load_encoder(name: str):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(0)
    use_gpu = torch.cuda.is_available() and os.environ.get("TERMMAP_FORCE_CPU", "0") != "1"
    if not use_gpu:
        torch.set_num_threads(int(os.environ.get("TERMMAP_CPU_THREADS", "2")))
    return SentenceTransformer(name, device="cuda" if use_gpu else "cpu", token=os.environ.get("HF_TOKEN") or None)


def encode(model, texts: list[str], batch_size: int = 256) -> np.ndarray:
    emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                       show_progress_bar=len(texts) > 5000, convert_to_numpy=True)
    return emb.astype(np.float32)


def build(cfg: Config, system: str, descriptions, prefix: str = "dense") -> None:
    """Embed every description of one terminology and cache the result (GPU recommended)."""
    emb_path, code_path, _ = _paths(cfg, system, prefix)
    model = load_encoder(cfg.dense_models[prefix])
    emb = encode(model, descriptions["norm"].tolist())
    np.save(emb_path, emb.astype(np.float16))
    np.save(code_path, descriptions["code"].to_numpy().astype(str))
    log.info("%s: wrote %s %s", system, emb_path.name, emb.shape)


def top_codes(sims: np.ndarray, codes: np.ndarray, k: int) -> list[tuple[str, float, str]]:
    """Best k distinct codes from one row of description similarities."""
    n = min(k * 3, len(sims) - 1)
    top = np.argpartition(-sims, n)[:n]
    top = top[np.argsort(-sims[top], kind="stable")]
    seen, out = set(), []
    for j in top:
        c = str(codes[j])
        if c in seen:
            continue
        seen.add(c)
        out.append((c, float(sims[j]), ""))
        if len(out) == k:
            break
    return out


class DenseRetriever:
    def __init__(self, cfg: Config, system: str, hits: dict | None, emb_path: Path | None, code_path: Path | None,
                 prefix: str = "dense"):
        self.cfg = cfg
        self.system = system
        self.prefix = prefix
        self._cache: dict[str, list] = hits or {}
        self.emb_path, self.code_path = emb_path, code_path
        self.emb = self.codes = self.model = None
        self.missing = 0

    @classmethod
    def load(cls, cfg: Config, system: str, prefix: str = "dense") -> "DenseRetriever | None":
        emb_path, code_path, hits_path = _paths(cfg, system, prefix)
        hits = None
        if hits_path.exists():
            h = pd.read_parquet(hits_path).sort_values(["query", "rank"])
            hits = {q: list(zip(g["code"].astype(str), g["score"].astype(float), [""] * len(g)))
                    for q, g in h.groupby("query", sort=False)}
            log.info("%s: precomputed %s hits for %d queries", system, prefix, len(hits))
        live = emb_path.exists() and code_path.exists()
        if hits is None and not live:
            return None
        return cls(cfg, system, hits, emb_path if live else None, code_path if live else None, prefix)

    def _live(self, queries: list[str], k: int, batch: int = 256):
        if self.emb is None:
            self.emb = np.load(self.emb_path, mmap_mode="r")
            self.codes = np.load(self.code_path, allow_pickle=False)
            self.model = load_encoder(self.cfg.dense_models[self.prefix])
            log.info("%s: encoding %d new queries with %s", self.system, len(queries), self.cfg.dense_models[self.prefix])
        qe = encode(self.model, queries)
        chunk = 100_000
        for start in range(0, len(queries), batch):
            q = qe[start : start + batch]
            sims = np.empty((len(q), self.emb.shape[0]), dtype=np.float32)
            for lo in range(0, self.emb.shape[0], chunk):
                sims[:, lo : lo + chunk] = q @ np.asarray(self.emb[lo : lo + chunk], dtype=np.float32).T
            for text, row in zip(queries[start : start + batch], sims):
                self._cache[text] = top_codes(row, self.codes, k)

    def retrieve(self, queries: list[str], k: int) -> list[list[tuple[str, float, str]]]:
        pending = [q for q in dict.fromkeys(queries) if q not in self._cache]
        if pending:
            if self.emb_path is not None:
                self._live(pending, k)
            else:
                self.missing += len(pending)
                log.warning("%s: %d queries have no precomputed dense hits", self.system, len(pending))
                for q in pending:
                    self._cache[q] = []
        return [self._cache[q][:k] for q in queries]

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
import time
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
    model = SentenceTransformer(name, device="cuda" if use_gpu else "cpu", token=os.environ.get("HF_TOKEN") or None)
    if not use_gpu and os.environ.get("TERMMAP_INT8_MODELS", "0") == "1":
        model = quantize_model(model)
    return model


def quantize_model(model):
    """Dynamic int8 quantisation of every linear layer (CPU only): about a quarter
    of the weight memory and faster matrix products on CPUs with int8 support."""
    import torch

    return torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


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


def _int8_paths(emb_path: Path) -> tuple[Path, Path]:
    stem = str(emb_path)[: -len(".emb.npy")]
    return Path(stem + ".emb.i8.npy"), Path(stem + ".scale.npy")


def quantize(emb_path: Path, chunk: int = 65_536) -> tuple[Path, Path]:
    """Write an int8 copy of a float16 embedding file with one scale per row.

    Each row x is stored as round(x / s) with s = max|x| / 127, so x is recovered
    as int8 * s. This halves memory against float16; the file is processed in
    chunks so the conversion itself needs little RAM.
    """
    src = np.load(emb_path, mmap_mode="r")
    i8_path, scale_path = _int8_paths(emb_path)
    out = np.lib.format.open_memmap(i8_path, mode="w+", dtype=np.int8, shape=src.shape)
    scales = np.empty(src.shape[0], dtype=np.float32)
    for lo in range(0, src.shape[0], chunk):
        block = np.asarray(src[lo : lo + chunk], dtype=np.float32)
        s = np.abs(block).max(axis=1) / 127.0
        s[s == 0] = 1.0
        out[lo : lo + chunk] = np.clip(np.rint(block / s[:, None]), -127, 127).astype(np.int8)
        scales[lo : lo + chunk] = s
    out.flush()
    del out
    np.save(scale_path, scales)
    return i8_path, scale_path


class DenseRetriever:
    def __init__(self, cfg: Config, system: str, hits: dict | None, emb_path: Path | None, code_path: Path | None,
                 prefix: str = "dense"):
        self.cfg = cfg
        self.system = system
        self.prefix = prefix
        self._cache: dict[str, list] = hits or {}
        self.emb_path, self.code_path = emb_path, code_path
        self.emb = self.codes = self.model = self.scale = None
        self.missing = 0

    @classmethod
    def load(cls, cfg: Config, system: str, prefix: str = "dense") -> "DenseRetriever | None":
        emb_path, code_path, hits_path = _paths(cfg, system, prefix)
        hits = None
        if hits_path.exists() and cfg.load_dense_hits:
            h = pd.read_parquet(hits_path).sort_values(["query", "rank"])
            hits = {q: list(zip(g["code"].astype(str), g["score"].astype(float), [""] * len(g)))
                    for q, g in h.groupby("query", sort=False)}
            log.info("%s: precomputed %s hits for %d queries", system, prefix, len(hits))
        live = emb_path.exists() and code_path.exists()
        if hits is None and not live:
            return None
        return cls(cfg, system, hits, emb_path if live else None, code_path if live else None, prefix)

    def warm(self) -> None:
        """Load the description embeddings and the encoder once and keep them in memory.

        Embeddings stay in RAM as float16 (half the size of float32); reading them
        through a memory map instead re-reads the file from disk on every query
        when memory is tight, which dominated latency for new terms.
        """
        if self.emb is None and self.emb_path is not None:
            t = time.time()
            i8_path, scale_path = _int8_paths(self.emb_path)
            if self.cfg.dense_int8 and i8_path.exists() and scale_path.exists():
                self.emb = np.load(i8_path)
                self.scale = np.load(scale_path)
            else:
                self.emb = np.load(self.emb_path)
                self.scale = None
            self.codes = np.load(self.code_path, allow_pickle=False)
            self.model = load_encoder(self.cfg.dense_models[self.prefix])
            log.info("%s: %s embeddings %s and encoder loaded (%.1fs)", self.system, self.prefix,
                     self.emb.shape, time.time() - t)

    def _live(self, queries: list[str], k: int, batch: int = 256, chunk: int = 65_536):
        self.warm()
        qe = encode(self.model, queries)
        n_keep = k * 3
        for start in range(0, len(queries), batch):
            q = qe[start : start + batch]
            # running top candidates per query, merged chunk by chunk, so the full
            # (queries x descriptions) similarity matrix is never materialised
            best_s = np.full((len(q), 0), -np.inf, dtype=np.float32)
            best_i = np.zeros((len(q), 0), dtype=np.int64)
            for lo in range(0, self.emb.shape[0], chunk):
                block = self.emb[lo : lo + chunk].astype(np.float32)
                s = q @ block.T
                if self.scale is not None:
                    s *= self.scale[lo : lo + chunk]
                m = min(n_keep, s.shape[1])
                part = np.argpartition(-s, m - 1, axis=1)[:, :m]
                best_s = np.concatenate([best_s, np.take_along_axis(s, part, axis=1)], axis=1)
                best_i = np.concatenate([best_i, part + lo], axis=1)
                if best_s.shape[1] > n_keep:
                    keep = np.argpartition(-best_s, n_keep - 1, axis=1)[:, :n_keep]
                    best_s = np.take_along_axis(best_s, keep, axis=1)
                    best_i = np.take_along_axis(best_i, keep, axis=1)
            for text, s_row, i_row in zip(queries[start : start + batch], best_s, best_i):
                order = np.argsort(-s_row, kind="stable")
                seen, out = set(), []
                for j in order:
                    c = str(self.codes[i_row[j]])
                    if c in seen:
                        continue
                    seen.add(c)
                    out.append((c, float(s_row[j]), ""))
                    if len(out) == k:
                        break
                self._cache[text] = out

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

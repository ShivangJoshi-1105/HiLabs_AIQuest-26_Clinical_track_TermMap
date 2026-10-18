"""Fetching model artifacts from a private Hugging Face Hub repository."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from .config import ROOT, Config

log = logging.getLogger(__name__)

# public model weights: fine-tuned cross-encoder and description embeddings
WEIGHTS = ("ce_model/*", "dense_*.emb.npy", "dense_*.codes.npy")
# files derived from the labelled data or the dictionaries; private repositories only
DATA_DERIVED = ("engine.joblib", "terminology_*.parquet", "dense_hits_*.parquet")
DEFAULT_REPO = "explolearner/termmap-weights"


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env reader: KEY=VALUE lines, existing environment wins."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def unpack(cfg: Config, source: Path) -> list[str]:
    """Extract release zip files (termmap_*.zip) from `source` into the artifact directory."""
    import zipfile

    zips = sorted(Path(source).glob("termmap_*.zip"))
    if not zips:
        raise SystemExit(f"no termmap_*.zip files in {source}")
    for z in zips:
        with zipfile.ZipFile(z) as fh:
            fh.extractall(cfg.artifact_dir)
        log.info("unpacked %s", z.name)
    return [z.name for z in zips]


def fetch(cfg: Config, repo_id: str | None = None, embeddings: bool = True) -> Path:
    """Download weights (and, from a private team repository, data-derived files)."""
    load_dotenv()
    repo_id = repo_id or os.environ.get("TERMMAP_ARTIFACTS_REPO") or DEFAULT_REPO
    from huggingface_hub import snapshot_download

    patterns = ["ce_model/*"] + (["dense_*.emb.npy", "dense_*.codes.npy"] if embeddings else []) + list(DATA_DERIVED)
    path = snapshot_download(repo_id=repo_id, repo_type="model", local_dir=str(cfg.artifact_dir),
                             allow_patterns=patterns, token=os.environ.get("HF_TOKEN") or None)
    log.info("artifacts from %s in %s", repo_id, path)
    return Path(path)

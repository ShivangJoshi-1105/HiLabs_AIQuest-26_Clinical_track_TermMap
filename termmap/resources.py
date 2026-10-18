"""Loading and caching of the code dictionaries and the labelled data."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import Config
from .normalize import clean, strip_semantic_tag

log = logging.getLogger(__name__)

# SNOMED semantic tags seen on fewer codes than this are treated as part of the
# description text rather than as a tag (e.g. "Pain (finding) (left)").
_MIN_TAG_CODES = 40


@dataclass
class Terminology:
    """One coding system: a description table and a code table."""

    system: str
    descriptions: pd.DataFrame  # code, text, norm, order
    codes: pd.DataFrame  # indexed by code: title, plus system-specific columns

    def title(self, code: str) -> str:
        try:
            return self.codes.at[code, "title"]
        except KeyError:
            return ""

    def __contains__(self, code: str) -> bool:
        return code in self.codes.index


def _icd_codes(desc: pd.DataFrame) -> pd.DataFrame:
    codes = desc.groupby("code", sort=True).agg(title=("text", "first"), n_desc=("text", "size"))
    keys = set(codes.index)
    prefixes = set()
    for c in keys:
        for k in range(3, len(c)):
            prefixes.add(c[:k])
    codes["is_leaf"] = [c not in prefixes for c in codes.index]
    codes["depth"] = [len(c.replace(".", "")) for c in codes.index]
    codes["chapter"] = [c[0] for c in codes.index]
    codes["block"] = [c[:3] for c in codes.index]
    t = codes["title"].str.lower()
    codes["unspecified"] = t.str.contains("unspecified")
    codes["other"] = t.str.contains(r"\bother\b")
    return codes


def _snomed_codes(desc: pd.DataFrame) -> pd.DataFrame:
    tags = desc.dropna(subset=["tag"]).groupby("code")["tag"].agg(lambda s: s.value_counts().index[0])
    counts = tags.value_counts()
    valid = set(counts[counts >= _MIN_TAG_CODES].index)
    first = desc.sort_values("order").groupby("code", sort=True).agg(
        title=("text", "first"), n_desc=("text", "size")
    )
    first["tag"] = tags.where(tags.isin(valid)).reindex(first.index).fillna("other")
    return first


def _load_icd(cfg: Config) -> Terminology:
    raw = pd.read_excel(cfg.raw_dir / cfg.icd_file, dtype=str)
    raw.columns = ["code", "text"]
    raw = raw.dropna()
    raw["code"] = raw["code"].str.strip()
    raw = raw[~raw["code"].str.contains("-")]  # chapter ranges are not assignable codes
    raw["text"] = raw["text"].str.strip()
    raw["order"] = raw.groupby("code").cumcount() + 1
    raw["norm"] = raw["text"].map(clean)
    raw = raw.drop_duplicates(["code", "norm"]).reset_index(drop=True)
    return Terminology("ICD10CM", raw[["code", "text", "norm", "order"]], _icd_codes(raw))


def _load_snomed(cfg: Config) -> Terminology:
    raw = pd.read_excel(cfg.raw_dir / cfg.snomed_file, sheet_name="SNOMED codes")
    raw.columns = ["code", "text", "order"]
    raw = raw.dropna(subset=["text"])
    raw["code"] = raw["code"].astype("int64").astype(str)
    split = raw["text"].map(strip_semantic_tag)
    raw["tag"] = [t for _, t in split]
    raw["norm"] = [clean(s) for s, _ in split]
    raw = raw[raw["norm"].str.len() > 0]
    codes = _snomed_codes(raw)
    desc = raw.drop_duplicates(["code", "norm"]).reset_index(drop=True)
    return Terminology("SNOMEDCT_US", desc[["code", "text", "norm", "order"]], codes)


def load_terminology(cfg: Config, system: str) -> Terminology:
    cache = Path(cfg.artifact_dir) / f"terminology_{system}"
    dpath, cpath = cache.with_suffix(".desc.parquet"), cache.with_suffix(".codes.parquet")
    if dpath.exists() and cpath.exists():
        return Terminology(system, pd.read_parquet(dpath), pd.read_parquet(cpath))
    log.info("parsing %s dictionary", system)
    term = _load_icd(cfg) if system == "ICD10CM" else _load_snomed(cfg)
    term.descriptions.to_parquet(dpath)
    term.codes.to_parquet(cpath)
    return term


def read_terms(path: str | Path) -> pd.DataFrame:
    """Read an input file of terms. Accepts the training schema
    (entity, chunk_text, coding_system, code) and the evaluation schema
    (chunk_id, clinical_entity)."""
    path = Path(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False) if path.suffix != ".xlsx" else pd.read_excel(path, dtype=str).fillna("")
    rename = {"clinical_entity": "entity", "chunk": "chunk_text", "context": "chunk_text"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns and v not in df.columns})
    if "entity" not in df.columns:
        raise ValueError(f"{path} has no entity / clinical_entity column")
    if "chunk_text" not in df.columns:
        df["chunk_text"] = ""
    return df

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    raw_dir: Path = ROOT / "data" / "raw"
    resource_dir: Path = ROOT / "data" / "resources"
    artifact_dir: Path = ROOT / "artifacts"
    output_dir: Path = ROOT / "outputs"

    train_file: str = "hackathon_train.csv"
    icd_file: str = "icd10cm_codes.xlsx"
    snomed_file: str = "SNOMEDCT_US_codes_and_descriptions.xlsx"

    # candidate generation
    k_char: int = 40
    k_word: int = 40
    k_alias: int = 20
    max_candidates: int = 60
    query_batch: int = 64

    # reranker
    n_folds: int = 5
    seed: int = 13
    shared_ranker: bool = True
    ranker_params: dict = field(default_factory=lambda: {
        "n_estimators": 600,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 30,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    })
    n_jobs: int = 2
    top_n_keep: int = 10

    # dense retrieval (optional, used when the embedding cache exists)
    # artifact prefix -> encoder used for live query encoding
    dense_models: dict = field(default_factory=lambda: {
        "dense": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "dense_biolord": "FremyCompany/BioLORD-2023",
    })
    k_dense: int = 40
    use_dense: bool = True
    # feature source name -> artifact file prefix (dense_hits_<system>.parquet etc.)
    dense_sources: dict = field(default_factory=lambda: {"dense": "dense", "dense_bl": "dense_biolord"})

    # second-stage scorers (used when score files exist in artifact_dir)
    use_stage2: bool = True
    # score missing cross-encoder requests with artifacts/ce_model at prediction time;
    # held-out evaluation turns this off so a model trained on all data cannot leak
    live_stage2: bool = True
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    stage2_top_n: int = 10
    llm_examples: int = 6
    # log-linear blend weights per system and scorer, chosen on the held-out split
    stage2_weights: dict = field(default_factory=lambda: {
        "ICD10CM": {"ce": 0.5},
        "SNOMEDCT_US": {"ce": 0.5},
    })

    # weight of the coding-team crosswalk prior per target system (0 disables)
    coupling_weights: dict = field(default_factory=lambda: {"ICD10CM": 0.0, "SNOMEDCT_US": 0.0})
    # add gold descriptions absent from the dictionary as aliases (lowered held-out accuracy)
    display_aliases: bool = False

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        cfg = cls()
        if path:
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            names = {f.name for f in fields(cls)}
            for key, value in data.items():
                if key not in names:
                    raise KeyError(f"unknown config key: {key}")
                if key.endswith("_dir"):
                    value = (ROOT / value) if not Path(value).is_absolute() else Path(value)
                setattr(cfg, key, value)
        for d in (cfg.artifact_dir, cfg.output_dir):
            Path(d).mkdir(parents=True, exist_ok=True)
        return cfg

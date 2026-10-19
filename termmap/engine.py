"""The mapping engine: retrieval, out-of-fold reranker training and prediction."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from dataclasses import dataclass, field

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from . import SYSTEMS, coupling, stage2
from .coupling import Crosswalk
from .candidates import AliasRetriever, DictionaryRetriever, generate, icd_unspecified_children
from .config import Config
from .features import N_SUBCONCEPT_FEATURES, PriorStats, featurize
from .normalize import AbbreviationExpander, clean
from .resources import Terminology, load_terminology

log = logging.getLogger(__name__)


def subconcepts(df: pd.DataFrame) -> list[str]:
    if "subconcept" not in df.columns:
        return [""] * len(df)
    return [s if isinstance(s, str) else "" for s in df["subconcept"]]


class Ranker:
    """LambdaRank candidate scorer (LightGBM).

    Candidates of one term form one query group, so the model learns to order
    candidates against each other rather than to score them in isolation.
    Two models are kept: one that uses the subconcept features (for inputs
    that carry a subconcept column) and one trained without them.
    """

    def __init__(self, cfg: Config):
        params = dict(random_state=cfg.seed, deterministic=True, force_row_wise=True,
                      n_jobs=cfg.n_jobs, verbose=-1, **cfg.ranker_params)
        self.full = lgb.LGBMRanker(**params)
        self.base = lgb.LGBMRanker(**params)

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> "Ranker":
        self.full.fit(X, y, group=groups)
        self.base.fit(X[:, :-N_SUBCONCEPT_FEATURES], y, group=groups)
        return self

    def raw(self, X: np.ndarray, has_subconcept: bool) -> np.ndarray:
        if has_subconcept:
            return self.full.predict(X)
        return self.base.predict(X[:, :-N_SUBCONCEPT_FEATURES])

    def score(self, X: np.ndarray, has_subconcept: bool) -> np.ndarray:
        """Softmax over one term's candidates: a probability-like confidence."""
        r = self.raw(X, has_subconcept)
        e = np.exp(r - r.max())
        return e / e.sum()


@dataclass
class Prediction:
    code: str
    confidence: float
    margin: float
    in_candidates: bool | None = None
    ranked: list = field(default_factory=list)  # top candidates as (code, score)


class SystemMapper:
    """Maps terms to one coding system."""

    def __init__(self, cfg: Config, term: Terminology, expander: AbbreviationExpander, dense: dict | None = None):
        self.cfg = cfg
        self.term = term
        self.expander = expander
        self.dense = dense or {}  # source name -> DenseRetriever
        t = time.time()
        self.retriever = DictionaryRetriever(term)
        log.info("%s: dictionary index over %d descriptions (%.0fs)", term.system, len(term.descriptions), time.time() - t)
        self.parent_children = icd_unspecified_children(term) if term.system == "ICD10CM" else None
        self.model: HistGradientBoostingClassifier | None = None
        self.aliases: pd.DataFrame | None = None

    # -- candidate and feature construction ---------------------------------

    def _variants(self, df: pd.DataFrame) -> list[list[str]]:
        return [self.expander.variants(e, c) for e, c in zip(df["entity"], df["chunk_text"])]

    def _candidates(self, df: pd.DataFrame, alias: AliasRetriever | None):
        variants = self._variants(df)
        flat = [q for qs in variants for q in qs]
        dict_hits = self.retriever.retrieve(flat, self.cfg.k_char, self.cfg.query_batch)
        alias_hits = alias.retrieve(flat, self.cfg.k_alias) if alias else None
        extra = {name: r.retrieve(flat, self.cfg.k_dense) for name, r in self.dense.items()} or None
        cands = generate(variants, dict_hits, alias_hits, extra, self.cfg.max_candidates, self.parent_children)
        return variants, cands

    def _features(self, df, variants, cands, prior, alias):
        blocks = []
        rows = zip(df["entity"], df["chunk_text"], subconcepts(df))
        for (entity, ctx, sub), vs, cs in zip(rows, variants, cands):
            exact = set()
            if alias:
                for v in vs:
                    exact |= alias.exact.get(v, set())
            if not cs:
                blocks.append(([], np.zeros((0, 0), dtype=np.float32)))
                continue
            # an abbreviation resolved by its context is compared under its reading
            term_text = vs[0] if len(vs) == 1 and vs[0] != clean(entity) else entity
            blocks.append(featurize(term_text, vs, ctx, cs, self.term, prior, exact, sub))
        return blocks

    def _fold_parts(self, train: pd.DataFrame):
        aliases = pd.DataFrame({"norm": train["entity"].map(clean), "code": train["code"]})
        if self.cfg.display_aliases and "display" in train.columns:
            # gold descriptions the coding team used that the supplied dictionary lacks
            # (other releases, local wording) are added as extra synonyms
            if not hasattr(self, "_desc_pairs"):
                d = self.term.descriptions
                self._desc_pairs = set(zip(d["code"], d["norm"]))
            disp = pd.DataFrame({"norm": train["display"].map(clean), "code": train["code"]})
            disp = disp[[(c, n) not in self._desc_pairs for c, n in zip(disp["code"], disp["norm"])]]
            aliases = pd.concat([aliases, disp], ignore_index=True)
        aliases = aliases[aliases["norm"].str.len() > 0]
        alias = AliasRetriever(aliases) if len(aliases) else None
        prior = PriorStats(train["code"].tolist(), self.term, subconcepts(train))
        return alias, prior, aliases

    # -- training -----------------------------------------------------------

    def fit(self, train: pd.DataFrame) -> "SystemMapper":
        X, y = self.out_of_fold(train)
        self.model = Ranker(self.cfg).fit(X, y, self.oof_groups)
        return self

    def out_of_fold(self, train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Build reranker training data from out-of-fold candidates, then fit
        the alias index and priors on the full training set."""
        train = train.reset_index(drop=True)
        X, y = [], []
        hit = 0
        # kept for building second-stage training data: (row, codes, features, fold alias index)
        self.oof_records = []
        kf = KFold(self.cfg.n_folds, shuffle=True, random_state=self.cfg.seed)
        for k, (tr_idx, va_idx) in enumerate(kf.split(train)):
            tr, va = train.iloc[tr_idx], train.iloc[va_idx]
            alias, prior, _ = self._fold_parts(tr)
            variants, cands = self._candidates(va, alias)
            blocks = self._features(va, variants, cands, prior, alias)
            for (_, row), (codes, feats) in zip(va.iterrows(), blocks):
                if not codes:
                    continue
                labels = np.array([c == row["code"] for c in codes], dtype=np.int8)
                hit += labels.any()
                X.append(feats)
                y.append(labels)
                self.oof_records.append((row, codes, feats, alias))
            log.info("%s: fold %d/%d done", self.term.system, k + 1, self.cfg.n_folds)
        log.info("%s: out-of-fold candidate recall %.4f", self.term.system, hit / len(train))
        self.oof_recall = hit / len(train)
        self.alias, self.prior, self.aliases = self._fold_parts(train)
        self.gold, self.gold_subs = train["code"].tolist(), subconcepts(train)
        self.oof_groups = np.array([len(x) for x in X])
        return np.vstack(X), np.concatenate(y)

    def oof_ranked(self) -> list[tuple[pd.Series, Prediction, AliasRetriever | None]]:
        """Out-of-fold candidate lists ordered by the fitted ranker."""
        out = []
        for row, codes, feats, alias in self.oof_records:
            sub = row.get("subconcept", "")
            p = self.model.score(feats, isinstance(sub, str) and bool(sub))
            order = np.argsort(-p, kind="stable")
            ranked = [(codes[j], float(p[j])) for j in order[: self.cfg.top_n_keep]]
            out.append((row, Prediction(ranked[0][0], ranked[0][1], 0.0, None, ranked), alias))
        return out

    # -- inference ----------------------------------------------------------

    def predict(self, df: pd.DataFrame, gold: pd.Series | None = None) -> list[Prediction]:
        df = df.reset_index(drop=True)
        variants, cands = self._candidates(df, self.alias)
        blocks = self._features(df, variants, cands, self.prior, self.alias)
        subs = subconcepts(df)
        out = []
        for i, (codes, feats) in enumerate(blocks):
            if not codes:
                out.append(Prediction("", 0.0, 0.0, False if gold is not None else None))
                continue
            p = self.model.score(feats, bool(subs[i]))
            order = np.argsort(-p, kind="stable")
            top = order[0]
            margin = float(p[top] - p[order[1]]) if len(order) > 1 else float(p[top])
            inc = (gold.iloc[i] in codes) if gold is not None else None
            ranked = [(codes[j], float(p[j])) for j in order[: self.cfg.top_n_keep]]
            out.append(Prediction(codes[top], float(p[top]), margin, inc, ranked))
        return out


class TermMapEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.expander = AbbreviationExpander.from_file(cfg.resource_dir / "abbreviations.tsv")
        self.mappers: dict[str, SystemMapper] = {}

    def _dense(self, system: str) -> dict:
        if not self.cfg.use_dense:
            return {}
        from .dense import DenseRetriever

        out = {}
        for name, prefix in self.cfg.dense_sources.items():
            r = DenseRetriever.load(self.cfg, system, prefix)
            if r is not None:
                out[name] = r
        return out

    def mapper(self, system: str) -> SystemMapper:
        if system not in self.mappers:
            term = load_terminology(self.cfg, system)
            self.mappers[system] = SystemMapper(self.cfg, term, self.expander, self._dense(system))
        return self.mappers[system]

    def fit(self, train: pd.DataFrame) -> "TermMapEngine":
        self.crosswalk = Crosswalk(train)
        data = {}
        for system in SYSTEMS:
            part = train[train["coding_system"] == system]
            data[system] = self.mapper(system).out_of_fold(part)
        if self.cfg.shared_ranker:
            # most features mean the same thing in both systems; pooling gives the
            # smaller ICD-10-CM set the benefit of the SNOMED examples
            X = np.vstack([data[s][0] for s in SYSTEMS])
            y = np.concatenate([data[s][1] for s in SYSTEMS])
            groups = np.concatenate([self.mappers[s].oof_groups for s in SYSTEMS])
            model = Ranker(self.cfg).fit(X, y, groups)
            for system in SYSTEMS:
                self.mappers[system].model = model
        else:
            for system in SYSTEMS:
                self.mappers[system].model = Ranker(self.cfg).fit(*data[system], self.mappers[system].oof_groups)
        return self

    # -- second stage: cross-encoder and LLM scores ------------------------

    @property
    def stage2_scores(self) -> dict:
        if not hasattr(self, "_stage2_scores"):
            self._stage2_scores = stage2.load_scores(self.cfg.artifact_dir) if self.cfg.use_stage2 else {}
        return self._stage2_scores

    def builder(self, system: str) -> stage2.RequestBuilder:
        # built once per system: it indexes every description of the terminology
        if not hasattr(self, "_builders"):
            self._builders = {}
        if system not in self._builders:
            self._builders[system] = stage2.RequestBuilder(self.mapper(system), self.cfg.stage2_top_n, self.cfg.llm_examples)
        return self._builders[system]

    def stage2_requests(self, system: str, df: pd.DataFrame, preds: list[Prediction]) -> dict[str, list[dict]]:
        rows = [r for _, r in df.reset_index(drop=True).iterrows()]
        b = self.builder(system)
        return {"llm": b.llm_requests(rows, preds, self.mapper(system).alias),
                "ce": b.ce_requests(rows, preds)}

    def warm(self) -> None:
        """Load every model the prediction path can use, so the first request is not slower than the rest."""
        for system in SYSTEMS:
            for retriever in self.mapper(system).dense.values():
                retriever.warm()
            self.builder(system)
        model_dir = Path(self.cfg.artifact_dir) / "ce_model"
        if self.cfg.live_stage2 and (model_dir / "config.json").exists() and not hasattr(self, "_ce_scorer"):
            self._ce_scorer = stage2.CrossEncoderScorer(model_dir)

    def _score_ce_live(self, system: str, df: pd.DataFrame, preds: list[Prediction]) -> None:
        """Score requests that have no precomputed cross-encoder scores, when the
        fine-tuned model is available in artifacts/ce_model."""
        model_dir = Path(self.cfg.artifact_dir) / "ce_model"
        if not self.cfg.live_stage2 or not (model_dir / "config.json").exists():
            return
        scores = self.stage2_scores.setdefault("ce", {})
        rows = [r for _, r in df.reset_index(drop=True).iterrows()]
        reqs = [r for r in self.builder(system).ce_requests(rows, preds) if r["id"] not in scores]
        if not reqs:
            return
        if not hasattr(self, "_ce_scorer"):
            self._ce_scorer = stage2.CrossEncoderScorer(model_dir)
        t = time.time()
        scores.update(self._ce_scorer.score(reqs))
        log.info("%s: cross-encoder scored %d terms on %s (%.1fs)", system, len(reqs), self._ce_scorer.device,
                 time.time() - t)

    def stage2_training(self, system: str) -> dict[str, list[dict]]:
        """Fine-tuning data for the cross-encoder and the LoRA model, built from
        out-of-fold candidate lists so no example sees its own label."""
        records = self.mapper(system).oof_ranked()
        b = self.builder(system)
        return {"llm": b.llm_training(records, self.cfg.seed), "ce": b.ce_training(records)}

    def adjudicate(self, system: str, df: pd.DataFrame, preds: list[Prediction],
                   weights: dict | None = None) -> list[Prediction]:
        """Blend second-stage scores into predictions that have them; others pass through."""
        weights = self.cfg.stage2_weights.get(system, {}) if weights is None else weights
        if weights.get("ce"):
            self._score_ce_live(system, df, preds)
        scores = self.stage2_scores
        active = [n for n, w in weights.items() if w and n in scores]
        if not active:
            return preds
        out = []
        for (entity, ctx), p in zip(zip(df["entity"], df["chunk_text"]), preds):
            codes = [c for c, _ in p.ranked[: self.cfg.stage2_top_n]]
            rid = stage2.request_id(system, entity, ctx or "", codes)
            extra = {n: scores[n][rid] for n in active if rid in scores[n] and len(scores[n][rid]) == len(codes)}
            if not extra:
                out.append(p)
                continue
            ranked = stage2.blend(p.ranked, extra, weights)
            margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else ranked[0][1]
            out.append(Prediction(ranked[0][0], ranked[0][1], margin, p.in_candidates, ranked))
        return out

    # -- coupling through the coding team's own ICD/SNOMED pairings ---------

    def couple(self, system: str, preds: list[Prediction], other: list[Prediction],
               weight: float | None = None) -> list[Prediction]:
        weight = self.cfg.coupling_weights.get(system, 0.0) if weight is None else weight
        cw = getattr(self, "crosswalk", None)
        if not weight or cw is None:
            return preds
        out = []
        for p, o in zip(preds, other):
            cands = [c for c, _ in p.ranked]
            q = cw.prior(system, cands, o.ranked[:5]) if cands and o.ranked else None
            if q is None:
                out.append(p)
                continue
            ranked = coupling.rescore(p.ranked, q, weight)
            margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else ranked[0][1]
            out.append(Prediction(ranked[0][0], ranked[0][1], margin, p.in_candidates, ranked))
        return out

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        df = df.reset_index(drop=True)
        first = {s: self.adjudicate(s, df, self.mapper(s).predict(df)) for s in SYSTEMS}
        for system, col in zip(SYSTEMS, ("icd10cm_code", "snomed_code")):
            other = SYSTEMS[1] if system == SYSTEMS[0] else SYSTEMS[0]
            preds = self.couple(system, first[system], first[other])
            out[col] = [p.code for p in preds]
            out[f"{col}_confidence"] = [round(p.confidence, 4) for p in preds]
        if "coding_system" in out.columns:
            out["predicted_code"] = np.where(out["coding_system"] == "ICD10CM", out["icd10cm_code"], out["snomed_code"])
        return out

    # -- persistence --------------------------------------------------------

    def save(self, path):
        state = {s: {"model": m.model, "aliases": m.aliases, "gold": m.gold, "subconcepts": m.gold_subs}
                 for s, m in self.mappers.items()}
        joblib.dump({"systems": state, "crosswalk": getattr(self, "crosswalk", None)}, path)

    @classmethod
    def load(cls, cfg: Config, path) -> "TermMapEngine":
        eng = cls(cfg)
        saved = joblib.load(path)
        eng.crosswalk = saved["crosswalk"]
        for system, s in saved["systems"].items():
            m = eng.mapper(system)
            m.model = s["model"]
            m.aliases = s["aliases"]
            m.alias = AliasRetriever(s["aliases"]) if len(s["aliases"]) else None
            m.gold, m.gold_subs = s["gold"], s["subconcepts"]
            m.prior = PriorStats(m.gold, m.term, m.gold_subs)
        return eng

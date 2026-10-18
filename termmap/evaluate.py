"""Held-out evaluation on the labelled data."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import SYSTEMS
from .config import Config
from .engine import TermMapEngine
from .stage2 import weight_grid, write_jsonl

log = logging.getLogger(__name__)


def split(df: pd.DataFrame, frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    mask = rng.rand(len(df)) < frac
    return df[mask].reset_index(drop=True), df[~mask].reset_index(drop=True)


def score(system: str, test: pd.DataFrame, preds, train_codes: set[str]) -> tuple[dict, pd.DataFrame]:
    gold = test["code"].tolist()
    pred = [p.code for p in preds]
    rows = test[["entity", "chunk_text", "code"]].copy()
    rows["pred"] = pred
    rows["confidence"] = [p.confidence for p in preds]
    rows["margin"] = [p.margin for p in preds]
    rows["in_candidates"] = [p.in_candidates for p in preds]
    rows["correct"] = rows["pred"] == rows["code"]
    rows["code_seen"] = rows["code"].isin(train_codes)
    m = {
        "n": len(rows),
        "accuracy": rows["correct"].mean(),
        "candidate_recall": rows["in_candidates"].mean(),
        "accuracy_given_recall": rows.loc[rows["in_candidates"], "correct"].mean(),
        "accuracy_seen_code": rows.loc[rows["code_seen"], "correct"].mean(),
        "accuracy_unseen_code": rows.loc[~rows["code_seen"], "correct"].mean(),
        "share_unseen_code": (~rows["code_seen"]).mean(),
    }
    if system == "ICD10CM":
        m["accuracy_3char_block"] = np.mean([g[:3] == p[:3] for g, p in zip(gold, pred)])
    # accuracy if the least confident 5% of terms are abstained on
    cut = rows["confidence"].quantile(0.05)
    m["accuracy_at_95pct_coverage"] = rows.loc[rows["confidence"] > cut, "correct"].mean()
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in m.items()}, rows


def stage2_metrics(engine, system: str, te: pd.DataFrame, preds) -> dict:
    """Accuracy of each second-stage scorer and of the best blend.

    The blend weights are searched on a grid. The cross-fitted figure picks the
    weights on one half of the terms and scores them on the other half, so it is
    not tuned on itself.
    """
    names = sorted(engine.stage2_scores)
    gold = te["code"].to_numpy()

    def correct(weights):
        out = engine.adjudicate(system, te, preds, weights=weights)
        return np.array([p.code == g for p, g in zip(out, gold)])

    m = {}
    for n in names:
        m[f"accuracy_{n}_only"] = round(float(correct({n: 1000.0}).mean()), 4)
        m[f"accuracy_ranker_plus_{n}"] = round(float(correct({n: 1.0}).mean()), 4)
    results = [(w, correct(w)) for w in weight_grid(names)]
    half = np.arange(len(te)) % 2 == 0

    def pick(mask):
        # ties go to the smaller total weight
        return max(results, key=lambda r: (r[1][mask].mean(), -sum(r[0].values())))

    cross = [pick(a)[1][b] for a, b in ((half, ~half), (~half, half))]
    best_w, best_c = pick(np.ones(len(te), dtype=bool))
    m["stage2_best_weights"] = best_w
    m["stage2_best_accuracy"] = round(float(best_c.mean()), 4)
    m["stage2_crossfit_accuracy"] = round(float(np.concatenate(cross).mean()), 4)
    return m


COUPLING_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)


def coupling_metrics(engine, system: str, te: pd.DataFrame, preds, other_preds):
    """Effect of the crosswalk prior, and agreement between the two systems'
    independent predictions as a confidence signal."""
    gold = te["code"].to_numpy()
    results = {}
    for w in COUPLING_GRID:
        out = engine.couple(system, preds, other_preds, weight=w)
        results[w] = (out, np.array([p.code == g for p, g in zip(out, gold)]))
    half = np.arange(len(te)) % 2 == 0

    def pick(mask):
        return max(COUPLING_GRID, key=lambda w: (results[w][1][mask].mean(), -w))

    cross = np.concatenate([results[pick(a)][1][b] for a, b in ((half, ~half), (~half, half))])

    # agreement: was the pair (this system's top-1, other system's top-1) seen in the crosswalk?
    cw = engine.crosswalk
    base = results[0.0][1]
    evidence, agree = [], []
    for p, o in zip(preds, other_preds):
        c = cw.by_snomed.get(o.code) if system == "ICD10CM" else cw.by_icd.get(o.code)
        evidence.append(bool(c))
        agree.append(bool(c) and p.code in c)
    evidence, agree = np.array(evidence), np.array(agree)
    disagree = evidence & ~agree

    def acc(mask):
        return round(float(base[mask].mean()), 4) if mask.any() else None

    m = {
        "coupling_accuracy_by_weight": {str(w): round(float(results[w][1].mean()), 4) for w in COUPLING_GRID},
        "coupling_best_weight": pick(np.ones(len(te), dtype=bool)),
        "coupling_crossfit_accuracy": round(float(cross.mean()), 4),
        "pair_evidence_share": round(float(evidence.mean()), 4),
        "share_pair_agrees": round(float(agree.mean()), 4),
        "accuracy_when_pair_agrees": acc(agree),
        "accuracy_when_pair_disagrees": acc(disagree),
        "accuracy_no_pair_evidence": acc(~evidence),
    }
    coupled = engine.couple(system, preds, other_preds)
    return m, coupled


def run(cfg: Config, train_df: pd.DataFrame, frac: float = 0.8, out_dir: Path | None = None) -> dict:
    out_dir = Path(out_dir or cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.live_stage2 = False
    dev_train, dev_test = split(train_df, frac, cfg.seed)
    log.info("dev split: %d train / %d test", len(dev_train), len(dev_test))
    engine = TermMapEngine(cfg).fit(dev_train)
    report, finals = {}, {}
    files = {"llm_requests_dev": [], "ce_requests_dev": [], "llm_train_dev": [], "ce_train_dev": []}
    for system in SYSTEMS:
        tr = dev_train[dev_train["coding_system"] == system]
        te = dev_test[dev_test["coding_system"] == system].reset_index(drop=True)
        mapper = engine.mapper(system)
        preds = mapper.predict(te, gold=te["code"])

        reqs = engine.stage2_requests(system, te, preds)
        files["llm_requests_dev"] += reqs["llm"]
        files["ce_requests_dev"] += reqs["ce"]
        train_sets = engine.stage2_training(system)
        files["llm_train_dev"] += train_sets["llm"]
        files["ce_train_dev"] += train_sets["ce"]

        metrics, rows = score(system, te, preds, set(tr["code"]))
        metrics["oof_candidate_recall"] = round(mapper.oof_recall, 4)
        for k in (5, cfg.stage2_top_n):
            metrics[f"top{k}_recall"] = round(float(np.mean(
                [g in [c for c, _ in p.ranked[:k]] for p, g in zip(preds, te["code"])])), 4)
        if "subconcept" in te.columns:
            bare = mapper.predict(te.drop(columns=["subconcept"]), gold=te["code"])
            metrics["accuracy_without_subconcept"] = round(
                float(np.mean([p.code == g for p, g in zip(bare, te["code"])])), 4)
        final = preds
        if engine.stage2_scores:
            metrics.update(stage2_metrics(engine, system, te, preds))
            final = engine.adjudicate(system, te, preds)
            metrics["accuracy_final"] = round(float(np.mean([p.code == g for p, g in zip(final, te["code"])])), 4)
            rows["pred_final"] = [p.code for p in final]
        finals[system] = (te, final, rows)
        report[system] = metrics
        log.info("%s: %s", system, metrics)

    for system in SYSTEMS:
        te, final, rows = finals[system]
        other = SYSTEMS[1] if system == SYSTEMS[0] else SYSTEMS[0]
        other_preds = engine.adjudicate(other, te, engine.mapper(other).predict(te))
        cm, coupled = coupling_metrics(engine, system, te, final, other_preds)
        report[system].update(cm)
        rows["pred_coupled"] = [p.code for p in coupled]
        rows["other_system_pred"] = [p.code for p in other_preds]
        rows.to_csv(out_dir / f"eval_{system}.csv", index=False)
        log.info("%s coupling: %s", system, cm)

    for name, items in files.items():
        write_jsonl(items, out_dir / f"{name}.jsonl")
    report["headline_accuracy"] = min(report[s]["accuracy"] for s in SYSTEMS)
    if all("accuracy_final" in report[s] for s in SYSTEMS):
        report["headline_accuracy_final"] = min(report[s]["accuracy_final"] for s in SYSTEMS)
    if all("accuracy_without_subconcept" in report[s] for s in SYSTEMS):
        report["headline_accuracy_without_subconcept"] = min(
            report[s]["accuracy_without_subconcept"] for s in SYSTEMS)
    with open(out_dir / "eval_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return report

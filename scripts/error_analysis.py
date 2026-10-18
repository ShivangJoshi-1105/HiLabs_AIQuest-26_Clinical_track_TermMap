"""Failure-mode breakdown of a held-out evaluation run.

    python scripts/error_analysis.py outputs/eval_<tag>

Reads eval_<system>.csv from the run directory and writes error_analysis.md
next to it: every error is assigned to the first matching failure mode below.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from termmap.config import Config  # noqa: E402
from termmap.normalize import AbbreviationExpander, clean  # noqa: E402
from termmap.resources import load_terminology  # noqa: E402

MODES = {
    "retrieval_miss": "Gold code never retrieved into the candidate set",
    "specificity": "Right ICD-10-CM category (3-character block), wrong specificity",
    "hierarchy": "Right words, wrong SNOMED CT hierarchy (e.g. finding vs observable entity vs procedure)",
    "abbreviation": "Term is or contains a short abbreviation / shorthand",
    "noise": "Term has OCR / typing noise (token not found in any description)",
    "multi_concept": "Term bundles more than one clinical idea",
    "convention": "Gold label is far from the term's wording (local coding convention)",
    "other": "Other near-miss between close candidates",
}


COMMON = set("her his him she he they them their its our your my me we you pt yo y o per due "
             "also very much more most some any each such than then into onto upon over".split())


def is_typo(tok: str, vocab: set[str]) -> bool:
    """A token absent from every description but one or two edits from a word that is present,
    or two dictionary words run together."""
    if len(tok) < 5 or not tok.isalpha() or tok in vocab:
        return False
    if any(tok[:i] in vocab and tok[i:] in vocab for i in range(3, len(tok) - 2)):
        return True
    match = process.extractOne(tok, vocab, scorer=Levenshtein.distance, score_cutoff=2)
    return match is not None


def classify(row, term, vocab, abbrev) -> str:
    if row["in_candidates"] in (False, "False"):
        return "retrieval_miss"
    gold, pred = row["code"], row["final"]
    if term.system == "ICD10CM" and gold[:3] == pred[:3]:
        return "specificity"
    if term.system == "SNOMEDCT_US":
        gt, pt = term.codes.at[gold, "tag"] if gold in term.codes.index else "", term.codes.at[pred, "tag"] if pred in term.codes.index else ""
        gtitle, ptitle = clean(term.title(gold)), clean(term.title(pred))
        if gt != pt and (set(gtitle.split()) & set(ptitle.split())):
            return "hierarchy"
    toks = clean(row["entity"]).split()
    if any(t in abbrev or (len(t) <= 4 and t.isalpha() and t not in vocab and t not in COMMON) for t in toks):
        return "abbreviation"
    if any(is_typo(t, vocab) for t in toks):
        return "noise"
    if len(toks) >= 6 or re.search(r"\b(and|with|w)\b", clean(row["entity"])) and len(toks) >= 4:
        return "multi_concept"
    g = set(clean(term.title(gold)).split())
    if not (g & set(toks)):
        return "convention"
    return "other"


def main(run_dir: str) -> None:
    run_dir = Path(run_dir)
    cfg = Config.load()
    abbrev = set(AbbreviationExpander.from_file(cfg.resource_dir / "abbreviations.tsv").table)
    lines = [f"# Error analysis: {run_dir.name}", ""]
    for system in ("ICD10CM", "SNOMEDCT_US"):
        path = run_dir / f"eval_{system}.csv"
        if not path.exists():
            continue
        term = load_terminology(cfg, system)
        vocab = {t for d in term.descriptions["norm"] for t in d.split()}
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        final_col = next(c for c in ("pred_final", "pred") if c in df.columns)
        df["final"] = df[final_col]
        wrong = df[df["final"] != df["code"]].copy()
        wrong["mode"] = [classify(r, term, vocab, abbrev) for _, r in wrong.iterrows()]
        acc = 1 - len(wrong) / len(df)
        lines += [f"## {system}", "", f"{len(df)} terms, accuracy {acc:.1%} (column `{final_col}`), {len(wrong)} errors.", "",
                  "| Failure mode | Errors | Share of errors |", "|---|---|---|"]
        counts = wrong["mode"].value_counts()
        for mode in MODES:
            n = int(counts.get(mode, 0))
            lines.append(f"| {MODES[mode]} | {n} | {n / max(len(wrong), 1):.0%} |")
        lines.append("")
        for mode in MODES:
            ex = wrong[wrong["mode"] == mode].head(4)
            if ex.empty:
                continue
            lines.append(f"**{MODES[mode]}**")
            lines.append("")
            for _, r in ex.iterrows():
                lines.append(f"- `{r['entity'][:60]}`: gold {r['code']} *{term.title(r['code'])[:70]}*, "
                             f"predicted {r['final']} *{term.title(r['final'])[:70]}*")
            lines.append("")
    out = run_dir / "error_analysis.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1])

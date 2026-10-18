"""Feature extraction for (term, candidate code) pairs."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
from rapidfuzz import fuzz

from .candidates import ABSENT_RANK, Candidate
from .normalize import STOPWORDS, clean
from .resources import Terminology

SOURCES = ("char", "word", "alias_char", "alias_word", "dense", "dense_bl")

# Qualifier words whose presence or absence usually changes the correct code.
QUALIFIERS = {
    "laterality": {"left", "right", "bilateral"},
    "acuity": {"acute", "chronic", "subacute"},
    "history": {"history", "hx", "previous", "prior", "past"},
    "family": {"family", "fhx"},
    "negation": {"no", "not", "denies", "negative", "without"},
    "type": {"type", "1", "2", "i", "ii"},
    "stage": {"stage", "grade", "class"},
    "encounter": {"initial", "subsequent", "sequela"},
    "screening": {"screening", "screen"},
}


class PriorStats:
    """Code and category frequencies among gold labels in a training fold."""

    def __init__(self, gold_codes: list[str], term: Terminology, subconcepts: list[str] | None = None):
        self.code = Counter(gold_codes)
        self.n = max(len(gold_codes), 1)
        groups = [self._group(c, term) for c in gold_codes]
        self.group = Counter(groups)
        subs = subconcepts or [""] * len(gold_codes)
        self.sub_n = Counter(subs)
        self.sub_group = Counter(zip(subs, groups))
        self.sub_code = Counter(zip(subs, gold_codes))
        self.n_groups = max(len(self.group), 1)

    def sub_logp(self, code: str, sub: str, term: Terminology) -> tuple[float, float]:
        """log P(group | subconcept) and log P(code | subconcept), smoothed."""
        n = self.sub_n.get(sub, 0)
        g = self._group(code, term)
        pg = (self.sub_group.get((sub, g), 0) + 0.5) / (n + 0.5 * self.n_groups)
        pc = (self.sub_code.get((sub, code), 0) + 0.1) / (n + 1)
        return math.log(pg), math.log(pc)

    @staticmethod
    def _group(code: str, term: Terminology) -> str:
        if term.system == "ICD10CM":
            return code[:3]
        try:
            return term.codes.at[code, "tag"]
        except KeyError:
            return "other"

    def code_logp(self, code: str) -> float:
        return math.log((self.code.get(code, 0) + 0.1) / self.n)

    def group_logp(self, code: str, term: Terminology) -> float:
        return math.log((self.group.get(self._group(code, term), 0) + 0.5) / self.n)


def _ctoks(text: str) -> set[str]:
    return {t for t in text.split() if t not in STOPWORDS}


def featurize(
    query: str,
    variants: list[str],
    context: str,
    cands: dict[str, Candidate],
    term: Terminology,
    prior: PriorStats,
    exact_alias: set[str],
    subconcept: str | None = None,
) -> tuple[list[str], np.ndarray]:
    """Return candidate codes and a feature matrix (one row per candidate)."""
    q = clean(query)
    q_toks = _ctoks(q)
    ctx_toks = _ctoks(clean(context)) - q_toks
    q_all = set(q.split())
    is_icd = term.system == "ICD10CM"
    codes_tbl = term.codes

    rows, codes = [], []
    max_by_source = {s: max((c.scores.get(s, 0.0) for c in cands.values()), default=0.0) for s in SOURCES}
    for code, c in cands.items():
        title = clean(term.title(code))
        best = c.best_text or title
        v = variants[c.variant] if c.variant < len(variants) else q
        d_toks = _ctoks(best)
        t_toks = _ctoks(title)
        v_toks = _ctoks(v)

        f = []
        for s in SOURCES:
            sc = c.scores.get(s, 0.0)
            f += [sc, sc - max_by_source[s], min(c.ranks.get(s, ABSENT_RANK), ABSENT_RANK)]
        f.append(len(c.scores))
        f.append(float(code in exact_alias))

        # surface similarity against the matched description and the title
        f += [
            fuzz.ratio(v, best) / 100,
            fuzz.token_set_ratio(v, best) / 100,
            fuzz.token_sort_ratio(v, best) / 100,
            fuzz.partial_ratio(v, best) / 100,
            fuzz.ratio(q, title) / 100,
            fuzz.token_set_ratio(v, title) / 100,
        ]
        inter = v_toks & d_toks
        f += [
            len(inter) / max(len(v_toks), 1),  # query coverage
            len(inter) / max(len(d_toks), 1),  # description coverage
            len(d_toks - v_toks),  # unexplained description words
            len((d_toks - v_toks) & ctx_toks) / max(len(d_toks - v_toks), 1),  # ...explained by context
            len(v_toks - d_toks),
            len(t_toks & ctx_toks) / max(len(t_toks), 1),
            len(v_toks),
            len(d_toks),
            float(c.variant > 0),
            float(c.via_parent),
        ]

        d_all = set(best.split()) | set(title.split())
        for words in QUALIFIERS.values():
            in_q, in_d = bool(q_all & words), bool(d_all & words)
            f += [float(in_q and not in_d), float(in_d and not in_q)]

        f += [prior.code_logp(code), prior.group_logp(code, term), float(prior.code.get(code, 0) > 0)]

        if is_icd:
            r = codes_tbl.loc[code] if code in codes_tbl.index else None
            f += [
                float(r is not None and r["is_leaf"]),
                float(r["depth"]) if r is not None else 0.0,
                float(r is not None and r["unspecified"]),
                float(r is not None and r["other"]),
                float(r is not None and r["n_desc"]),
            ]
        else:
            n_desc = float(codes_tbl.at[code, "n_desc"]) if code in codes_tbl.index else 0.0
            f += [0.0, 0.0, float("unspecified" in title), 0.0, n_desc]
        f.append(float(is_icd))
        # subconcept features go last so a model without them can drop the columns
        f += list(prior.sub_logp(code, subconcept, term)) if subconcept else [0.0, 0.0]
        rows.append(f)
        codes.append(code)
    return codes, np.asarray(rows, dtype=np.float32)


def feature_names() -> list[str]:
    names = []
    for s in SOURCES:
        names += [f"{s}_score", f"{s}_gap", f"{s}_rank"]
    names += ["n_sources", "exact_alias"]
    names += ["fz_ratio", "fz_tset", "fz_tsort", "fz_partial", "fz_title_ratio", "fz_title_tset"]
    names += ["q_cov", "d_cov", "d_extra", "d_extra_ctx", "q_extra", "title_ctx", "q_len", "d_len",
              "from_variant", "via_parent"]
    for g in QUALIFIERS:
        names += [f"{g}_q_only", f"{g}_d_only"]
    names += ["prior_code", "prior_group", "seen_code"]
    names += ["is_leaf", "depth", "unspecified", "other", "n_desc"]
    names += ["is_icd", "sub_group", "sub_code"]
    return names


N_SUBCONCEPT_FEATURES = 2

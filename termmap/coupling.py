"""Coupling the two coding systems through the coding team's own pairings.

Terms in the labelled data that carry both an ICD-10-CM and a SNOMED CT gold
code form a local crosswalk: when the team chose SNOMED concept s for a term,
which ICD-10-CM code did it choose for the same term, and vice versa. The
crosswalk is used as a soft prior. For a candidate c in one system,

    q(c) = sum_s p_other(s) * P(c | s)

where p_other is the engine's own distribution over the other system's top
candidates for the same term. The prior is blended log-linearly with the
system's own scores. Because both predictions are made independently first,
neither system is derived from the other; the crosswalk only re-weights
candidates the system already retrieved.

For ICD-10-CM targets the pairing backs off to the three-character block when
the exact code was never paired.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ICD, SNOMED = "ICD10CM", "SNOMEDCT_US"


class Crosswalk:
    def __init__(self, train: pd.DataFrame):
        key = train["entity"].astype(str) + "\x1f" + train["chunk_text"].fillna("").astype(str)
        both = pd.DataFrame({"key": key, "system": train["coding_system"], "code": train["code"]})
        icd = both[both.system == ICD].drop_duplicates("key").set_index("key")["code"]
        sno = both[both.system == SNOMED].drop_duplicates("key").set_index("key")["code"]
        common = icd.index.intersection(sno.index)
        self.n_pairs = len(common)
        self.pairs = Counter(zip(sno.loc[common], icd.loc[common]))
        self.by_snomed: dict[str, Counter] = defaultdict(Counter)
        self.by_icd: dict[str, Counter] = defaultdict(Counter)
        self.block_by_snomed: dict[str, Counter] = defaultdict(Counter)
        for (s, i), n in self.pairs.items():
            self.by_snomed[s][i] += n
            self.by_icd[i][s] += n
            self.block_by_snomed[s][i[:3]] += n

    def _cond(self, target_system: str, cand: str, given: str) -> float | None:
        """P(cand | given), or None when `given` was never paired."""
        if target_system == ICD:
            c = self.by_snomed.get(given)
            if not c:
                return None
            n = sum(c.values())
            exact = c.get(cand, 0) / n
            block = self.block_by_snomed[given].get(cand[:3], 0) / n
            return 0.7 * exact + 0.3 * block
        c = self.by_icd.get(given)
        if not c:
            return None
        return c.get(cand, 0) / sum(c.values())

    def prior(self, target_system: str, cands: list[str], other_ranked: list[tuple[str, float]]):
        """Coupling prior over `cands`, or None when no paired evidence exists."""
        q = np.zeros(len(cands))
        mass = 0.0
        for code, p in other_ranked:
            vals = [self._cond(target_system, c, code) for c in cands]
            if vals[0] is None:
                continue
            q += p * np.asarray(vals)
            mass += p
        if mass == 0 or q.sum() == 0:
            return None
        return q / mass


def rescore(ranked: list[tuple[str, float]], q: np.ndarray, weight: float, floor: float = 0.02):
    """Blend the coupling prior into a ranked list (log-linear, floored prior)."""
    codes = [c for c, _ in ranked]
    p = np.asarray([s for _, s in ranked], dtype=np.float64)
    p = p / p.sum() if p.sum() > 0 else np.full(len(p), 1.0 / len(p))
    s = np.log(p + 1e-9) + weight * np.log(q + floor)
    e = np.exp(s - s.max())
    e /= e.sum()
    order = np.argsort(-e, kind="stable")
    return [(codes[i], float(e[i])) for i in order]

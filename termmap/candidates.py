"""Candidate generation: turn one term into a short list of plausible codes."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .index import LexicalIndex
from .resources import Terminology

ABSENT_RANK = 99


@dataclass
class Candidate:
    code: str
    scores: dict = field(default_factory=dict)  # source -> best score
    ranks: dict = field(default_factory=dict)  # source -> best code-level rank
    best_text: str = ""  # description that produced the best lexical score
    best_text_score: float = -1.0
    variant: int = 0  # index of the query variant that produced best_text
    via_parent: bool = False

    def add(self, source: str, score: float, rank: int, text: str | None = None, variant: int = 0):
        if score > self.scores.get(source, -1.0):
            self.scores[source] = score
        if rank < self.ranks.get(source, ABSENT_RANK):
            self.ranks[source] = rank
        if text is not None and score > self.best_text_score:
            self.best_text, self.best_text_score, self.variant = text, score, variant


class DictionaryRetriever:
    """Character and word TF-IDF retrieval over one terminology's descriptions."""

    sources = ("char", "word")

    def __init__(self, term: Terminology):
        self.term = term
        d = term.descriptions
        self.codes = d["code"].to_numpy()
        self.texts = d["norm"].tolist()
        self.char = LexicalIndex("char_wb", (3, 4), max_df=0.2).fit(self.texts)
        self.word = LexicalIndex("word", (1, 2), max_df=0.1).fit(self.texts)
        self._cache: dict[tuple[str, str], list[tuple[str, float, str]]] = {}

    def _search(self, source: str, queries: list[str], k: int, batch: int):
        pending = [q for q in dict.fromkeys(queries) if (source, q) not in self._cache]
        if pending:
            index = self.char if source == "char" else self.word
            for q, hits in zip(pending, index.search(pending, k * 3, batch)):
                seen, out = set(), []
                for h in hits:
                    c = self.codes[h.row]
                    if c in seen:
                        continue
                    seen.add(c)
                    out.append((c, h.score, self.texts[h.row]))
                    if len(out) == k:
                        break
                self._cache[(source, q)] = out
        return [self._cache[(source, q)] for q in queries]

    def retrieve(self, queries: list[str], k: int, batch: int = 256):
        return {s: self._search(s, queries, k, batch) for s in self.sources}


class AliasRetriever:
    """Retrieval over training terms, each pointing at its gold code.

    Captures local coding convention and shorthand that no dictionary holds.
    """

    def __init__(self, aliases: pd.DataFrame):
        a = aliases.drop_duplicates(["norm", "code"]).reset_index(drop=True)
        self.codes = a["code"].to_numpy()
        self.texts = a["norm"].tolist()
        self.exact = defaultdict(set)
        for t, c in zip(self.texts, self.codes):
            self.exact[t].add(c)
        self.char = LexicalIndex("char_wb", (3, 4), max_df=0.5).fit(self.texts)
        self.word = LexicalIndex("word", (1, 2), max_df=0.5).fit(self.texts)

    def retrieve(self, queries: list[str], k: int):
        out = {}
        for name, index in (("alias_char", self.char), ("alias_word", self.word)):
            res = []
            for hits in index.search(queries, k * 3):
                seen, lst = set(), []
                for h in hits:
                    c = self.codes[h.row]
                    if c in seen:
                        continue
                    seen.add(c)
                    lst.append((c, h.score, self.texts[h.row]))
                    if len(lst) == k:
                        break
                res.append(lst)
            out[name] = res
        return out


def icd_unspecified_children(term: Terminology) -> dict[str, list[str]]:
    """Map each non-leaf ICD code to its leaf descendants that are the
    'unspecified' or 'without complication' default for that category."""
    codes = term.codes
    leaves = codes[codes["is_leaf"]]
    default = leaves[
        leaves["title"].str.lower().str.contains("unspecified|without complication|without mention")
    ].index
    out: dict[str, list[str]] = defaultdict(list)
    nonleaf = set(codes.index[~codes["is_leaf"]])
    for c in default:
        for k in range(3, len(c)):
            p = c[:k]
            if p in nonleaf:
                out[p].append(c)
    for p in out:
        out[p] = sorted(out[p], key=len)[:3]
    return dict(out)


def generate(
    variants: list[list[str]],
    dict_hits: dict,
    alias_hits: dict | None,
    extra_hits: dict | None,
    max_candidates: int,
    parent_children: dict[str, list[str]] | None = None,
) -> list[dict[str, Candidate]]:
    """Merge retrieval results from every source and every query variant.

    `variants[i]` is the list of query strings for term i; the hit dicts hold
    one result list per flattened query, in the same order.
    """
    out = []
    pos = 0
    sources = {**dict_hits, **(alias_hits or {}), **(extra_hits or {})}
    for qs in variants:
        cands: dict[str, Candidate] = {}
        for v in range(len(qs)):
            for source, results in sources.items():
                for rank, (code, score, text) in enumerate(results[pos + v]):
                    c = cands.get(code)
                    if c is None:
                        c = cands[code] = Candidate(code)
                    lexical = source in ("char", "word", "alias_char", "alias_word")
                    c.add(source, score, rank, text if lexical else None, v)
        pos += len(qs)
        if parent_children:
            for code in list(cands):
                for child in parent_children.get(code, ()):
                    if child not in cands:
                        parent = cands[code]
                        cands[child] = Candidate(child, dict(parent.scores), dict(parent.ranks),
                                                 parent.best_text, parent.best_text_score, parent.variant, True)
        if len(cands) > max_candidates:
            # scores from different sources are not on one scale; rank is
            keep = sorted(
                cands.values(), key=lambda c: (min(c.ranks.values()), -max(c.scores.values()))
            )[:max_candidates]
            cands = {c.code: c for c in keep}
        out.append(cands)
    return out

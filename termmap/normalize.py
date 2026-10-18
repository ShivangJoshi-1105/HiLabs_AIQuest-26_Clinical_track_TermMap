"""Text normalisation and abbreviation expansion."""
from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path

_HTML = re.compile(r"<[^>]+>")
_HEADER = re.compile(r"header_\d+\s*:")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_SPACE = re.compile(r"\s+")
_SEMTAG = re.compile(r"\s*\(([a-z /+\-]+)\)\s*$", re.I)

# Tokens that carry no meaning for coding but appear often in extracted terms.
STOPWORDS = frozenset(
    "a an the of and or with without to in on at by for from as is was are be "
    "pt patient hx history h o c w w o s p r o".split()
)


def clean(text: str) -> str:
    """Lowercase, drop markup and punctuation, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = _HTML.sub(" ", text)
    text = _HEADER.sub(" ", text)
    text = text.lower().replace("'s", "")
    text = _NON_ALNUM.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def strip_semantic_tag(description: str) -> tuple[str, str | None]:
    """Split 'Chest pain (finding)' into ('Chest pain', 'finding')."""
    m = _SEMTAG.search(description or "")
    if not m:
        return description, None
    return description[: m.start()].strip(), m.group(1).lower()


def tokens(text: str) -> list[str]:
    return [t for t in clean(text).split() if t]


def content_tokens(text: str) -> set[str]:
    return {t for t in tokens(text) if t not in STOPWORDS}


class AbbreviationExpander:
    """Expands clinical abbreviations into one or more candidate readings.

    Ambiguous abbreviations keep every reading; the downstream ranker and the
    chunk context decide between them.
    """

    def __init__(self, table: dict[str, list[str]], cues: dict[tuple[str, str], set[str]] | None = None):
        self.table = table
        self.cues = cues or {}  # (abbreviation, reading) -> context words that signal the reading

    @classmethod
    def from_file(cls, path: Path) -> "AbbreviationExpander":
        table: dict[str, list[str]] = {}
        cues: dict[tuple[str, str], set[str]] = {}
        with open(path, "r", encoding="utf-8") as fh:
            for row in csv.reader(fh, delimiter="\t"):
                if not row or row[0].startswith("#") or len(row) < 2:
                    continue
                abbr, reading = clean(row[0]), clean(row[1])
                table.setdefault(abbr, []).append(reading)
                if len(row) > 2 and row[2].strip():
                    cues[(abbr, reading)] = {clean(c) for c in row[2].split(",") if clean(c)}
        return cls(table, cues)

    def _support(self, abbr: str, reading: str, ctx: set[str]) -> int:
        return len((set(reading.split()) | self.cues.get((abbr, reading), set())) & ctx)

    def resolve(self, abbr: str, context: str) -> str | None:
        """The single reading of an ambiguous abbreviation that the context supports, if any."""
        opts = self.table.get(abbr, [])
        if len(opts) < 2:
            return None
        ctx = content_tokens(context) - {abbr}
        scores = sorted(((self._support(abbr, o, ctx), o) for o in opts), reverse=True)
        if scores[0][0] > 0 and scores[0][0] > scores[1][0]:
            return scores[0][1]
        return None

    def variants(self, text: str, context: str = "", limit: int = 4) -> list[str]:
        base = clean(text)
        toks = base.split()
        hits = [(i, self.table[t]) for i, t in enumerate(toks) if t in self.table]
        if not hits:
            return [base]
        # a bare ambiguous abbreviation whose reading the context settles is searched
        # under that reading only, so frequent readings cannot outvote the context
        if len(toks) <= 2 and len(hits) == 1:
            reading = self.resolve(toks[hits[0][0]], context)
            if reading:
                new = list(toks)
                new[hits[0][0]] = reading
                return [" ".join(new)]
        ctx = content_tokens(context)
        out = [base]
        # one fully expanded variant per reading of the first ambiguous token,
        # ordered by how well each reading is supported by the context
        first_i, first_opts = hits[0]
        ranked = sorted(first_opts, key=lambda e: -len(set(e.split()) & ctx))
        for opt in ranked:
            new = list(toks)
            new[first_i] = opt
            for i, opts in hits[1:]:
                new[i] = max(opts, key=lambda e: len(set(e.split()) & ctx))
            out.append(" ".join(new))
            if len(out) > limit:
                break
        return list(dict.fromkeys(out))


@lru_cache(maxsize=1)
def default_expander(resource_dir: str) -> AbbreviationExpander:
    return AbbreviationExpander.from_file(Path(resource_dir) / "abbreviations.tsv")

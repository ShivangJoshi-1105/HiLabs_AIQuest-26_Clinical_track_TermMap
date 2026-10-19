"""Second-stage scorers over the reranker's top candidates.

Three scorers can re-score the reranker's top-N candidates of a term:

- ``ce``    a cross-encoder fine-tuned on (term + context, candidate description) pairs
- ``llm``   an instruction-tuned LLM, zero-shot, with retrieved in-context examples
- ``lora``  the same LLM fine-tuned with LoRA to pick the gold candidate

Each scorer needs a GPU and runs outside the core pipeline, either through the
notebooks in notebooks/ or `python -m termmap llm-score` on a GPU machine. The
engine writes request files and reads score files:

    requests: {"id", "system", ..., "codes": [...]}
    scores:   {"id", "probs": [...]}     one probability per code, same order

Scores are combined with the reranker by a log-linear blend,

    score(c) = log p_ranker(c) + sum_s  w_s * log p_s(c)

with per-system weights chosen on held-out data.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import logging
import random
from pathlib import Path

import numpy as np

from .normalize import clean

log = logging.getLogger(__name__)

SCORERS = {"ce": "ce_scores", "llm": "llm_scores", "lora": "lora_scores"}
LETTERS = "ABCDEFGHIJ"
SYSTEM_NAMES = {"ICD10CM": "ICD-10-CM", "SNOMEDCT_US": "SNOMED CT"}

SYSTEM_PROMPT = (
    "You are an experienced clinical coder. You map a clinical term extracted from a "
    "medical chart to exactly one code from a list of candidates. Use the chart context "
    "to resolve abbreviations and ambiguity. Do not add specificity the text does not "
    "support: when the text gives no further detail, the unspecified code is correct. "
    "Follow the coding conventions shown in the examples from the same coding team."
)


def request_id(system: str, entity: str, context: str, codes: list[str]) -> str:
    key = "\x1f".join([system, entity, context, *codes])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]


def context_window(entity: str, context: str, width: int = 600) -> str:
    context = " ".join((context or "").replace("<br>", " ").split())
    if len(context) <= width:
        return context
    pos = context.lower().find(entity.lower()[:20])
    if pos < 0:
        return context[:width] + " ..."
    start = max(0, pos - width // 2)
    return ("... " if start else "") + context[start : start + width] + " ..."


class CandidateText:
    """Readable description of a code: preferred term, semantic tag, synonyms."""

    def __init__(self, term):
        self.term = term
        self._desc = term.descriptions.groupby("code")["text"].apply(list).to_dict()

    def __call__(self, code: str, max_len: int = 160) -> str:
        title = self.term.title(code)
        extra = [t for t in dict.fromkeys(self._desc.get(code, [])) if t != title][:2]
        s = title
        if self.term.system == "SNOMEDCT_US":
            s += f" [{self.term.codes.at[code, 'tag']}]"
        if extra:
            s += " | also: " + "; ".join(extra)
        return s[:max_len]


def _examples(alias, entities: list[str], n: int) -> list[list[tuple[str, str]]]:
    """Nearest training terms with their gold codes (in-context examples)."""
    if alias is None or n <= 0:
        return [[] for _ in entities]
    out = []
    for hits in alias.char.search([clean(e) for e in entities], n * 2):
        seen, ex = set(), []
        for h in hits:
            text, code = alias.texts[h.row], alias.codes[h.row]
            if text in seen:
                continue
            seen.add(text)
            ex.append((text, code))
            if len(ex) == n:
                break
        out.append(ex)
    return out


def _sub(row) -> str:
    s = row.get("subconcept", "")
    return s if isinstance(s, str) else ""


class RequestBuilder:
    """Builds LLM prompts and cross-encoder inputs for one coding system."""

    def __init__(self, mapper, top_n: int = 10, n_examples: int = 6):
        self.mapper = mapper
        self.term = mapper.term
        self.system = mapper.term.system
        self.top_n = min(top_n, len(LETTERS))
        self.n_examples = n_examples
        self.describe = CandidateText(self.term)

    # -- LLM ----------------------------------------------------------------

    def _prompt(self, entity, ctx, sub, examples, codes) -> list[dict]:
        name = SYSTEM_NAMES[self.system]
        lines = [f'Clinical term: "{entity}"']
        window = context_window(entity, ctx)
        if window:
            lines.append(f'Chart context: "{window}"')
        if sub:
            lines.append(f"Entity category: {sub}")
        if examples:
            lines.append(f"\nHow the coding team mapped similar terms to {name}:")
            lines += [f'- "{t}" -> {c} {self.term.title(c)}' for t, c in examples]
        lines.append(f"\nCandidate {name} codes:")
        lines += [f"{LETTERS[i]}. {c} {self.describe(c)}" for i, c in enumerate(codes)]
        lines.append("\nAnswer with the letter of the single best candidate.")
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n".join(lines)}]

    def llm_requests(self, rows: list, preds: list, alias) -> list[dict]:
        examples = _examples(alias, [r["entity"] for r in rows], self.n_examples)
        reqs = []
        for row, ex, p in zip(rows, examples, preds):
            codes = [c for c, _ in p.ranked[: self.top_n]]
            if len(codes) < 2:
                continue
            ctx = row.get("chunk_text", "") or ""
            reqs.append({
                "id": request_id(self.system, row["entity"], ctx, codes),
                "system": self.system,
                "messages": self._prompt(row["entity"], ctx, _sub(row), ex, codes),
                "codes": codes,
            })
        return reqs

    def llm_training(self, records: list, seed: int) -> list[dict]:
        """Fine-tuning examples from out-of-fold candidate lists.

        The gold code is placed in the list when retrieval missed it, and the
        list is shuffled so the model cannot learn to trust the first position.
        """
        rng = random.Random(seed)
        out = []
        by_alias: dict[int, list] = {}
        for rec in records:
            by_alias.setdefault(id(rec[2]), []).append(rec)
        for recs in by_alias.values():
            alias = recs[0][2]
            examples = _examples(alias, [r["entity"] for r, _, _ in recs], self.n_examples)
            for (row, p, _), ex in zip(recs, examples):
                gold = row["code"]
                codes = [c for c, _ in p.ranked[: self.top_n]]
                if len(codes) < 2:
                    continue
                if gold not in codes:
                    codes[-1] = gold
                rng.shuffle(codes)
                messages = self._prompt(row["entity"], row.get("chunk_text", "") or "", _sub(row), ex, codes)
                out.append({"system": self.system, "messages": messages, "answer": LETTERS[codes.index(gold)]})
        return out

    # -- cross-encoder ------------------------------------------------------

    def _ce_query(self, row) -> str:
        q = row["entity"]
        sub = _sub(row)
        if sub:
            q += f" ({sub})"
        window = context_window(row["entity"], row.get("chunk_text", "") or "", 300)
        return f"{q} [SEP] {window}" if window else q

    def ce_requests(self, rows: list, preds: list) -> list[dict]:
        reqs = []
        for row, p in zip(rows, preds):
            codes = [c for c, _ in p.ranked[: self.top_n]]
            if len(codes) < 2:
                continue
            ctx = row.get("chunk_text", "") or ""
            reqs.append({
                "id": request_id(self.system, row["entity"], ctx, codes),
                "system": self.system,
                "query": self._ce_query(row),
                "docs": [f"{c} {self.describe(c)}" for c in codes],
                "codes": codes,
            })
        return reqs

    def ce_training(self, records: list, n_neg: int = 15) -> list[dict]:
        """One group per term: the gold description first, then hard negatives."""
        out = []
        for row, p, _ in records:
            gold = row["code"]
            if gold not in self.term:
                continue
            negs = [c for c, _ in p.ranked if c != gold][:n_neg]
            if not negs:
                continue
            out.append({
                "system": self.system,
                "query": self._ce_query(row),
                "docs": [f"{c} {self.describe(c)}" for c in [gold] + negs],
            })
        return out


def write_jsonl(items: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in items:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("wrote %d records to %s", len(items), path)


def load_scores(directory: Path) -> dict[str, dict[str, list[float]]]:
    """scorer name -> request id -> probabilities"""
    out: dict[str, dict[str, list[float]]] = {}
    for name, prefix in SCORERS.items():
        scores: dict[str, list[float]] = {}
        for f in sorted(Path(directory).glob(f"{prefix}*.jsonl")):
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        r = json.loads(line)
                        scores[r["id"]] = r["probs"]
        if scores:
            out[name] = scores
            log.info("loaded %d %s scores", len(scores), name)
    return out


def _norm(p, n):
    p = np.asarray(p, dtype=np.float64)[:n]
    return p / p.sum() if p.sum() > 0 else np.full(n, 1.0 / n)


def blend(ranked: list[tuple[str, float]], extra: dict[str, list[float]], weights: dict[str, float]):
    """Log-linear blend over the candidates every scorer has seen."""
    n = min([len(ranked)] + [len(p) for p in extra.values()])
    codes = [c for c, _ in ranked[:n]]
    s = np.log(_norm([x for _, x in ranked], n) + 1e-9)
    for name, probs in extra.items():
        w = weights.get(name, 0.0)
        if w:
            s = s + w * np.log(_norm(probs, n) + 1e-9)
    e = np.exp(s - s.max())
    e /= e.sum()
    order = np.argsort(-e, kind="stable")
    return [(codes[i], float(e[i])) for i in order]


class CrossEncoderScorer:
    """Runs the fine-tuned cross-encoder locally (GPU if present, otherwise CPU)."""

    def __init__(self, model_dir: Path, max_len: int = 192):
        import os

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() and os.environ.get("TERMMAP_FORCE_CPU", "0") != "1" else "cpu"
        if self.device == "cpu":
            torch.set_num_threads(int(os.environ.get("TERMMAP_CPU_THREADS", "2")))
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        if self.device == "cpu" and os.environ.get("TERMMAP_INT8_MODELS", "0") == "1":
            from .dense import quantize_model

            self.model = quantize_model(self.model)
        self.max_len = max_len

    def score(self, reqs: list[dict], batch: int = 64) -> dict[str, list[float]]:
        torch = self.torch
        pairs = [(r["query"], d) for r in reqs for d in r["docs"]]
        logits: list[float] = []
        with torch.no_grad():
            for b in range(0, len(pairs), batch):
                chunk = pairs[b : b + batch]
                enc = self.tok([p[0] for p in chunk], [p[1] for p in chunk], truncation="longest_first",
                               max_length=self.max_len, padding=True, return_tensors="pt").to(self.device)
                logits += self.model(**enc).logits.squeeze(-1).float().cpu().tolist()
        out, pos = {}, 0
        for r in reqs:
            n = len(r["docs"])
            p = torch.softmax(torch.tensor(logits[pos : pos + n]), dim=0).tolist()
            out[r["id"]] = [round(x, 6) for x in p]
            pos += n
        return out


WEIGHT_GRID = (0.0, 0.5, 1.0, 2.0, 4.0)


def weight_grid(names: list[str]):
    for combo in itertools.product(WEIGHT_GRID, repeat=len(names)):
        yield dict(zip(names, combo))

"""Command line interface.

    python -m termmap train      fit the engine on the labelled data
    python -m termmap predict    map every row of an input file
    python -m termmap evaluate   held-out evaluation on the labelled data
    python -m termmap map        map a single term from the command line
    python -m termmap embed      build the dense description index
    python -m termmap export-queries   list every query string for offline dense search
    python -m termmap export-stage2    write cross-encoder and LLM requests for an input file
    python -m termmap fetch-artifacts  download trained artifacts from the Hub repository
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from . import SYSTEMS
from .config import Config
from .engine import TermMapEngine
from .resources import load_terminology, read_terms


def _engine_path(cfg: Config) -> Path:
    return Path(cfg.artifact_dir) / "engine.joblib"


def _load_or_train(cfg: Config) -> TermMapEngine:
    path = _engine_path(cfg)
    if path.exists():
        return TermMapEngine.load(cfg, path)
    logging.info("no trained engine found, training first")
    return cmd_train(cfg, None)


def cmd_train(cfg: Config, args) -> TermMapEngine:
    from .stage2 import write_jsonl

    train = read_terms(cfg.raw_dir / cfg.train_file)
    engine = TermMapEngine(cfg).fit(train)
    engine.save(_engine_path(cfg))
    logging.info("saved %s", _engine_path(cfg))
    # fine-tuning data for the second-stage scorers, from out-of-fold candidates
    sets = {"llm": [], "ce": []}
    for system in SYSTEMS:
        for k, items in engine.stage2_training(system).items():
            sets[k] += items
    for k, items in sets.items():
        write_jsonl(items, Path(cfg.output_dir) / f"{k}_train_full.jsonl")
    return engine


def cmd_predict(cfg: Config, args) -> None:
    df = read_terms(args.input)
    engine = _load_or_train(cfg)
    t = time.time()
    out = engine.predict(df)
    elapsed = time.time() - t
    cols = [c for c in ("chunk_id", "clinical_entity", "entity", "coding_system") if c in out.columns]
    if "clinical_entity" not in out.columns:
        out = out.rename(columns={"entity": "clinical_entity"})
        cols = ["clinical_entity" if c == "entity" else c for c in cols]
    keep = cols + ["icd10cm_code", "snomed_code"]
    if args.with_confidence:
        keep += ["icd10cm_code_confidence", "snomed_code_confidence"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out[keep].to_csv(output, index=False)
    logging.info("wrote %d rows to %s (%.1f ms per term)", len(out), output, 1000 * elapsed / max(len(out), 1))


def cmd_evaluate(cfg: Config, args) -> None:
    from .evaluate import run

    out_dir = Path(cfg.output_dir) / f"eval_{args.tag}"
    report = run(cfg, read_terms(cfg.raw_dir / cfg.train_file), frac=args.train_fraction, out_dir=out_dir)
    print(json.dumps(report, indent=2))


def cmd_map(cfg: Config, args) -> None:
    import pandas as pd

    engine = _load_or_train(cfg)
    df = pd.DataFrame({"entity": [args.term], "chunk_text": [args.context or ""]})
    row = engine.predict(df).iloc[0]
    for system, col in zip(SYSTEMS, ("icd10cm_code", "snomed_code")):
        code = row[col]
        title = engine.mapper(system).term.title(code)
        print(f"{system:<12} {code:<12} {row[col + '_confidence']:.3f}  {title}")


def cmd_export_queries(cfg: Config, args) -> None:
    from .normalize import AbbreviationExpander

    expander = AbbreviationExpander.from_file(cfg.resource_dir / "abbreviations.tsv")
    files = [cfg.raw_dir / cfg.train_file] + [Path(p) for p in (args.input or [])]
    queries: dict[str, None] = {}
    for f in files:
        df = read_terms(f)
        for e, c in zip(df["entity"], df["chunk_text"]):
            for q in expander.variants(e, c):
                if q:
                    queries[q] = None
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(queries) + "\n", encoding="utf-8")
    logging.info("wrote %d queries to %s", len(queries), out)


def cmd_export_stage2(cfg: Config, args) -> None:
    """Write cross-encoder and LLM requests for every row of an input file."""
    from .stage2 import write_jsonl

    df = read_terms(args.input).reset_index(drop=True)
    engine = _load_or_train(cfg)
    out = {"llm": [], "ce": []}
    for system in SYSTEMS:
        reqs = engine.stage2_requests(system, df, engine.mapper(system).predict(df))
        for k in out:
            out[k] += reqs[k]
    suffix = f"_{args.name}" if args.name else ""
    for k, items in out.items():
        write_jsonl(items, Path(cfg.output_dir) / f"{k}_requests{suffix}.jsonl")


def cmd_fetch_artifacts(cfg: Config, args) -> None:
    from .hub import fetch

    fetch(cfg, args.repo, embeddings=not args.no_embeddings)


def cmd_embed(cfg: Config, args) -> None:
    from .dense import build

    for system in args.systems or SYSTEMS:
        descriptions = load_terminology(cfg, system).descriptions
        for prefix in cfg.dense_models:
            build(cfg, system, descriptions, prefix)


def main(argv=None) -> None:
    from .hub import load_dotenv

    load_dotenv()
    p = argparse.ArgumentParser(prog="termmap")
    p.add_argument("--config", default=None, help="YAML config file")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("train")
    pp = sub.add_parser("predict")
    pp.add_argument("--input", required=True)
    pp.add_argument("--output", default="outputs/predictions.csv")
    pp.add_argument("--with-confidence", action="store_true")
    pe = sub.add_parser("evaluate")
    pe.add_argument("--train-fraction", type=float, default=0.8)
    pe.add_argument("--tag", default="latest", help="results go to outputs/eval_<tag>/")
    pm = sub.add_parser("map")
    pm.add_argument("term")
    pm.add_argument("--context", default="")
    pb = sub.add_parser("embed")
    pb.add_argument("--systems", nargs="*", choices=SYSTEMS)
    pq = sub.add_parser("export-queries")
    pq.add_argument("--input", nargs="*", help="additional term files, e.g. the evaluation file")
    pq.add_argument("--output", default="outputs/dense_queries.txt")
    pf = sub.add_parser("fetch-artifacts")
    pf.add_argument("--repo", default=None, help="Hub repository id (default: TERMMAP_ARTIFACTS_REPO)")
    pf.add_argument("--no-embeddings", action="store_true", help="skip the description embeddings")
    pl = sub.add_parser("export-stage2")
    pl.add_argument("--input", required=True)
    pl.add_argument("--name", default="", help="suffix for the request file names")

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    cfg = Config.load(args.config)
    {"train": cmd_train, "predict": cmd_predict, "evaluate": cmd_evaluate,
     "map": cmd_map, "embed": cmd_embed, "export-queries": cmd_export_queries,
     "export-stage2": cmd_export_stage2, "fetch-artifacts": cmd_fetch_artifacts}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()

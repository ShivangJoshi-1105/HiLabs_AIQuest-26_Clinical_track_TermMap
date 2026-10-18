"""Upload model weights to the Hugging Face Hub.

    python scripts/upload_artifacts.py --repo <user>/termmap-weights --public
    python scripts/upload_artifacts.py --repo <user>/termmap-artifacts --with-engine

Public repositories receive model weights only: the fine-tuned cross-encoder and
the description embeddings. The trained engine, the dense hits and the parsed
dictionaries contain labelled terms and dictionary text, so they are uploaded only
to a private repository (--with-engine) and the script refuses to send them to a
public one.

The token is read from the HF_TOKEN environment variable (or .env); it is never
printed or written anywhere.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from termmap.config import Config  # noqa: E402
from termmap.hub import DATA_DERIVED, WEIGHTS, load_dotenv  # noqa: E402

MODEL_CARD = """---
license: apache-2.0
tags:
- clinical-nlp
- entity-linking
- icd-10-cm
- snomed-ct
---

# TermMap weights

Model weights for [TermMap](https://github.com/brindoxx/HiLabs-TermMap), a clinical term mapping engine for
ICD-10-CM and SNOMED CT.

| Path | Content |
|---|---|
| `ce_model/` | `ncbi/MedCPT-Cross-Encoder` fine-tuned to re-score the top 10 candidates of a term |
| `dense_<system>.emb.npy`, `.codes.npy` | SapBERT embeddings of every dictionary description (float16) and their codes |
| `dense_biolord_<system>.emb.npy`, `.codes.npy` | BioLORD-2023 embeddings of every dictionary description and their codes |

Download into the repository's `artifacts/` folder with `python -m termmap fetch-artifacts`.
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="Hub repository id, e.g. team/termmap-weights")
    p.add_argument("--public", action="store_true", help="create the repository as public (weights only)")
    p.add_argument("--with-engine", action="store_true", help="also upload data-derived files (private repositories only)")
    args = p.parse_args()

    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set (environment variable or .env)")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="model", private=not args.public, exist_ok=True)
    private = api.repo_info(args.repo, repo_type="model").private
    if args.with_engine and not private:
        sys.exit(f"{args.repo} is public; data-derived files are only uploaded to private repositories")

    patterns = list(WEIGHTS) + (list(DATA_DERIVED) if args.with_engine else [])
    cfg = Config.load()
    api.upload_file(path_or_fileobj=MODEL_CARD.encode("utf-8"), path_in_repo="README.md",
                    repo_id=args.repo, repo_type="model", commit_message="Model card")
    api.upload_folder(folder_path=str(cfg.artifact_dir), repo_id=args.repo, repo_type="model",
                      allow_patterns=patterns, commit_message="Upload weights")
    print(f"uploaded to {args.repo} ({'private' if private else 'public'})")


if __name__ == "__main__":
    main()

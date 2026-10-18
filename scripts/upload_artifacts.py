"""Upload model artifacts to a private Hugging Face Hub repository.

    python scripts/upload_artifacts.py --repo <user>/termmap-artifacts [--with-embeddings]

The token is read from the HF_TOKEN environment variable (or .env); it is never
written anywhere. The repository is created as private if it does not exist.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from termmap.config import Config  # noqa: E402
from termmap.hub import ARTIFACTS, EMBEDDINGS, load_dotenv  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="Hub repository id, e.g. team/termmap-artifacts")
    p.add_argument("--with-embeddings", action="store_true", help="also upload description embeddings (about 2.3 GB)")
    args = p.parse_args()

    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set (environment variable or .env)")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="model", private=True, exist_ok=True)
    info = api.repo_info(args.repo, repo_type="model")
    if not info.private:
        sys.exit(f"{args.repo} is public; artifacts are derived from the challenge data and must stay private")

    patterns = list(ARTIFACTS) + (list(EMBEDDINGS) if args.with_embeddings else [])
    cfg = Config.load()
    api.upload_folder(folder_path=str(cfg.artifact_dir), repo_id=args.repo, repo_type="model",
                      allow_patterns=patterns, commit_message="Upload artifacts")
    print(f"uploaded to {args.repo} (private)")


if __name__ == "__main__":
    main()

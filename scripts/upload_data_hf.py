"""Upload training data and precomputed artifacts to Hugging Face Hub.

Pushes train.parquet, dev.parquet, sample_1M.parquet, and zone-pair stats
so training notebooks on Kaggle/Colab can download without re-processing.

Usage:
    # Requires HF_TOKEN environment variable
    python scripts/upload_data_hf.py

    # Custom repo
    python scripts/upload_data_hf.py --repo-id your-username/eta-data
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_REPO = "sarthakbiswas/nyc-eta-data"

FILES_TO_UPLOAD = [
    (DATA_DIR / "train.parquet", "train.parquet"),
    (DATA_DIR / "dev.parquet", "dev.parquet"),
    (DATA_DIR / "sample_1M.parquet", "sample_1M.parquet"),
    (DATA_DIR / "zone_pair_stats" / "zone_pair_stats.pkl", "zone_pair_stats.pkl"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload ETA data to HF Hub")
    parser.add_argument("--repo-id", default=DEFAULT_REPO, help="HF dataset repo ID")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN environment variable first.")

    from huggingface_hub import HfApi

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)
    logger.info("Repo: https://huggingface.co/datasets/%s", args.repo_id)

    for local_path, remote_name in FILES_TO_UPLOAD:
        if not local_path.exists():
            logger.warning("Skipping %s (not found)", local_path)
            continue
        size_mb = local_path.stat().st_size / (1024 * 1024)
        logger.info("Uploading %s (%.1f MB)...", remote_name, size_mb)
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_name,
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        logger.info("  Done: %s", remote_name)

    logger.info("All uploads complete.")


if __name__ == "__main__":
    main()

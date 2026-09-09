"""Copy the consumer views of benchmark-data-pipeline into 0_input/, with their provenance."""

import json
import logging
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from benchmark_forecasting.config import (
    BASELINES_FILE,
    INPUT_DIR,
    MANIFEST_FILE,
    PIPELINE_DIR,
    PROVENANCE_FILE,
    SCORES_FILE,
)

log = logging.getLogger(__name__)


def pipeline_repo(pipeline_dir: Path) -> str:
    """The pipeline's git remote URL, or its folder name when it has none.

    Recorded instead of the local checkout path, which would carry a user's home
    directory into a tracked file and means nothing on another machine."""
    try:
        out = subprocess.run(["git", "-C", str(pipeline_dir), "remote", "get-url", "origin"],
                             capture_output=True, text=True, check=True).stdout.strip()
        return out or pipeline_dir.name
    except (OSError, subprocess.CalledProcessError):
        return pipeline_dir.name

# (relative path in the pipeline repository, file name here)
SOURCES = (
    (Path("3_views") / SCORES_FILE, SCORES_FILE),
    (Path("3_views") / BASELINES_FILE, BASELINES_FILE),
    (Path("2_database") / MANIFEST_FILE, MANIFEST_FILE),
)


def sync(pipeline_dir: Path = PIPELINE_DIR, input_dir: Path = INPUT_DIR) -> dict:
    """Copy the views and the manifest, write provenance.json, return it."""
    pipeline_dir = Path(pipeline_dir)
    input_dir.mkdir(exist_ok=True)
    for rel, name in SOURCES:
        src = pipeline_dir / rel
        if not src.exists():
            raise FileNotFoundError(
                f"{src}: run `python -m benchmark_data build` in the pipeline first"
            )
        shutil.copyfile(src, input_dir / name)
        log.info("copied %s (%d bytes)", src, src.stat().st_size)
    manifest = json.loads((input_dir / MANIFEST_FILE).read_text())
    provenance = {
        "pipeline_repo": pipeline_repo(pipeline_dir),
        "pipeline_commit": manifest.get("git_commit", ""),
        "pipeline_built_at": manifest.get("built_at", ""),
        "schema_version": manifest.get("schema_version", ""),
        "synced_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": [name for _, name in SOURCES],
    }
    (input_dir / PROVENANCE_FILE).write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance

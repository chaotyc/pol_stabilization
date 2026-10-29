"""Stable paths for isolated experiment artifacts."""
from pathlib import Path
from uuid import uuid4
import re


def new_run_id(prefix="run"):
    return f"{prefix}_{uuid4().hex[:12]}"


def run_directory(output_dir, dataset, run_id):
    for value in (dataset, run_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise ValueError("Dataset and run ID may contain only letters, numbers, underscores, and hyphens")
    return Path(output_dir) / "runs" / dataset / run_id


def create_run_directory(output_dir, dataset, run_id=None):
    path = run_directory(output_dir, dataset, run_id or new_run_id())
    # Reusing an ID must fail rather than replace the checkpoint behind published metrics.
    path.mkdir(parents=True, exist_ok=False)
    return path

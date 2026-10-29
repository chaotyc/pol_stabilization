"""Run training and read only the artifacts produced by that invocation."""
import json
from pathlib import Path
import subprocess
import sys

from src.utils.data import REPO_ROOT
from src.utils.runs import run_directory


def run_training(dataset, run_id, output_dir="results", **parameters):
    output_dir = Path(output_dir).resolve()
    if parameters.get("data_path"):
        parameters["data_path"] = str(Path(parameters["data_path"]).resolve())
    cmd = [sys.executable, "-m", "src.training.mamba_training",
           f"--wavelength-range={dataset}", "--run-id", run_id, "--output-dir", str(output_dir)]
    for key, value in parameters.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        elif value is not None:
            cmd.extend([flag, str(value)])
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    with (run_directory(output_dir, dataset, run_id) / "metrics.json").open() as stream:
        return json.load(stream)


def study_name(dataset, loss):
    return f"mamba_{dataset}_{loss.lower()}"

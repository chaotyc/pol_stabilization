"""Dataset locations and validation shared by training and evaluation."""
from pathlib import Path

import numpy as np
import scipy.io

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATHS = {
    "synthetic_1mm": "data/synthetic/100k_samples_txp_1551.5_pax_1552.5_polcon_and_fiber_1Hz.mat",
    **{
        f"synthetic_{delta}mm": f"data/synthetic/400k_samples_txp_1551.5_pax_{wavelength}_polcon_and_fiber_2_1Hz.mat"
        for delta, wavelength in [(-5, "1546.5"), (5, "1556.5"), (10, "1561.5"), (14, "1565.5")]
    },
    **{
        f"loop_{delta}mm": f"data/chicago_loop/txp_1551.5_pax_{wavelength}_fiber_loop.mat"
        for delta, wavelength in [(1, "1552.5"), (5, "1556.5"), (10, "1561.5"), (14, "1565.5")]
    },
}
WAVELENGTHS = {name: float(name.split("_")[1][:-2]) for name in DATASET_PATHS}


def load_stokes(path):
    path = Path(path)
    with path.open("rb") as stream:
        if stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"{path} is a Git LFS pointer. Install Git LFS, then run 'git lfs pull' from the repository.")
    data = scipy.io.loadmat(path)
    names = [f"s{i}_{instrument}" for instrument in ("txp", "pax") for i in range(1, 4)]
    missing = [name for name in names if name not in data]
    if missing:
        raise ValueError(f"{path} is missing Stokes arrays: {', '.join(missing)}")
    arrays = [np.asarray(data[name]).reshape(-1) for name in names]
    if len({len(a) for a in arrays}) != 1 or not len(arrays[0]):
        raise ValueError("All six Stokes arrays must have the same nonzero length")
    values = np.column_stack(arrays).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Stokes arrays contain NaN or infinite values")
    return values[:, :3], values[:, 3:]

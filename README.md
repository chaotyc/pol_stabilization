# Mamba FiO Polarization Stabilization

Predict the next target (PAX) Stokes vector from a window of control (TxP) Stokes vectors. The default model, `PolarizationMambaSO3`, processes consecutive differences with Mamba, predicts a rotation, and applies it to the latest normalized control vector. `PolarizationMambaSO3Combined` feeds both absolute coordinates and consecutive differences into Mamba with the same rotation output. Both predict one time step ahead. The original direct-output `PolarizationMamba` remains available for legacy checkpoints.

## Setup

Use Python 3.10+ and run commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

CPU training and inference use the included PyTorch Mamba implementation on every operating system. On Linux with a CUDA device, `--backend auto` uses `mamba-ssm` if installed and otherwise falls back to PyTorch. For optional optimized kernels, install `mamba-ssm` in an environment compatible with your PyTorch/CUDA versions. `--backend pytorch` forces the portable implementation; `--backend cuda` requires the optimized kernels. `--device cpu` explicitly selects CPU even on a GPU machine.

The MATLAB datasets are stored in **Git LFS**. Install Git LFS for your platform, then download the actual data:

```bash
git lfs install --local
git lfs pull
```

Each `.mat` file must contain equally sized, synchronized arrays `s1_txp`, `s2_txp`, `s3_txp`, `s1_pax`, `s2_pax`, and `s3_pax`. The loader detects unresolved LFS pointers and reports how to fetch them. Dataset names retain the historical `mm` suffix in CLI identifiers; wavelength separations in the experiments are measured in **nm**.

## Training and artifacts

```bash
python -m src.training.mamba_training \
  --wavelength-range loop_5mm \
  --window-size 128 --dim 32 --layers 3 \
  --epochs 20 --batch-size 64 --lr 0.00001 \
  --loss infidelity --run-id loop5_baseline
```

Run `python -m src.training.mamba_training --help` for all arguments. Defaults are window size 10, dimension 32, three layers, 20 epochs, batch size 64, learning rate `1e-5`, and infidelity loss. Loss options are `mse`, `regmse`, `angular`, and `infidelity` (case insensitive). SO3 windows must contain at least two samples. Use `--data-path` for an explicit MATLAB file, `--max-samples` to change the default cap of 300,000 samples, and `--seed` to change the default seed of 42.

Train the version with absolute coordinates and differences using:

```bash
python -m src.training.mamba_training \
  --model-type PolarizationMambaSO3Combined \
  --wavelength-range loop_5mm --run-id loop5_combined
```

For each observed step after the first, the combined model builds six features: `[s1(t), s2(t), s3(t), s1(t)-s1(t-1), s2(t)-s2(t-1), s3(t)-s3(t-1)]`. A window of `W` observations produces `W-1` feature vectors. Training and inference still accept raw three-coordinate windows; feature construction happens inside the model. Checkpoints record the model type so inference selects the correct version automatically.

`--model-type PolarizationMambaSO3CombinedTime` adds a scalar timestamp to the rotation head after the Mamba encoder. Time bypasses differencing and is scaled using only the training span. In normal training, timestamps use global sample indices, or seconds if `--sample-interval` is supplied. For inference with this model, pass `predictor.predict(window, target_time=...)` using the same origin and units as training; the predictor applies the saved normalization. Time continues beyond the training range without clipping.

Splits are chronological: 70% training, 10% validation, 20% test. Windows stay within each split. The best validation checkpoint is evaluated on the test split after training. Normalized RMSE projects predictions to unit length; fidelity follows `(1 + predicted_unit_vector · target_unit_vector) / 2`. Both directions are normalized for fidelity because measured vector lengths depart slightly from one; raw MSE retains the measured lengths.

Every invocation writes to `results/runs/<dataset>/<run-id>/`:

- `best_model.pt`: weights, architecture, window size, training settings, split boundaries, and best validation loss.
- `metrics.json`: raw and normalized test errors, fidelity, and the checkpoint path.
- `convergence.png` and `predictions.png`: plots for that run.

Omitting `--run-id` generates a unique ID. Reusing an existing ID fails rather than overwrite its artifacts. `--output-dir` changes the root directory. Plots use sample indices unless the acquisition interval is supplied with `--sample-interval SECONDS`.

## Inference

```python
from src.inference.inference import PolarizationPredictor

predictor = PolarizationPredictor(
    "results/runs/loop_5mm/loop5_baseline/best_model.pt",
    device="cpu",
)
# window has shape (128, 3); batches have shape (batch, 128, 3)
prediction = predictor.predict(window)
```

The committed `results/best_model_MAMBA.pt` is a legacy weights-only checkpoint. Its original window size is not recorded and cannot be recovered from its tensors. To load it, supply `legacy_config={"model_type": "PolarizationMambaSO3", "window_size": original_window_size}`. Use `"PolarizationMamba"` for checkpoints trained with the original direct-output model. These two architectures have matching tensor shapes but different behavior, so the loader requires an explicit legacy model type.

Measure per-window latency on CPU and, when available, CUDA:

```bash
python -m src.inference.inference_test \
  results/runs/loop_5mm/loop5_baseline/best_model.pt \
  data/chicago_loop/txp_1551.5_pax_1556.5_fiber_loop.mat
```

## Experiments and hyperparameter optimization

```bash
python -m src.evaluation.window_size_experiment
python -m src.evaluation.delta_lambda_experiment
python -m src.evaluation.run_all_datasets
python -m src.evaluation.performance_comparison --wavelength-range loop_5mm
```

The comparison trains the selected SO3 Mamba and the repository's Transformer baseline with the same data splits, sample cap, objective, optimizer, learning-rate schedule, batch order, and float32 precision. Pass `--model-type PolarizationMambaSO3Combined` to `src.evaluation.performance_comparison` to compare the combined model; results use the key `MambaSO3Combined`. The baseline attention uses PyTorch scaled dot-product attention. Results record model sizes, epochs run, training time, and synchronized forward timing per window. Forward timings exclude data transfers; the separate inference benchmark includes input preparation and returning predictions to CPU. These are comparisons of these implementations, not universal architecture speed claims.

Create a reproducible Optuna study and evaluate its best validation-selected configuration:

```bash
python -m src.evaluation.hpo \
  --wavelength-range loop_5mm --loss infidelity --n-trials 50 --epochs 100
python -m src.evaluation.eval_best \
  --datasets loop_5mm --loss infidelity --epochs 100
```

Study names include dataset and loss, such as `mamba_loop_5mm_infidelity`. Use `--db` to change `sqlite:///optuna_study.db`. HPO passes `--validation-only`, so trials do not evaluate the test split. Match the HPO dataset override, seed, and sample cap during final evaluation. Best parameters are exported under `results/hpo/`; each trial and final run retains its own checkpoint. The historical HPO database and full trial history were not committed and cannot be reconstructed from the single retained log; rerun HPO to obtain a new study. Previously reported results need to be regenerated after these fixes.

## Temporal drift experiment

Compare the three six-feature methods on the long 5 nm Chicago loop recording:

```bash
python -m src.evaluation.temporal_drift_experiment \
  --data-path data/chicago_loop/txp_1551.5_pax_1556.5_fiber_loop.mat \
  --train-samples 300000 --total-samples 2000000 \
  --run-id chicago_2m_temporal
```

This experiment uses a different split from normal training. Both initial models fit all of the first 300,000 samples for a fixed 20 epochs, with no test-based checkpoint selection. Defaults are window size 10, model dimension 32, three layers, batch size 256, learning rate `1e-4`, and infidelity loss. The frozen and time-conditioned models start with matching backbone/head weights and use the same minibatch order. The time coefficient starts at zero. The adaptive model begins as an exact copy of the trained frozen model.

Every target from zero-based index 300,000 through 1,999,999 is predicted from its preceding TxP window. Windows retain past context across the train/test boundary and evaluation blocks, so each method scores exactly 1.7 million targets. Test-time policies are:

- **Six features, frozen:** no weight updates after initial training.
- **Six features + time:** a timestamp enters the rotation head alongside Mamba's representation; weights stay fixed after initial training. Without an acquisition interval, time is the global sample index divided by 299,999. Supplying `--sample-interval SECONDS` gives equivalent elapsed-time conditioning and plots in hours.
- **Six features + periodic adaptation:** after every 10,000 scored test targets, fine-tune only the rotation head for 20 steps at learning rate `1e-4`. Each batch draws approximately 80% of its samples from the last 20,000 observed test targets, favoring recent observations exponentially, and 20% from the initial training prefix. No future targets enter updates. The frozen backbone stays in evaluation mode. This policy assumes PAX feedback is available after a prediction is scored.

Performance is measured separately in successive 10,000-target blocks. Adaptation frequency and reporting frequency can differ: the runner splits prediction batches at update boundaries so that all three methods score each target before its label can be used for adaptation. There is no unused gap between blocks. The final block does not trigger an update because there are no remaining predictions to evaluate.

Artifacts appear under `results/runs/temporal_drift/<run-id>/`:

- `protocol.json`: source file, index boundaries, time normalization, seeds, hyperparameters, and feedback policy.
- `training.csv`: loss and runtime for every initial training epoch.
- `metrics_by_time.csv`: mean, median, and 95th percentile angular error, fidelity, raw and normalized errors, and prediction runtime per method/block. Indices are zero-based and end-exclusive; a final partial block is included if needed.
- `performance_over_time.png` and `.svg`: mean/p95 angular error, fidelity, and normalized RMSE plotted over the recording. The shaded prefix marks initial training.
- `adaptation.jsonl`: every update boundary, observed-label range, sampled-label limits, update loss, and runtime.
- `six_features_initial.pt`, `six_features_time_initial.pt`, and `six_features_adaptive_final.pt`: loadable checkpoints with architecture and time metadata.
- `summary.json`: sample-weighted aggregate errors, first/last-block angular error, and adaptation overhead. The adaptive score measures prediction before feedback updates, rather than rescoring data it has just learned.

Use `--epochs 1` for a pilot with the same data coverage. `--metric-block-size`, `--adapt-every`, `--adapt-window`, `--adapt-steps`, `--adapt-lr`, and `--replay-fraction` control reporting and adaptation. Freeze these choices before examining the test curves. Use distinct `--run-id` values for repeat experiments; the runner refuses to overwrite an existing run.

## Checks

```bash
MPLBACKEND=Agg python -m unittest discover -s tests -v
```

Tests cover loss geometry, CPU/backend selection, checkpoint loading, window validation, artifact isolation, small training/inference runs, normalized metrics, model comparisons, and HPO/final-evaluation integration. Temporal tests verify global timestamps, extrapolation beyond the training time range, matching initial predictions, head-only adaptation, future-label isolation, prediction-before-update ordering, and complete test coverage across nonaligned reporting/update boundaries. Test fixtures are generated separately from the research datasets.

## References

- Ely Eastman et al., *Direct Tracking of State-of-Polarization Fluctuations in Fiber due to Polarization Mode Dispersion Using Machine Learning*.
- Gu, A., & Dao, T. (2023), *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*.

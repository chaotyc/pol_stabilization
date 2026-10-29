import contextlib
import copy
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import scipy.io
import torch

from src.evaluation import temporal_drift_experiment as experiment
from src.inference.inference import PolarizationPredictor
from src.model.mamba import PolarizationMambaSO3Combined, PolarizationMambaSO3CombinedTime
from src.training.args import parse_args
from src.training.data import SParameterDataset
from src.training.mamba_training import train


torch.set_num_threads(1)


class TemporalDriftTests(unittest.TestCase):
    def fixture(self, directory, count=60):
        t = np.arange(count) * 0.1
        features = np.column_stack((np.cos(t), np.sin(t), np.zeros(count))).astype(np.float32)
        target_t = t + np.arange(count) * 0.005
        targets = np.column_stack((np.cos(target_t), np.sin(target_t), np.zeros(count))).astype(np.float32)
        path = Path(directory) / "drifting.mat"
        scipy.io.savemat(path, {f"s{i + 1}_{instrument}": array[:, i]
                               for instrument, array in (("txp", features), ("pax", targets)) for i in range(3)})
        return path, features, targets

    def options(self, path, directory):
        return ["--data-path", str(path), "--output-dir", str(directory), "--run-id", "temporal_test",
                "--total-samples", "52", "--train-samples", "20", "--epochs", "1", "--window-size", "3",
                "--dim", "8", "--layers", "1", "--batch-size", "8", "--eval-batch-size", "8",
                "--metric-block-size", "7", "--adapt-every", "5", "--adapt-window", "10",
                "--adapt-steps", "2", "--adapt-batch-size", "8", "--device", "cpu", "--backend", "pytorch"]

    def test_defaults_use_requested_prefix_and_training_boundary(self):
        args = experiment.parse_experiment_args([])
        self.assertEqual((args.train_samples, args.total_samples), (300_000, 2_000_000))
        self.assertEqual(Path(args.data_path).name, "txp_1551.5_pax_1556.5_fiber_loop.mat")

    def test_time_coordinates_are_global_and_do_not_reset_at_split(self):
        values = np.arange(30, dtype=np.float32).reshape(10, 3)
        dataset = SParameterDataset(values[5:], values[5:], 2, start_index=5, include_time=True, time_step=0.25)
        (x, timestamp), y = dataset[0]
        np.testing.assert_array_equal(x.numpy(), values[5:7])
        np.testing.assert_array_equal(y.numpy()[0], values[7])
        self.assertEqual(timestamp.item(), 7 * 0.25)

    def test_time_conditions_head_and_preserves_baseline_initialization(self):
        torch.manual_seed(42)
        baseline = PolarizationMambaSO3Combined(3, 8, 1, backend="pytorch")
        torch.manual_seed(42)
        timed = PolarizationMambaSO3CombinedTime(3, 8, 1, backend="pytorch", time_scale=19)
        x = torch.nn.functional.normalize(torch.randn(2, 4, 3), dim=-1)
        torch.testing.assert_close(baseline(x), timed(x, torch.tensor([19., 38.])), rtol=1e-6, atol=1e-6)
        seen = []
        hook = timed.head.register_forward_pre_hook(lambda _module, args: seen.append(args[0].detach().clone()))
        try:
            timed(x, torch.tensor([19., 38.]))
        finally:
            hook.remove()
        torch.testing.assert_close(seen[0][:, -1], torch.tensor([1., 2.]))
        with torch.no_grad():
            timed.head.weight[:, -1] = torch.tensor([0.1, 0.2, 0.3])
        self.assertFalse(torch.allclose(timed(x, 19), timed(x, 38)))
        torch.testing.assert_close(timed(x, 38).norm(dim=-1), torch.ones(2, 1), atol=1e-5, rtol=1e-5)
        for invalid in (torch.tensor([1., 2., 3.]), torch.tensor([float("nan"), 1.])):
            with self.assertRaisesRegex(ValueError, "target_time"):
                timed(x, invalid)

    def test_timed_checkpoints_round_trip_with_original_time_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            model = PolarizationMambaSO3CombinedTime(3, 8, 1, backend="pytorch", time_origin=10, time_scale=25).eval()
            x = torch.randn(2, 4, 3)
            times = np.array([40., 80.])
            expected = model(x, times).detach().numpy()[:, 0]
            config = {"model_type": type(model).__name__, "input_dim": 3, "d_model": 8, "n_layers": 1,
                      "window_size": 4, "time_origin": 10, "time_scale": 25}
            for legacy in (False, True):
                path = Path(directory) / "model.pt"
                torch.save(model.state_dict() if legacy else {"state_dict": model.state_dict(), "config": config}, path)
                predictor = PolarizationPredictor(path, device="cpu", legacy_config=config if legacy else None)
                with self.assertRaisesRegex(ValueError, "target_time"):
                    predictor.predict(x)
                np.testing.assert_allclose(predictor.predict(x, target_time=times), expected, atol=1e-6)
                self.assertEqual(float(predictor.model.time_scale), 25)

    def test_adaptation_changes_only_head_and_ignores_unobserved_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            path, features, targets = self.fixture(directory)
            args = experiment.parse_experiment_args(self.options(path, directory))
            torch.manual_seed(args.seed)
            initial = PolarizationMambaSO3Combined(3, 8, 1, backend="pytorch")
            changed_future = targets.copy()
            changed_future[30:] *= -1
            adapters = [experiment.PeriodicHeadAdapter(copy.deepcopy(initial),
                        SParameterDataset(features, y, args.window_size), args, torch.device("cpu"))
                        for y in (targets, changed_future)]
            for adapter in adapters:
                log = adapter.update(30)
                self.assertLess(log["max_target_used"], 30)
                self.assertEqual(log["recent_end_exclusive"], 30)
                self.assertGreater(log["replay_samples_per_step"], 0)
                for key, before in initial.state_dict().items():
                    if not key.startswith("head."):
                        torch.testing.assert_close(before, adapter.model.state_dict()[key], rtol=0, atol=0)
            for key, value in adapters[0].model.state_dict().items():
                torch.testing.assert_close(value, adapters[1].model.state_dict()[key], rtol=0, atol=0)
            self.assertFalse(torch.equal(initial.head.weight, adapters[0].model.head.weight))

    def test_timestamp_inference_benchmark_uses_recording_time_units(self):
        from src.inference.inference_test import run_benchmark
        with tempfile.TemporaryDirectory() as directory:
            model = PolarizationMambaSO3CombinedTime(3, 8, 1, backend="pytorch", time_scale=25)
            path = Path(directory) / "seconds.pt"
            config = {"model_type": type(model).__name__, "input_dim": 3, "d_model": 8,
                      "n_layers": 1, "window_size": 3, "time_units": "seconds"}
            torch.save({"state_dict": model.state_dict(), "config": config}, path)
            predictor = PolarizationPredictor(path, device="cpu")
            with self.assertRaisesRegex(ValueError, "sample-interval"):
                run_benchmark(predictor, np.ones((5, 3)), 3, max_windows=1)
            with contextlib.redirect_stdout(io.StringIO()):
                times = run_benchmark(predictor, np.ones((5, 3)), 3, max_windows=1, sample_interval=0.1)
            self.assertEqual(len(times), 1)

    def test_angles_are_correct_and_inputs_are_preserved(self):
        p = np.array([[1., 0., 0.], [1., 0., 0.], [1., 0., 0.]])
        y = np.array([[1., 0., 0.], [0., 1., 0.], [-1., 0., 0.]])
        metrics = experiment.temporal_metrics(p, y)
        self.assertAlmostEqual(metrics["mean_angular_error_deg"], 90)
        self.assertAlmostEqual(metrics["mean_fidelity"], 0.5)
        scaled = 2 * y
        experiment.temporal_metrics(p, scaled)
        np.testing.assert_array_equal(scaled, 2 * y)

    def test_summary_weights_partial_blocks_by_their_sample_counts(self):
        rows = []
        for method in experiment.METHODS:
            for count, mse, fidelity, angle in ((3, 4, 0.5, 10), (1, 16, 1.0, 30)):
                rows.append({"method": method, "n_samples": count, "test_mse": mse, "norm_test_mse": mse,
                             "mean_fidelity": fidelity, "mean_angular_error_deg": angle, "prediction_seconds": 1.0})
        summary = experiment.summarize(rows, [], experiment.parse_experiment_args([]))
        for method in experiment.METHODS:
            self.assertAlmostEqual(summary[method]["mean_angular_error_deg"], 15)
            self.assertAlmostEqual(summary[method]["test_rmse"], np.sqrt(7))
            self.assertAlmostEqual(summary[method]["mean_fidelity"], 0.625)

    def test_full_small_experiment_scores_every_target_before_updating(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, _ = self.fixture(directory)
            args = self.options(path, directory)
            events, first_predictions = [], {}
            original_predict = experiment.predict_range
            original_update = experiment.PeriodicHeadAdapter.update

            def record_predict(model, dataset, start, end, *rest):
                result = original_predict(model, dataset, start, end, *rest)
                events.append(("predict", start, end))
                if start == 20:
                    first_predictions[len(first_predictions)] = result.copy()
                return result

            def record_update(adapter, end):
                self.assertTrue(all(event[0] == "predict" and event[2] == end for event in events[-3:]))
                events.append(("update", end, end))
                return original_update(adapter, end)

            with contextlib.redirect_stdout(io.StringIO()), patch.object(experiment, "predict_range", side_effect=record_predict), \
                    patch.object(experiment.PeriodicHeadAdapter, "update", record_update):
                summary = experiment.main(args)
            # Frozen and adaptive models make identical predictions before feedback is used.
            np.testing.assert_allclose(first_predictions[0], first_predictions[2], rtol=1e-6, atol=1e-6)
            run_dir = Path(summary["run_dir"])
            with (run_dir / "metrics_by_time.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            for method in experiment.METHODS:
                selected = [row for row in rows if row["method"] == method]
                self.assertEqual(len(selected), 5)
                self.assertEqual(sum(int(row["n_samples"]) for row in selected), 32)
                self.assertEqual(int(selected[0]["start_index"]), 20)
                self.assertEqual(int(selected[-1]["end_index_exclusive"]), 52)
                self.assertEqual(summary[method]["test_samples"], 32)
                for before, after in zip(selected, selected[1:]):
                    self.assertEqual(before["end_index_exclusive"], after["start_index"])
            updates = [json.loads(line) for line in (run_dir / "adaptation.jsonl").read_text().splitlines()]
            self.assertEqual([row["observed_end_exclusive"] for row in updates], [25, 30, 35, 40, 45, 50])
            self.assertTrue(all(row["max_target_used"] < row["observed_end_exclusive"] for row in updates))
            for method in ("six_features", "six_features_time"):
                checkpoint = torch.load(run_dir / f"{method}_initial.pt", weights_only=True)
                self.assertEqual(checkpoint["trained_through_index_exclusive"], 20)
            self.assertGreater((run_dir / "performance_over_time.png").stat().st_size, 0)
            self.assertTrue((run_dir / "performance_over_time.svg").is_file())

    def test_generic_training_and_comparison_accept_timestamps(self):
        from src.evaluation.performance_comparison import main as compare
        with tempfile.TemporaryDirectory() as directory:
            path, features, _ = self.fixture(directory, count=100)
            options = ["--model-type", "PolarizationMambaSO3CombinedTime", "--data-path", str(path),
                       "--output-dir", directory, "--epochs", "1", "--dim", "8", "--layers", "1",
                       "--window-size", "2", "--batch-size", "16", "--device", "cpu", "--backend", "pytorch"]
            with contextlib.redirect_stdout(io.StringIO()):
                metrics = train(parse_args(options + ["--run-id", "time_training"]))
                results = compare(options + ["--run-id", "time_comparison"])
            predictor = PolarizationPredictor(metrics["checkpoint_path"], device="cpu")
            self.assertEqual(predictor.predict(features[:2], target_time=2).shape, (3,))
            self.assertEqual(metrics["config"]["time_scale"], 69)
            self.assertTrue(np.isfinite(results["MambaSO3CombinedTime"]["mean_fidelity"]))


if __name__ == "__main__":
    unittest.main()

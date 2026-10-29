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

from src.inference.inference import PolarizationPredictor
from src.model.loss import PoincareRegularizedMSE, Infidelity
from src.model.mamba import MambaBlock, PolarizationMamba, PolarizationMambaSO3, PolarizationMambaSO3Combined
from src.training.args import parse_args
from src.training.data import SParameterDataset
from src.training.mamba_training import train
from src.utils.data import load_stokes
from src.utils.metrics import prediction_metrics
from src.utils.runs import create_run_directory


torch.set_num_threads(1)


class ModelRegressionTests(unittest.TestCase):
    def test_regularization_is_zero_for_perfect_unit_vectors(self):
        vectors = torch.tensor([[1., 0., 0.], [1., 1., 1.]])
        vectors = torch.nn.functional.normalize(vectors, dim=-1)
        for x in (vectors, vectors[:, None, :]):
            self.assertAlmostEqual(PoincareRegularizedMSE()(x, x).item(), 0.0, places=6)
        x = vectors[:, None, :] * 2
        self.assertAlmostEqual(PoincareRegularizedMSE()(x, x).item(), 0.2, places=6)

    def test_fidelity_is_bounded_for_measured_nonunit_vectors(self):
        pred, target = torch.tensor([[2., 0., 0.]]), torch.tensor([[1.02, 0., 0.]])
        self.assertAlmostEqual(Infidelity()(pred, target).item(), 0)
        self.assertAlmostEqual(Infidelity()(pred, -target).item(), 1)
        self.assertAlmostEqual(prediction_metrics(pred.numpy(), target.numpy())["mean_fidelity"], 1)

    def test_linux_cpu_falls_back_without_importing_cuda_package(self):
        for model_type in (PolarizationMamba, PolarizationMambaSO3, PolarizationMambaSO3Combined):
            model = model_type(3, 8, 1, system="Linux", device="cpu")
            self.assertIsInstance(model.layers[0], MambaBlock)
            output = model(torch.randn(2, 4, 3))
            self.assertEqual(tuple(output.shape), (2, 1, 3))
            output.sum().backward()

    def test_missing_optional_package_falls_back_on_cuda_target(self):
        with patch("torch.cuda.is_available", return_value=True), patch.dict("sys.modules", {"mamba_ssm": None}):
            with self.assertWarns(RuntimeWarning):
                model = PolarizationMambaSO3(3, 8, 1, system="Linux", device="cuda")
            self.assertEqual(model.backend, "pytorch")
            with self.assertRaisesRegex(RuntimeError, "mamba-ssm"):
                PolarizationMambaSO3(3, 8, 1, system="Linux", device="cuda", backend="cuda")

    def test_so3_normalization_and_short_window_error(self):
        for cls in (PolarizationMambaSO3, PolarizationMambaSO3Combined):
            with self.subTest(model=cls.__name__):
                model = cls(3, 8, 1, backend="pytorch")
                result = model(torch.randn(3, 2, 3))
                torch.testing.assert_close(result.norm(dim=-1), torch.ones(3, 1), atol=1e-5, rtol=1e-5)
                with self.assertRaisesRegex(ValueError, "window_size"):
                    model(torch.randn(2, 1, 3))
        with self.assertRaises(SystemExit):
            parse_args(["--window-size", "1"])

    def test_combined_encoder_receives_coordinates_and_causal_differences(self):
        model = PolarizationMambaSO3Combined(3, 8, 1, backend="pytorch")
        # The model receives observations only; the next target is outside this window.
        x = torch.tensor([[[1., 2., 3.], [3., 1., 4.], [2., 5., 6.]]])
        seen = []
        hook = model.embedding.register_forward_pre_hook(lambda _module, args: seen.append(args[0].detach().clone()))
        try:
            model(x)
            model(x + 10)
        finally:
            hook.remove()
        expected = torch.tensor([[[3., 1., 4., 2., -1., 1.], [2., 5., 6., -1., 4., 2.]]])
        torch.testing.assert_close(seen[0], expected)
        # Equal motion at different absolute positions must remain distinguishable.
        torch.testing.assert_close(seen[1][..., :3], seen[0][..., :3] + 10)
        torch.testing.assert_close(seen[1][..., 3:], seen[0][..., 3:])

    def test_modern_and_legacy_checkpoints_preserve_model_behavior(self):
        with tempfile.TemporaryDirectory() as folder:
            for cls in (PolarizationMamba, PolarizationMambaSO3, PolarizationMambaSO3Combined):
                model = cls(3, 8, 1, backend="pytorch").eval()
                x = torch.randn(2, 4, 3)
                expected = model(x).detach().numpy()[:, 0, :]
                config = {"model_type": cls.__name__, "input_dim": 3, "d_model": 8, "n_layers": 1, "window_size": 4}
                for legacy in (False, True):
                    path = Path(folder) / "model.pt"
                    torch.save(model.state_dict() if legacy else {"config": config, "state_dict": model.state_dict()}, path)
                    if legacy:
                        with self.assertRaisesRegex(ValueError, "legacy_config"):
                            PolarizationPredictor(path, device="cpu")
                    predictor = PolarizationPredictor(path, device="cpu", legacy_config=config if legacy else None)
                    np.testing.assert_allclose(predictor.predict(x), expected, atol=1e-6)
                    self.assertEqual(predictor.predict(x[:1]).shape, (3,))
                    with self.assertRaises(ValueError):
                        predictor.predict([1., 0., 0.])
                    if cls is not PolarizationMamba:
                        short_config = {**config, "window_size": 1}
                        torch.save({"config": short_config, "state_dict": model.state_dict()}, path)
                        with self.assertRaisesRegex(ValueError, "window_size"):
                            PolarizationPredictor(path, device="cpu")

    def test_bundled_legacy_checkpoint_loads_with_explicit_configuration(self):
        path = Path(__file__).resolve().parents[1] / "results/best_model_MAMBA.pt"
        predictor = PolarizationPredictor(path, device="cpu", legacy_config={"model_type": "PolarizationMambaSO3", "window_size": 4})
        self.assertEqual(predictor.predict(np.ones((4, 3))).shape, (3,))


class PipelineRegressionTests(unittest.TestCase):
    def make_data(self, directory):
        path = Path(directory) / "fixture.mat"
        t = np.arange(100) * 0.1
        x = np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)])
        scipy.io.savemat(path, {f"s{i+1}_{instrument}": x[:, i] for instrument in ("txp", "pax") for i in range(3)})
        return path

    def test_lfs_pointer_gives_actionable_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.mat"
            path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:example\nsize 100\n")
            with self.assertRaisesRegex(ValueError, "git lfs pull"):
                load_stokes(path)

    def test_windows_predict_next_target_without_crossing_split(self):
        values = np.arange(30).reshape(10, 3)
        dataset = SParameterDataset(values, values, 2)
        x, y = dataset[0]
        np.testing.assert_array_equal(x.numpy(), values[:2])
        np.testing.assert_array_equal(y.numpy()[0], values[2])
        self.assertEqual(len(dataset), 8)

    def test_run_ids_prevent_checkpoint_replacement(self):
        with tempfile.TemporaryDirectory() as folder:
            first = create_run_directory(folder, "loop_5mm", "run_1")
            second = create_run_directory(folder, "loop_5mm", "run_2")
            self.assertNotEqual(first, second)
            with self.assertRaises(FileExistsError):
                create_run_directory(folder, "loop_5mm", "run_1")
            self.assertNotEqual(create_run_directory(folder, "loop_5mm"), create_run_directory(folder, "loop_5mm"))

    def test_training_inference_and_evaluation_schema(self):
        from src.evaluation.eval_best import plot_results
        with tempfile.TemporaryDirectory() as folder:
            path = self.make_data(folder)
            args = parse_args(["--data-path", str(path), "--output-dir", folder, "--run-id", "smoke",
                               "--epochs", "1", "--dim", "8", "--layers", "1", "--window-size", "2", "--batch-size", "16", "--device", "cpu"])
            metrics = train(args)
            self.assertTrue(np.isfinite(metrics["norm_test_rmse"]))
            predictor = PolarizationPredictor(metrics["checkpoint_path"], device="cpu")
            features, _ = load_stokes(path)
            self.assertEqual(predictor.predict(features[:2]).shape, (3,))
            metrics["delta_nm"] = 5
            plot_results({"loop_5mm": metrics}, folder)
            args.validation_only, args.run_id = True, "validation"
            with patch("src.training.mamba_training.prediction_metrics", side_effect=AssertionError("Test set evaluated during HPO")):
                validation = train(args)
            self.assertIn("best_val_loss", validation)
            self.assertNotIn("test_mse", validation)
            with open(Path(folder) / "runs/loop_5mm/validation/metrics.json") as stream:
                self.assertNotIn("test_mse", json.load(stream))

    def test_comparison_runs_current_so3_and_transformer(self):
        from src.evaluation.performance_comparison import main
        with tempfile.TemporaryDirectory() as folder:
            path = self.make_data(folder)
            results = main(["--data-path", str(path), "--output-dir", folder, "--epochs", "1", "--dim", "8", "--layers", "1", "--window-size", "2", "--batch-size", "16", "--device", "cpu"])
            self.assertEqual(results["MambaSO3"]["config"]["model_type"], "PolarizationMambaSO3")
            for name in ("MambaSO3", "Transformer"):
                self.assertGreater(results[name]["windows_per_second"], 0)
                self.assertIn("norm_test_rmse", results[name])
            self.assertEqual(results["MambaSO3"]["test_windows"], results["Transformer"]["test_windows"])

    def test_combined_training_checkpoint_and_comparison(self):
        from src.evaluation.performance_comparison import main as compare
        with tempfile.TemporaryDirectory() as folder:
            path = self.make_data(folder)
            options = ["--model-type", "PolarizationMambaSO3Combined", "--data-path", str(path),
                       "--output-dir", folder, "--epochs", "1", "--dim", "8", "--layers", "1",
                       "--window-size", "2", "--batch-size", "16", "--device", "cpu", "--backend", "pytorch"]
            metrics = train(parse_args(options + ["--run-id", "combined"]))
            self.assertEqual(metrics["config"]["model_type"], "PolarizationMambaSO3Combined")
            self.assertEqual(metrics["config"]["input_dim"], 3)
            self.assertTrue(np.isfinite(metrics["norm_test_rmse"]))
            predictor = PolarizationPredictor(metrics["checkpoint_path"], device="cpu")
            self.assertIsInstance(predictor.model, PolarizationMambaSO3Combined)
            features, _ = load_stokes(path)
            self.assertEqual(predictor.predict(features[:2]).shape, (3,))
            results = compare(options + ["--run-id", "combined_comparison"])
            self.assertEqual(results["MambaSO3Combined"]["config"]["model_type"], "PolarizationMambaSO3Combined")
            self.assertEqual(results["MambaSO3Combined"]["test_windows"], results["Transformer"]["test_windows"])
            self.assertTrue(np.isfinite(results["MambaSO3Combined"]["norm_test_rmse"]))

    def test_hpo_and_final_evaluation_share_protocol_and_use_current_artifacts(self):
        from src.evaluation.hpo import main as optimize
        from src.evaluation.eval_best import main as evaluate
        with tempfile.TemporaryDirectory() as folder:
            db = "sqlite:///" + str(Path(folder) / "study.db")
            args = ["--db", db, "--output-dir", folder, "--epochs", "1"]
            with patch("src.evaluation.hpo.run_training", return_value={"best_val_loss": 0.1, "checkpoint_path": "trial.pt"}) as runner:
                optimize(args + ["--n-trials", "1"])
                self.assertTrue(runner.call_args.kwargs["validation_only"])
            metrics = {**prediction_metrics(np.array([[1., 0., 0.]]), np.array([[1., 0., 0.]])), "best_val_loss": 0.1}
            with patch("src.evaluation.eval_best.run_training", return_value=metrics), patch("src.evaluation.eval_best.plot_results") as plot:
                result = evaluate(args + ["--datasets", "loop_5mm"])
                self.assertEqual(result["loop_5mm"]["norm_test_rmse"], 0)
                self.assertEqual(plot.call_args.args[0], result)

    def test_normalized_metrics_match_expected_values(self):
        result = prediction_metrics(np.array([[2., 0., 0.]]), np.array([[1., 0., 0.]]))
        self.assertAlmostEqual(result["test_mse"], 1/3)
        self.assertAlmostEqual(result["norm_test_rmse"], 0)
        self.assertAlmostEqual(result["mean_fidelity"], 1)


if __name__ == "__main__":
    unittest.main()

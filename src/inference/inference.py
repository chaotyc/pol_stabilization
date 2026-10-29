from collections.abc import Mapping

import torch

from src.model.mamba import PolarizationMamba, SO3_MODEL_TYPES

MODEL_TYPES = {
    "PolarizationMamba": PolarizationMamba,
    **SO3_MODEL_TYPES,
}


class PolarizationPredictor:
    def __init__(self, model_path, device=None, *, legacy_config=None, backend="auto"):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            config = dict(checkpoint["config"])
        else:
            # Architecture and window size cannot be recovered from tensor shapes.
            if not legacy_config or not {"model_type", "window_size"} <= legacy_config.keys():
                raise ValueError("Legacy checkpoint requires legacy_config with the original model_type and window_size")
            if not isinstance(checkpoint, Mapping) or "embedding.weight" not in checkpoint:
                raise ValueError("Checkpoint is not a supported model state dictionary")
            state_dict = checkpoint
            d_model, feature_dim = state_dict["embedding.weight"].shape
            model_class = MODEL_TYPES.get(legacy_config["model_type"])
            input_dim = feature_dim // getattr(model_class, "feature_multiplier", 1)
            layer_indices = {int(k.split(".")[1]) for k in state_dict if k.startswith("layers.")}
            config = {"input_dim": input_dim, "d_model": d_model, "n_layers": len(layer_indices), **legacy_config}
        model_type = config.get("model_type")
        if model_type not in MODEL_TYPES:
            raise ValueError(f"Unsupported model_type: {model_type}")
        self.window_size = int(config["window_size"])
        minimum = 2 if model_type in SO3_MODEL_TYPES else 1
        if self.window_size < minimum:
            raise ValueError(f"{model_type} requires window_size >= {minimum}")
        self.config = config
        model_class = MODEL_TYPES[model_type]
        time_config = {key: config[key] for key in ("time_origin", "time_scale") if key in config}
        self.model = model_class(
            input_dim=config["input_dim"], d_model=config["d_model"], n_layers=config["n_layers"],
            device=self.device, backend=backend, **time_config,
        ).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _prepare_input(self, data):
        x = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        if x.ndim not in (2, 3) or x.shape[-2:] != (self.window_size, 3):
            raise ValueError(f"Data must have shape ({self.window_size}, 3) or (batch, {self.window_size}, 3)")
        return x.unsqueeze(0) if x.ndim == 2 else x

    def predict(self, data, *, target_time=None):
        x = self._prepare_input(data)
        needs_time = getattr(self.model, "requires_time", False)
        if needs_time and target_time is None:
            raise ValueError("This model requires target_time in the checkpoint's original time units")
        if not needs_time and target_time is not None:
            raise ValueError("This model does not accept target_time")
        with torch.inference_mode():
            pred = self.model(x, target_time) if needs_time else self.model(x)
        output = pred[:, 0, :].cpu().numpy()
        return output[0] if len(output) == 1 else output

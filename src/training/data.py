import numpy as np
import torch
from torch.utils.data import Dataset


class SParameterDataset(Dataset):
    def __init__(self, features, targets, window_size, *, start_index=0, include_time=False, time_step=1.0):
        if window_size < 2:
            raise ValueError("SO3 prediction requires window_size >= 2")
        if len(features) != len(targets) or len(features) <= window_size:
            raise ValueError("Each data split must contain more samples than the window size")
        self.features = torch.as_tensor(np.asarray(features), dtype=torch.float32)
        self.targets = torch.as_tensor(np.asarray(targets), dtype=torch.float32)
        self.window_size = window_size
        if not np.isfinite(time_step) or time_step <= 0:
            raise ValueError("time_step must be finite and positive")
        self.start_index = start_index
        self.include_time = include_time
        self.time_step = time_step

    def __len__(self):
        return len(self.features) - self.window_size

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.window_size]
        y = self.targets[idx + self.window_size].unsqueeze(0)
        if self.include_time:
            timestamp = torch.tensor((self.start_index + idx + self.window_size) * self.time_step,
                                     dtype=torch.float64)
            return (x, timestamp), y
        return x, y


def batch_inputs(inputs, device):
    """Move ordinary windows or (windows, timestamps) to the model's device."""
    if isinstance(inputs, (tuple, list)):
        return tuple(value.to(device) for value in inputs)
    return (inputs.to(device),)

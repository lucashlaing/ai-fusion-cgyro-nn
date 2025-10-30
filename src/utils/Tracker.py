import hashlib
import json
import numpy as np
import torch

class UsageTracker:
    def __init__(self):
        self.used = set()

    def _make_key(self, input_tensor: torch.Tensor) -> str:
        """Hash the input features into a short unique key.
        Only uses first 31 dims (physical parameters, not ky)."""
        # Only hash first 31 dimensions to match physical parameters
        assert input_tensor.ndim == 1, f"Expected 1D tensor, got shape {tuple(input_tensor.shape)}"
        assert input_tensor.shape[0] in (31, 32), f"Expected tensor of length 31 or 32, got {input_tensor.shape[0]}"

        arr = input_tensor[:31].detach().cpu().numpy().astype(np.float32)
        return hashlib.sha1(arr.tobytes()).hexdigest()

    def mark_used(self, input_tensor: torch.Tensor):
        """Mark a datapoint as used (after it’s actually added to training)."""
        key = self._make_key(input_tensor)
        self.used.add(key)

    def is_used(self, input_tensor: torch.Tensor) -> bool:
        key = self._make_key(input_tensor)
        return key in self.used

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(list(self.used), f)

    def load(self, path: str):
        with open(path, "r") as f:
            self.used = set(json.load(f))

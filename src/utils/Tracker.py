import hashlib
import json
import numpy as np
import torch

class UsageTracker:
    def __init__(self):
        self.used = set()

    def _make_key(self, input_tensor: torch.Tensor) -> str:
        """Hash the input features into a short unique key."""
        arr = input_tensor.detach().cpu().numpy().astype(np.float32)
        return hashlib.sha1(arr.tobytes()).hexdigest()

    def mark_used(self, input_tensor: torch.Tensor):
        """Mark a datapoint as used (after it’s actually added to training)."""
        key = self._make_key(input_tensor)
        self.used.add(key)

    def is_used(self, input_tensor: torch.Tensor) -> bool:
        key = self._make_key(input_tensor)
        return key in self.used

    def filter_unused(self, candidates: list[torch.Tensor]):
        """Return only candidates not yet marked as used."""
        return [c for c in candidates if not self.is_used(c)]

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(list(self.used), f)

    def load(self, path: str):
        with open(path, "r") as f:
            self.used = set(json.load(f))

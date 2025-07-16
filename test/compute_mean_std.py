import json
import math
from collections import defaultdict

# Load input file
input_file = "test\mean_std_with_rho.json"
output_file = "test\merged_dist.json"

with open(input_file, "r") as f:
    data = json.load(f)

merged = defaultdict(lambda: {
    "sum": 0, "sum_sq": 0, "count": 0,
    "min": float("inf"), "max": -float("inf")
})

for rho_dict in data.values():
    for var, stats in rho_dict.items():
        merged[var]["sum"] += stats["sum"]
        merged[var]["sum_sq"] += stats["sum_sq"]
        merged[var]["count"] += stats["count"]
        merged[var]["min"] = min(merged[var]["min"], stats["min"])
        merged[var]["max"] = max(merged[var]["max"], stats["max"])

# Recompute mean and std
for var, stats in merged.items():
    s = stats["sum"]
    ssq = stats["sum_sq"]
    n = stats["count"]
    mean = s / n
    var_ = ssq / n - mean**2
    std = math.sqrt(var_) if var_ > 0 else 0.0
    stats["mean"] = mean
    stats["std"] = std

with open(output_file, "w") as f:
    json.dump(merged, f, indent=2)

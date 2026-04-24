"""Try to reproduce the segfault: torch first, then load parquet."""
import sys
print("Step 1: torch", flush=True)
import torch
print("torch ok:", torch.__version__, flush=True)

print("Step 2: pandas", flush=True)
import pandas as pd
print("pd ok:", pd.__version__, flush=True)

print("Step 3: read parquet", flush=True)
df = pd.read_parquet("data/features_v1.parquet")
print("loaded shape:", df.shape, flush=True)

print("Step 4: numpy op on df", flush=True)
import numpy as np
arr = df.select_dtypes(include=[np.number]).iloc[:1000].to_numpy()
print("array shape:", arr.shape, flush=True)

print("Step 5: torch tensor from numpy", flush=True)
t = torch.from_numpy(arr.astype(np.float32))
print("tensor shape:", t.shape, flush=True)
print("DONE", flush=True)

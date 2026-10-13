"""Generate small, deterministic fixtures for the documentation tutorial."""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
weights = {
    name: rng.normal(size=(3, width)) for name, width in [("gexp", 20), ("meth", 12)]
}
for folder, count, prefix in [("input", 60, "train"), ("new-input", 12, "new")]:
    target = Path("outputs/tutorial") / folder
    target.mkdir(parents=True, exist_ok=True)
    ids = [f"{prefix}_{i:03d}" for i in range(count)]
    pd.DataFrame(
        {
            "model_id": ids,
            "model_type": [f"source_{i % 3}" for i in range(count)],
            "tissue": [f"tissue_{i % 2}" for i in range(count)],
        }
    ).to_csv(target / "metadata.csv", index=False)
    latent = rng.normal(size=(count, 3))
    for name, loading in weights.items():
        values = latent @ loading + rng.normal(
            scale=0.1, size=(count, loading.shape[1])
        )
        values[rng.random(values.shape) < 0.03] = np.nan
        pd.DataFrame(
            values.T,
            index=[f"{name}_{j}" for j in range(loading.shape[1])],
            columns=ids,
        ).to_csv(target / f"{name}.csv", index_label="feature")
print("Created training and projection tables in outputs/tutorial/")

"""Conversion utility for converting CSV datasets to MuData format.

Usage:
    mosa convert --samplesheet samplesheet.csv \
                 --view gexp_voom:data/gexp_voom.csv \
                 --view meth_combat:data/meth_combat.csv \
                 --output data.h5mu \
                 [--mutations mutations.csv]
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData
from mudata import MuData

logger = logging.getLogger(__name__)


def csv_to_mudata(
    samplesheet_path: str,
    view_specs: list[tuple[str, str]],
    output_path: str,
    mutations_path: str | None = None,
    format: str = "h5mu",
) -> None:
    """Convert CSV dataset to MuData (.h5mu or .zarr) format.

    Args:
        samplesheet_path: Path to samplesheet CSV (must contain model_id, model_type, tissue).
        view_specs: List of (view_name, csv_path) tuples. CSVs are features x samples.
        output_path: Path to save the MuData file.
        mutations_path: Optional path to mutations CSV (features x samples, binary).
        format: Output format, "h5mu" or "zarr".
    """
    import anndata

    # Enable nullable string writing for compatibility
    anndata.settings.allow_write_nullable_strings = True

    logger.info("Converting CSV dataset to MuData format")

    # 1. Load samplesheet
    logger.debug("Loading samplesheet from %s", samplesheet_path)
    samplesheet = pd.read_csv(samplesheet_path).set_index("model_id")
    # Drop any unnamed index-artifact columns (e.g. 'Unnamed: 0') that appear
    # when the source CSV was written with df.to_csv() without index=False.
    samplesheet = samplesheet.loc[:, ~samplesheet.columns.str.match(r"^Unnamed")]

    # 2. Load view CSVs (features x samples) and transpose to samples x features
    logger.debug("Loading view CSVs")
    omics: dict[str, pd.DataFrame] = {}
    for view_name, csv_path in view_specs:
        df = pd.read_csv(csv_path, index_col=0).T.astype(float)
        omics[view_name] = df
        logger.debug("  view '%s': %d samples x %d features", view_name, *df.shape)

    # 3. Find common samples across all views and samplesheet
    common = set(samplesheet.index)
    for df in omics.values():
        common &= set(df.index)

    mutations_df = None
    if mutations_path:
        mutations_df = pd.read_csv(mutations_path, index_col=0).T
        common &= set(mutations_df.index)

    common_samples = sorted(common)
    logger.debug("Common samples: %d", len(common_samples))

    if not common_samples:
        raise ValueError("No common samples found across samplesheet and view CSVs")

    # 4. Align all data to common samples
    samplesheet = samplesheet.loc[common_samples]
    for name in omics:
        omics[name] = omics[name].loc[common_samples]
    if mutations_df is not None:
        mutations_df = mutations_df.loc[common_samples]

    # 5. Create AnnData objects per modality (without .obs to avoid prefix conflicts)
    logger.debug("Creating AnnData objects")
    adatas = {}

    for view_name, df in omics.items():
        X = df.values.astype(np.float32)
        mask = ~np.isnan(X)  # True where data is present, False where missing

        var_df = pd.DataFrame(index=df.columns)

        # Create AnnData without obs (will be set globally on MuData)
        adata = AnnData(X=X, var=var_df, dtype=np.float32)
        # Use samplesheet index as obs_names
        adata.obs_names = samplesheet.index
        adata.layers["mask"] = mask

        adatas[view_name] = adata

    # 6. Create MuData and set global .obs
    logger.debug("Creating MuData object")
    mdata = MuData(adatas)

    # Set the global .obs to the samplesheet (same for all modalities)
    mdata.obs = samplesheet.copy()

    # 7. Add mutation columns to global .obs (prefixed with "mutation_")
    if mutations_df is not None:
        mutations_df = mutations_df.add_prefix("mutation_")
        for col in mutations_df.columns:
            mdata.obs[col] = mutations_df[col].values

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving MuData (%s) to %s", format, output_path)
    if format == "zarr":
        mdata.write_zarr(str(output_path_obj))
    else:
        mdata.write(str(output_path_obj))
    logger.info("Conversion complete: %d samples, %d modalities",
                len(common_samples), len(adatas))

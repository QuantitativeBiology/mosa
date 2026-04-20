"""MuData I/O: CSV-to-MuData conversion and file inspection."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData
from mudata import MuData

logger = logging.getLogger(__name__)

# Fraction of samplesheet IDs found in CSV row index that triggers an orientation error.
ORIENTATION_ERROR_THRESHOLD = 0.5
# Fraction of samplesheet IDs found in CSV column names below which a warning is emitted.
ORIENTATION_WARN_THRESHOLD = 0.10


# Validation helpers

def _validate_samplesheet(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Samplesheet not found: {path}")

    try:
        ss = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Cannot read samplesheet '{path}': {e}") from e

    if "model_id" not in ss.columns:
        raise ValueError(
            f"Samplesheet '{path}' is missing required column 'model_id'.\n"
            f"  Found columns: {list(ss.columns)}\n"
            f"  'model_id' must contain unique sample identifiers that match the "
            f"column headers of your omic CSVs (e.g. 'ACH-000001', 'TCGA-A1-A0SO')."
        )

    if "model_type" not in ss.columns:
        raise ValueError(
            f"Samplesheet '{path}' is missing required column 'model_type'.\n"
            f"  Found columns: {list(ss.columns)}\n"
            f"  'model_type' is used for conditional encoding, class balancing, and "
            f"batch correction. Add the column even if all samples share the same value."
        )

    if "tissue" not in ss.columns:
        logger.warning(
            "Samplesheet '%s' has no 'tissue' column. Tissue conditioning will be "
            "disabled. Add a 'tissue' column if you want to condition on tissue of origin.",
            path,
        )

    ss = ss.set_index("model_id")
    ss = ss.loc[:, ~ss.columns.str.match(r"^Unnamed")]

    if ss.index.duplicated().any():
        dupes = list(ss.index[ss.index.duplicated(keep=False)].unique())
        raise ValueError(
            f"Samplesheet '{path}' has duplicate model_id values: {dupes[:10]}"
            + (" (and more)" if len(dupes) > 10 else "")
        )

    return ss


def _check_view_orientation(
    df_raw: pd.DataFrame,
    ss_ids: set[str],
    view_name: str,
    csv_path: str,
) -> None:
    """Check whether a view CSV is features x samples (correct) or transposed.

    df_raw has NOT been transposed: rows = potential features, cols = potential samples.
    """
    n_ss = max(len(ss_ids), 1)
    row_overlap = len(set(df_raw.index) & ss_ids) / n_ss
    col_overlap = len(set(df_raw.columns) & ss_ids) / n_ss

    if row_overlap >= ORIENTATION_ERROR_THRESHOLD:
        raise ValueError(
            f"View '{view_name}' ({csv_path}): CSV appears to be samples x features "
            f"(row overlap with samplesheet IDs: {row_overlap:.0%}, "
            f"threshold: {ORIENTATION_ERROR_THRESHOLD:.0%}).\n"
            f"  MOSA expects features x samples: features as rows, samples as columns.\n"
            f"  Fix: transpose your CSV before converting, or re-export from R/Python "
            f"with features as rows and sample IDs as column headers."
        )

    if col_overlap < ORIENTATION_WARN_THRESHOLD and len(ss_ids) > 10:
        logger.warning(
            "View '%s' (%s): only %.0f%% of samplesheet sample IDs found in CSV column "
            "names (threshold: %.0f%%). If conversion produces 0 samples, check that "
            "sample IDs use the same format in both files (e.g. 'ACH-000001' vs 'ACH000001').",
            view_name, csv_path, col_overlap * 100, ORIENTATION_WARN_THRESHOLD * 100,
        )


def _validate_view_numeric(df_raw: pd.DataFrame, view_name: str, csv_path: str) -> None:
    for col in df_raw.columns:
        coerced = pd.to_numeric(df_raw[col], errors="coerce")
        if coerced.isna().any() and not df_raw[col].isna().all():
            bad_vals = df_raw[col][coerced.isna() & df_raw[col].notna()].unique()
            raise ValueError(
                f"View '{view_name}' ({csv_path}): column '{col}' contains non-numeric "
                f"values: {list(bad_vals[:5])}"
                + (" (and more)" if len(bad_vals) > 5 else "") + ".\n"
                f"  All omic CSV values must be numeric. Missing values should be empty "
                f"cells or NaN, not strings like 'NA' or 'N/A'."
            )


def _check_format_extension(output_path: str, fmt: str) -> None:
    ext = Path(output_path).suffix.lstrip(".")
    if fmt == "zarr" and ext == "h5mu":
        logger.warning(
            "Output path '%s' has a .h5mu extension but --format zarr was specified. "
            "The output will be a zarr store. Consider renaming to .zarr for clarity.",
            output_path,
        )
    elif fmt == "h5mu" and ext == "zarr":
        logger.warning(
            "Output path '%s' has a .zarr extension but --format h5mu was specified. "
            "The output will be an HDF5 file. Consider renaming to .h5mu for clarity.",
            output_path,
        )


# Arrow serialisation helpers

def _dearrow_df(df: pd.DataFrame) -> None:
    """Convert Arrow-backed string columns/index to object dtype in-place."""
    if hasattr(df.index, "dtype") and pd.api.types.is_string_dtype(df.index):
        df.index = df.index.astype(object)
    if hasattr(df.columns, "dtype") and pd.api.types.is_string_dtype(df.columns):
        df.columns = df.columns.astype(object)
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(object)


def _dearrow_mudata(mdata: MuData) -> None:
    """Strip Arrow-backed string types from all DataFrames in a MuData.

    MuData.update() can (re-)introduce ArrowStringArray types that anndata
    cannot serialise to zarr/h5. Call this once before writing.
    """
    _dearrow_df(mdata.obs)
    _dearrow_df(mdata.var)
    for mod in mdata.mod.values():
        _dearrow_df(mod.obs)
        _dearrow_df(mod.var)


# Conversion

def csv_to_mudata(
    samplesheet_path: str,
    view_specs: list[tuple[str, str]],
    output_path: str,
    mutations_path: str | None = None,
    format: str = "h5mu",
) -> None:
    """Convert CSV tables to MuData format.

    Parameters
    ----------
    samplesheet_path : str
        Path to samplesheet CSV (requires model_id, model_type; tissue optional).
    view_specs : list of tuple
        (view_name, csv_path) tuples; CSVs must be features x samples.
    output_path : str
        Output path for MuData file.
    mutations_path : str or None
        Optional mutations CSV (features x samples, binary).
    format : str
        Output format: "h5mu" or "zarr".
    """
    import anndata

    anndata.settings.allow_write_nullable_strings = True

    logger.info("Converting CSV dataset to MuData format")

    # Pre-flight validation
    _check_format_extension(output_path, format)
    samplesheet = _validate_samplesheet(samplesheet_path)
    ss_ids = set(samplesheet.index)

    for view_name, csv_path in view_specs:
        if not Path(csv_path).exists():
            raise FileNotFoundError(
                f"View '{view_name}': CSV file not found: {csv_path}"
            )
        try:
            df_raw = pd.read_csv(csv_path, index_col=0)
        except Exception as e:
            raise ValueError(
                f"View '{view_name}': cannot read '{csv_path}': {e}"
            ) from e

        _check_view_orientation(df_raw, ss_ids, view_name, csv_path)
        _validate_view_numeric(df_raw, view_name, csv_path)

    if mutations_path and not Path(mutations_path).exists():
        raise FileNotFoundError(f"Mutations CSV not found: {mutations_path}")

    # Load views
    logger.debug("Loading view CSVs")
    omics: dict[str, pd.DataFrame] = {}
    view_sample_sets: dict[str, set[str]] = {}
    for view_name, csv_path in view_specs:
        df = pd.read_csv(csv_path, index_col=0).T.astype(float)
        omics[view_name] = df
        view_sample_sets[view_name] = set(df.index)
        logger.debug("  view '%s': %d samples x %d features", view_name, *df.shape)

    # Sample alignment
    all_view_samples: set[str] = set()
    for s in view_sample_sets.values():
        all_view_samples |= s
    common_samples = sorted(all_view_samples & ss_ids)

    if not common_samples:
        ss_examples = list(ss_ids)[:5]
        view_name0, _ = view_specs[0]
        view_examples = list(view_sample_sets[view_name0])[:5]
        raise ValueError(
            f"No samples found in the samplesheet that appear in any view CSV.\n"
            f"  Samplesheet model_ids (first 5): {ss_examples}\n"
            f"  View '{view_name0}' column names (first 5): {view_examples}\n"
            f"  Check that sample IDs use the same format in both files."
        )

    logger.debug("Union samples (with samplesheet metadata): %d", len(common_samples))
    for view_name, ss in view_sample_sets.items():
        n_present = len(ss & set(common_samples))
        logger.debug("  view '%s': %d / %d samples present",
                     view_name, n_present, len(common_samples))

    mutations_df = None
    if mutations_path:
        mutations_df = pd.read_csv(mutations_path, index_col=0).T

    # Align to common samples
    samplesheet = samplesheet.loc[common_samples]
    for name in omics:
        omics[name] = omics[name].reindex(common_samples)
    if mutations_df is not None:
        mutations_df = mutations_df.reindex(common_samples).fillna(0)

    # Build AnnData objects
    logger.debug("Creating AnnData objects")
    adatas = {}
    for view_name, df in omics.items():
        X = df.values.astype(np.float32)
        mask = ~np.isnan(X)
        X = np.nan_to_num(X, nan=0.0)
        adata = AnnData(X=X, var=pd.DataFrame(index=df.columns), dtype=np.float32)
        adata.obs_names = samplesheet.index
        adata.layers["mask"] = mask
        adatas[view_name] = adata

    # Build MuData
    logger.debug("Creating MuData object")
    mdata = MuData(adatas)

    for view_name in omics:
        presence = np.array(
            [s in view_sample_sets[view_name] for s in common_samples],
            dtype=bool,
        )
        mdata.obsm[view_name] = presence

    mdata.obs = samplesheet.copy()

    if mutations_df is not None:
        mutations_df = mutations_df.add_prefix("mutation_")
        for col in mutations_df.columns:
            mdata.obs[col] = mutations_df[col].values

    _dearrow_mudata(mdata)

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving MuData (%s) to %s", format, output_path)
    if format == "zarr":
        mdata.write_zarr(str(output_path_obj))
    else:
        mdata.write(str(output_path_obj))
    logger.info("Conversion complete: %d samples, %d modalities",
                len(common_samples), len(adatas))


# Inspection

def inspect_mudata(path: str) -> None:
    """Print a human-readable summary of a MuData file for post-conversion verification."""
    import mudata

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    is_zarr = p.is_dir() or path.endswith(".zarr")
    mdata = mudata.read_zarr(path) if is_zarr else mudata.read(path)

    n_obs = mdata.n_obs
    n_mod = len(mdata.mod)

    print(f"\nMuData: {n_obs} samples x {n_mod} modalities")
    print(f"  File: {path}")

    print("\nModalities:")
    for mod_name, adata in mdata.mod.items():
        n_feat = adata.n_vars

        if "mask" in adata.layers:
            mask = adata.layers["mask"]
            n_present_samples = int(mask.any(axis=1).sum())
            n_present_vals = int(mask.sum())
            n_total_vals = mask.size
            pct = 100.0 * n_present_vals / n_total_vals if n_total_vals > 0 else 0.0
            presence_str = (
                f"{n_present_samples}/{n_obs} samples present, "
                f"{pct:.1f}% values non-missing"
            )
        else:
            presence_str = "no mask layer"

        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        finite = X[np.isfinite(X)]
        if finite.size > 0:
            range_str = (
                f"min={finite.min():.3g}, mean={finite.mean():.3g}, "
                f"max={finite.max():.3g}"
            )
        else:
            range_str = "no finite values"

        flags = []
        if "mask" in adata.layers and adata.layers["mask"].any(axis=1).sum() == 0:
            flags.append("[WARNING: 0 samples present]")
        if finite.size > 0 and np.abs(finite).max() < 1e-9:
            flags.append("[WARNING: all values are ~zero]")

        flag_str = " " + " ".join(flags) if flags else ""
        print(f"  {mod_name}: {n_feat} features | {presence_str} | {range_str}{flag_str}")

    print("\nSample metadata (obs):")
    obs = mdata.obs
    if obs.empty:
        print("  (no metadata)")
    else:
        for col in obs.columns:
            s = obs[col]
            if pd.api.types.is_categorical_dtype(s) or s.dtype == object:
                vc = s.value_counts()
                if len(vc) <= 8:
                    summary = ", ".join(f"{k}: {v}" for k, v in vc.items())
                else:
                    top = ", ".join(f"{k}: {v}" for k, v in vc.head(5).items())
                    summary = f"{top} ... ({len(vc)} unique values)"
            else:
                summary = (
                    f"min={s.min():.3g}, mean={s.mean():.3g}, max={s.max():.3g}"
                )
            print(f"  {col}: {summary}")

    sample_ids = list(mdata.obs_names[:5])
    suffix = f"  ... ({n_obs} total)" if n_obs > 5 else ""
    print(f"\nSample IDs (first 5): {', '.join(sample_ids)}{suffix}")

    obsm_keys = [k for k in mdata.obsm.keys() if k in mdata.mod]
    if obsm_keys:
        print("\nPer-view sample presence (obsm):")
        for key in obsm_keys:
            arr = mdata.obsm[key]
            n = int(arr.sum()) if arr.dtype == bool else int((arr > 0).sum())
            print(f"  {key}: {n}/{n_obs} samples")

    print()

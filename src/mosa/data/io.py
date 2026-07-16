"""MuData I/O: CSV-to-MuData conversion, file loading, and inspection."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from anndata import AnnData
from mudata import MuData
from scipy.sparse import issparse

from mosa.data.dataset import MultiOmicDataset

logger = logging.getLogger(__name__)

# Fraction of samplesheet IDs found in CSV row index that triggers an orientation error.
ORIENTATION_ERROR_THRESHOLD = 0.5
# Fraction of samplesheet IDs found in CSV column names below which a warning is emitted.
ORIENTATION_WARN_THRESHOLD = 0.10


# Zarr path helpers (shared with datamodule.LazyZarrDataset)

def _zarr_view_x_key(view_name: str) -> str:
    return f"mod/{view_name}/X"


def _zarr_view_mask_key(view_name: str, mask_layer: str) -> str:
    return f"mod/{view_name}/layers/{mask_layer}"


# Zarr encoding helpers (used by load_mudata and LazyZarrDataset)

def _zarr_index_key(group) -> str:
    """Return the key that stores the index for a zarr obs/var group.

    AnnData/MuData zarr stores record the index column name in the ``_index``
    attribute. The data lives under ``group[attrs["_index"]]``, not literally
    under ``group["_index"]`` (unless the DataFrame index was named ``_index``).
    """
    return group.attrs.get("_index", "_index")


def _read_zarr_column(group) -> np.ndarray:
    """Decode a single obs/var column from MuData's zarr encoding."""
    if isinstance(group, zarr.Array):
        return np.asarray(group)

    keys = set(group.keys())
    if {"categories", "codes"} <= keys:
        cats_node = group["categories"]
        if isinstance(cats_node, zarr.Group) and "values" in cats_node:
            cats = np.asarray(cats_node["values"])
        else:
            cats = np.asarray(cats_node)
        codes = np.asarray(group["codes"])
        return cats[codes]

    if "values" in keys:
        return np.asarray(group["values"])

    raise ValueError(f"Cannot decode zarr column with keys {keys}")


# MuData loading

def _summary_from_mdata(mdata) -> dict:
    """Build a summarize_structure()-shaped dict from an already-loaded MuData."""
    return {
        "format": "loaded",
        "modalities": {
            name: {"n_features": adata.n_vars, "layers": list(adata.layers.keys())}
            for name, adata in mdata.mod.items()
        },
        "obs_columns": list(mdata.obs.columns),
        "model_type_categories": None,
    }


def _verify_mudata_structure(
    mdata,
    view_names: list[str],
    mask_layer_name: str,
) -> None:
    """Validate MuData structure has required columns and views. Raises ValueError.

    Delegates to DataConfig.validate_against_data so this load-time check and the
    `mosa validate` data-requirements check share one rule implementation.
    """
    from mosa.config import DataConfig

    data_cfg = DataConfig(path="unused", views=view_names, mask_layer_name=mask_layer_name)
    data_cfg.validate_against_data(_summary_from_mdata(mdata))


def _load_h5mu(
    path: str,
    view_names: list[str],
    mask_layer_name: str,
) -> MultiOmicDataset:
    """Load an h5mu file into a MultiOmicDataset."""
    import time
    import mudata

    logger.info("Loading MuData from %s", path)
    t0 = time.perf_counter()
    mdata = mudata.read(path)
    logger.debug("h5mu read took %.2fs", time.perf_counter() - t0)
    _verify_mudata_structure(mdata, view_names, mask_layer_name)

    views: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    feature_names: dict[str, list[str]] = {}

    for view_name in view_names:
        tv = time.perf_counter()
        adata = mdata.mod[view_name]
        X = adata.X
        if issparse(X):
            X = X.toarray()
        X = X.astype(np.float32)

        mask = adata.layers[mask_layer_name]
        if issparse(mask):
            mask = mask.toarray()
        mask = mask.astype(bool)

        if view_name in mdata.obsm:
            presence = np.asarray(mdata.obsm[view_name]).flatten().astype(bool)
            X[~presence] = 0.0
            mask[~presence] = False

        views[view_name] = X
        masks[view_name] = mask
        feature_names[view_name] = list(adata.var_names)
        logger.debug("  view '%s': %d samples x %d features (%.2fs)",
                     view_name, X.shape[0], X.shape[1], time.perf_counter() - tv)

    obs_df = mdata.obs.loc[:, ~mdata.obs.columns.str.match(r"^Unnamed")]
    n_samples = len(obs_df)
    view_summary = ", ".join(f"{k}: {v.shape[1]}" for k, v in views.items())
    logger.info("Loaded %d samples — %s (%.2fs)",
                n_samples, view_summary, time.perf_counter() - t0)

    return MultiOmicDataset(
        views=views,
        masks=masks,
        metadata=obs_df.copy(),
        feature_names=feature_names,
    )


def _load_zarr(
    path: str,
    view_names: list[str],
    mask_layer_name: str,
) -> MultiOmicDataset:
    """Load a zarr store into a MultiOmicDataset (all data read into memory)."""
    import time

    logger.info("Loading MuData from %s", path)
    t0 = time.perf_counter()
    store = zarr.open_group(path, mode="r")

    obs_group = store["obs"]
    obs_idx_key = _zarr_index_key(obs_group)
    sample_names = list(_read_zarr_column(obs_group[obs_idx_key]))

    obs_dict: dict[str, np.ndarray] = {
        "model_type": _read_zarr_column(obs_group["model_type"])
    }
    if "tissue" in obs_group:
        obs_dict["tissue"] = _read_zarr_column(obs_group["tissue"])
    for key in obs_group:
        if key.startswith("mutation_"):
            obs_dict[key] = _read_zarr_column(obs_group[key])
    obs_df = pd.DataFrame(obs_dict, index=sample_names)

    views: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    feature_names: dict[str, list[str]] = {}

    for view_name in view_names:
        mod_key = f"mod/{view_name}"
        if mod_key not in store:
            raise ValueError(f"View '{view_name}' not found in zarr store at {path}")

        views[view_name] = store[f"{mod_key}/X"][:].astype(np.float32)
        masks[view_name] = store[f"{mod_key}/layers/{mask_layer_name}"][:].astype(bool)

        var_group = store[f"{mod_key}/var"]
        var_idx_key = _zarr_index_key(var_group)
        feature_names[view_name] = list(_read_zarr_column(var_group[var_idx_key]))

    view_summary = ", ".join(f"{k}: {v.shape[1]}" for k, v in views.items())
    n_samples = len(obs_df)
    logger.info("Loaded %d samples — %s (%.2fs)",
                n_samples, view_summary, time.perf_counter() - t0)

    return MultiOmicDataset(
        views=views,
        masks=masks,
        metadata=obs_df,
        feature_names=feature_names,
    )


def load_mudata(
    path: str,
    view_names: list[str],
    mask_layer_name: str = "mask",
) -> MultiOmicDataset:
    """Load a MuData file (h5mu or zarr) into a MultiOmicDataset.

    This is the public data loading API. No scaling, no splitting, no config
    mutation — just reads the file and returns the generic container.

    Supports both h5mu and zarr formats (detected from path extension or
    directory structure).

    Parameters
    ----------
    path : str
        Path to .h5mu file or zarr directory.
    view_names : list of str
        Modality names to load (must exist in the file).
    mask_layer_name : str
        Name of the per-feature presence mask layer.
    """
    p = Path(path)
    if p.suffix == ".zarr" or (p.is_dir() and p.suffix != ".h5mu"):
        return _load_zarr(path, view_names, mask_layer_name)
    return _load_h5mu(path, view_names, mask_layer_name)


def summarize_structure(path: str) -> dict:
    """Read MuData structure metadata without loading any matrix data.

    Used by `mosa validate` (and pre-train checks) to verify config↔data
    requirements cheaply, before a full `load_mudata()`. Returns:

        {
          "format": "h5mu" | "zarr",
          "modalities": {view_name: {"n_features": int, "layers": [str, ...]}},
          "obs_columns": [str, ...],
          "model_type_categories": [str, ...] | None,
        }
    """
    p = Path(path)
    if p.suffix == ".zarr" or (p.is_dir() and p.suffix != ".h5mu"):
        return _summarize_zarr(path)
    return _summarize_h5mu(path)


def _summarize_h5mu(path: str) -> dict:
    import mudata

    mdata = mudata.read_h5mu(path, backed=True)
    modalities = {
        name: {"n_features": adata.n_vars, "layers": list(adata.layers.keys())}
        for name, adata in mdata.mod.items()
    }
    obs_columns = list(mdata.obs.columns)
    model_type_categories = None
    if "model_type" in obs_columns:
        model_type_categories = list(pd.unique(mdata.obs["model_type"]))

    return {
        "format": "h5mu",
        "modalities": modalities,
        "obs_columns": obs_columns,
        "model_type_categories": model_type_categories,
    }


def _summarize_zarr(path: str) -> dict:
    store = zarr.open_group(path, mode="r")

    modalities: dict[str, dict] = {}
    if "mod" in store:
        for view_name in store["mod"]:
            mod_group = store[f"mod/{view_name}"]
            layers = list(mod_group["layers"].keys()) if "layers" in mod_group else []
            n_features = 0
            if "var" in mod_group:
                var_group = mod_group["var"]
                var_idx_key = _zarr_index_key(var_group)
                n_features = len(_read_zarr_column(var_group[var_idx_key]))
            modalities[view_name] = {"n_features": n_features, "layers": layers}

    obs_columns: list[str] = []
    model_type_categories = None
    if "obs" in store:
        obs_group = store["obs"]
        idx_key = _zarr_index_key(obs_group)
        obs_columns = [k for k in obs_group if k != idx_key]
        if "model_type" in obs_group:
            model_type_categories = list(pd.unique(_read_zarr_column(obs_group["model_type"])))

    return {
        "format": "zarr",
        "modalities": modalities,
        "obs_columns": obs_columns,
        "model_type_categories": model_type_categories,
    }


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
    # Only object-dtype columns can contain non-numeric strings; float/int are already clean.
    for col in df_raw.select_dtypes(include="object").columns:
        coerced = pd.to_numeric(df_raw[col], errors="coerce")
        bad_mask = coerced.isna() & df_raw[col].notna()
        if bad_mask.any():
            bad_vals = df_raw[col][bad_mask].unique()
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
        # Do NOT impute here; keep NaN for z-score in datamodule
        # Only mask layer records which values are missing
        adata = AnnData(X=X.astype(np.float32), var=pd.DataFrame(index=df.columns), dtype=np.float32)
        adata.obs_names = samplesheet.index
        adata.layers["mask"] = mask
        adatas[view_name] = adata

    # Build MuData
    logger.debug("Creating MuData object")
    mdata = MuData(adatas)

    # Set obsm presence flags based on actual mask (≥1 feature present), not CSV presence
    for view_name in omics:
        adata = adatas[view_name]
        mask = adata.layers["mask"]
        # True if sample has ≥1 non-missing feature
        presence = mask.any(axis=1).reshape(-1, 1)
        mdata.obsm[view_name] = presence
        logger.debug(
            "  %s: %d / %d samples have ≥1 feature present",
            view_name, presence.sum(), len(presence),
        )

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
            adata = mdata.mod[key]
            if "mask" in adata.layers:
                mask_present = int(adata.layers["mask"].any(axis=1).sum())
                if n > mask_present:
                    print(f"    [WARNING: {n - mask_present} samples present in obsm but have all-NaN data — they contribute nothing to this modality]")

    print()

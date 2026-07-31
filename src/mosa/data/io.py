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
from mosa.utils import ensure_dir

logger = logging.getLogger(__name__)

# Fraction of conditionals IDs found in CSV row index that triggers an orientation error.
ORIENTATION_ERROR_THRESHOLD = 0.5
# Fraction of conditionals IDs found in CSV column names below which a warning is emitted.
ORIENTATION_WARN_THRESHOLD = 0.10


# Zarr path helpers (shared with datamodule.LazyZarrDataset)

def is_zarr_path(path: str | Path) -> bool:
    """Detect zarr vs h5mu from a path: h5mu is always a single file, zarr a directory."""
    p = Path(path)
    return p.is_dir() or p.suffix == ".zarr"


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
    with mudata.set_options(pull_on_update=False):
        mdata = mudata.read(path)
    logger.debug("h5mu read took %.2fs", time.perf_counter() - t0)
    _verify_mudata_structure(mdata, view_names, mask_layer_name)

    views: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    feature_names: dict[str, list[str]] = {}

    for view_name in view_names:
        tv = time.perf_counter()
        adata = mdata.mod[view_name]
        # MuData does not guarantee a modality's row order matches the global
        # .obs; pairing X rows with obs positionally would silently mislabel
        # samples. MOSA-produced files are always aligned (convert reindexes
        # every view to one order), so require that here rather than corrupt.
        if not adata.obs_names.equals(mdata.obs_names):
            raise ValueError(
                f"View '{view_name}' sample order does not match the global "
                f".obs order. MOSA requires every modality to be aligned to "
                f".obs; reindex the modality to mdata.obs_names before loading."
            )
        adata.var_names_make_unique()
        X = adata.X
        if issparse(X):
            X = X.toarray()
        X = X.astype(np.float32)

        mask = adata.layers[mask_layer_name]
        if issparse(mask):
            mask = mask.toarray()
        mask = mask.astype(bool)

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

    # Read every obs column (not just model_type/tissue/mutation_*), so the
    # zarr loader preserves the same metadata the h5mu loader does.
    obs_dict: dict[str, np.ndarray] = {}
    for key in obs_group:
        if key == obs_idx_key:
            continue
        try:
            obs_dict[key] = _read_zarr_column(obs_group[key])
        except (ValueError, KeyError) as e:
            logger.debug("Skipping undecodable obs column '%s': %s", key, e)
    obs_df = pd.DataFrame(obs_dict, index=sample_names)

    views: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    feature_names: dict[str, list[str]] = {}

    for view_name in view_names:
        mod_key = f"mod/{view_name}"
        if mod_key not in store:
            raise ValueError(f"View '{view_name}' not found in zarr store at {path}")

        # Same alignment requirement as _load_h5mu: the modality's row order
        # must match the global obs order, else X rows pair with the wrong
        # samples. Checked when the modality stores its own obs index.
        mod_obs_key = f"{mod_key}/obs"
        if mod_obs_key in store:
            mod_obs = store[mod_obs_key]
            mod_idx = list(_read_zarr_column(mod_obs[_zarr_index_key(mod_obs)]))
            if mod_idx != sample_names:
                raise ValueError(
                    f"View '{view_name}' sample order does not match the global "
                    f".obs order in the zarr store. MOSA requires every modality "
                    f"to be aligned to .obs."
                )

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
    if is_zarr_path(path):
        dataset = _load_zarr(path, view_names, mask_layer_name)
    else:
        dataset = _load_h5mu(path, view_names, mask_layer_name)
    dataset.validate()
    return dataset


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
    if is_zarr_path(path):
        return _summarize_zarr(path)
    return _summarize_h5mu(path)


def _summarize_h5mu(path: str) -> dict:
    import mudata

    with mudata.set_options(pull_on_update=False):
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


# Table reading

# Compression suffixes pandas decompresses transparently from the filename.
_COMPRESSION_SUFFIXES = {".gz", ".bz2", ".xz", ".zip"}
_DELIMITERS = {"csv": ",", "tsv": "\t"}


def _table_format(path: str | Path) -> str:
    """Infer table format from a file extension, ignoring any compression suffix."""
    suffixes = [s.lower() for s in Path(path).suffixes]
    if suffixes and suffixes[-1] in _COMPRESSION_SUFFIXES:
        suffixes = suffixes[:-1]
    ext = suffixes[-1] if suffixes else ""

    if ext in (".csv",):
        return "csv"
    if ext in (".tsv", ".txt", ".tab"):
        return "tsv"
    if ext in (".parquet", ".pq"):
        return "parquet"
    raise ValueError(
        f"Unsupported table format '{ext or path}'. Supported: .csv, .tsv, .txt, "
        f".parquet (optionally .gz/.bz2/.xz/.zip for the delimited formats)."
    )


def _read_table(path: str | Path, index_col: int | None = 0) -> pd.DataFrame:
    """Read a table into a DataFrame, dispatching on file extension."""
    fmt = _table_format(path)

    if fmt == "parquet":
        df = pd.read_parquet(path)
        # pandas restores its own index from parquet metadata; files from other
        # tools (R arrow, Spark) arrive with a RangeIndex, first column = index.
        if index_col is not None and isinstance(df.index, pd.RangeIndex):
            df = df.set_index(df.columns[index_col])
        return df

    return pd.read_csv(path, index_col=index_col, sep=_DELIMITERS[fmt])


# Validation helpers

def _require_exists(path: str | Path, what: str = "File") -> Path:
    """Raise a descriptive FileNotFoundError if path is missing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{what} not found: {path}")
    return p


def _validate_conditionals(path: str) -> pd.DataFrame:
    _require_exists(path, what="Conditionals")

    try:
        conditionals = _read_table(path, index_col=None)
    except Exception as e:
        raise ValueError(f"Cannot read conditionals '{path}': {e}") from e

    if "model_id" not in conditionals.columns:
        raise ValueError(
            f"Conditionals '{path}' is missing required column 'model_id'.\n"
            f"  Found columns: {list(conditionals.columns)}\n"
            f"  'model_id' must contain unique sample identifiers that match the "
            f"column headers of your omic CSVs (e.g. 'ACH-000001', 'TCGA-A1-A0SO')."
        )

    if "model_type" not in conditionals.columns:
        raise ValueError(
            f"Conditionals '{path}' is missing required column 'model_type'.\n"
            f"  Found columns: {list(conditionals.columns)}\n"
            f"  'model_type' is used for conditional encoding, class balancing, and "
            f"batch correction. Add the column even if all samples share the same value."
        )

    if "tissue" not in conditionals.columns:
        logger.warning(
            "Conditionals '%s' has no 'tissue' column. Tissue conditioning will be "
            "disabled. Add a 'tissue' column if you want to condition on tissue of origin.",
            path,
        )

    conditionals = conditionals.set_index("model_id")
    conditionals = conditionals.loc[:, ~conditionals.columns.str.match(r"^Unnamed")]

    if conditionals.index.duplicated().any():
        dupes = list(conditionals.index[conditionals.index.duplicated(keep=False)].unique())
        raise ValueError(
            f"Conditionals '{path}' has duplicate model_id values: {dupes[:10]}"
            + (" (and more)" if len(dupes) > 10 else "")
        )

    return conditionals


def _filter_metadata(
    metadata: pd.DataFrame,
    filters: dict[str, list[str]],
    path: str,
) -> pd.DataFrame:
    """Restrict metadata rows to allowed values per column, before alignment.

    The sample axis is the view samples intersected with the metadata index, so
    shrinking the metadata shrinks the axis.
    """
    for column, allowed in filters.items():
        if column not in metadata.columns:
            raise ValueError(
                f"Cannot filter on '{column}': not a column of '{path}'.\n"
                f"  Available columns: {list(metadata.columns)}"
            )

        before = len(metadata)
        observed = list(pd.unique(metadata[column].dropna()))
        metadata = metadata[metadata[column].isin(allowed)]
        logger.info("filter %s in %s: kept %d of %d samples",
                    column, allowed, len(metadata), before)

        if metadata.empty:
            raise ValueError(
                f"Filter '{column}' in {allowed} matched no samples in '{path}'.\n"
                f"  Values present in '{column}': {observed[:20]}"
                + (" (and more)" if len(observed) > 20 else "")
            )

    return metadata


def _load_id_map(path: str) -> dict[str, str]:
    """Read a sample-ID crosswalk mapping provider IDs onto canonical model_ids.

    Applied to omic views only; metadata and mutations already use canonical IDs.
    """
    _require_exists(path, what="ID map")

    try:
        df = _read_table(path, index_col=None)
    except Exception as e:
        raise ValueError(f"Cannot read ID map '{path}': {e}") from e

    missing = {"source_id", "model_id"} - set(df.columns)
    if missing:
        raise ValueError(
            f"ID map '{path}' is missing required column(s): {sorted(missing)}.\n"
            f"  Found columns: {list(df.columns)}\n"
            f"  Expected 'source_id' (the ID as it appears in your omic files) and "
            f"'model_id' (the canonical ID used in your metadata)."
        )

    return dict(zip(df["source_id"].astype(str), df["model_id"].astype(str)))


def _check_view_orientation(
    df_raw: pd.DataFrame,
    conditionals_ids: set[str],
    view_name: str,
    csv_path: str,
) -> None:
    """Check whether a view CSV is features x samples (correct) or transposed.

    df_raw has NOT been transposed: rows = potential features, cols = potential samples.
    """
    n_conditionals = max(len(conditionals_ids), 1)
    row_overlap = len(set(df_raw.index) & conditionals_ids) / n_conditionals
    col_overlap = len(set(df_raw.columns) & conditionals_ids) / n_conditionals

    if row_overlap >= ORIENTATION_ERROR_THRESHOLD:
        raise ValueError(
            f"View '{view_name}' ({csv_path}): CSV appears to be samples x features "
            f"(row overlap with conditionals IDs: {row_overlap:.0%}, "
            f"threshold: {ORIENTATION_ERROR_THRESHOLD:.0%}).\n"
            f"  MOSA expects features x samples: features as rows, samples as columns.\n"
            f"  Fix: transpose your CSV before converting, or re-export from R/Python "
            f"with features as rows and sample IDs as column headers."
        )

    if col_overlap < ORIENTATION_WARN_THRESHOLD and len(conditionals_ids) > 10:
        logger.warning(
            "View '%s' (%s): only %.0f%% of conditionals sample IDs found in CSV column "
            "names (threshold: %.0f%%). If conversion produces 0 samples, check that "
            "sample IDs use the same format in both files (e.g. 'ACH-000001' vs 'ACH000001').",
            view_name, csv_path, col_overlap * 100, ORIENTATION_WARN_THRESHOLD * 100,
        )


def _validate_view_numeric(df_raw: pd.DataFrame, view_name: str, csv_path: str) -> None:
    # Only object-dtype columns can contain non-numeric strings; float/int are already clean.
    for col in df_raw.select_dtypes(include=["object", "str"]).columns:
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


# Alignment

def _combine_view_files(
    frames: list[pd.DataFrame],
    view_name: str,
    on_collision: str = "error",
) -> pd.DataFrame:
    """Assemble one omic from several files: concatenate samples, union features.

    A feature absent from one file is NaN for that file's samples rather than
    dropped. Collisions are checked over the view's pooled samples, catching
    both a sample present in two files and two columns an ID map resolved onto
    one sample.
    """
    combined = frames[0] if len(frames) == 1 else pd.concat(frames, axis=0)

    duplicates = combined.index[combined.index.duplicated()].unique()
    if len(duplicates) > 0:
        if on_collision == "error":
            raise ValueError(
                f"View '{view_name}': {len(duplicates)} sample ID(s) occur more "
                f"than once: {list(duplicates[:10])}"
                + (" (and more)" if len(duplicates) > 10 else "") + ".\n"
                f"  Two columns resolving to one sample is ambiguous. Pass "
                f"--on-collision first to keep the first occurrence, or "
                f"resolve it upstream."
            )
        combined = combined[~combined.index.duplicated(keep="first")]
        logger.warning(
            "View '%s': kept the first of %d colliding sample ID(s): %s%s",
            view_name, len(duplicates), list(duplicates[:10]),
            " (and more)" if len(duplicates) > 10 else "",
        )

    if len(frames) > 1:
        logger.debug("  view '%s': combined %d files -> %d samples x %d features",
                     view_name, len(frames), *combined.shape)
    return combined


def align_views(
    views: dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    min_views: int = 1,
    shared_features: bool = False,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Align every view and the metadata onto one shared sample axis.

    The axis is the sorted union of samples across views, restricted to those
    the metadata describes; views gain all-NaN rows for samples they lack.

    Parameters
    ----------
    views : dict of str to DataFrame
        {view_name: frame} indexed by sample ID (samples x features).
    metadata : DataFrame
        Sample metadata, indexed by sample ID.
    min_views : int
        Drop samples holding real values in fewer than this many views. 1 keeps
        the whole union; len(views) keeps only samples complete across omics.
    shared_features : bool
        Reduce every view to the features they all share. Off by default; valid
        only when all views use one identifier namespace, e.g. every omic
        summarised to gene level.

    Returns
    -------
    (aligned_views, aligned_metadata, presence), all sharing one row order.
    `presence` is a bool frame with one has_<view> column per view, True where
    the sample appeared in that view's file. A sample absent from a view and one
    whose values are all NaN yield identical matrices, so presence records the
    difference; min_views instead counts views holding at least one real value.
    """
    metadata_ids = set(metadata.index)
    view_samples = {name: set(df.index) for name, df in views.items()}

    all_view_samples: set[str] = set()
    for samples in view_samples.values():
        all_view_samples |= samples

    sample_axis = sorted(all_view_samples & metadata_ids)
    if not sample_axis:
        first_view = next(iter(views), None)
        view_examples = list(views[first_view].index)[:5] if first_view else []
        raise ValueError(
            f"No samples found in the metadata that appear in any view.\n"
            f"  Metadata model_ids (first 5): {list(metadata_ids)[:5]}\n"
            f"  View '{first_view}' sample IDs (first 5): {view_examples}\n"
            f"  Check that sample IDs use the same format in both files."
        )

    # `presence` = the sample appeared in the view's file, which no mask can
    # recover. `observed` = it holds a real value there, which min_views counts.
    presence = pd.DataFrame(
        {f"has_{name}": [s in samples for s in sample_axis]
         for name, samples in view_samples.items()},
        index=sample_axis, dtype=bool,
    )
    observed = pd.DataFrame(
        {name: df.notna().any(axis=1).reindex(sample_axis, fill_value=False)
         for name, df in views.items()},
        index=sample_axis, dtype=bool,
    )

    logger.debug("Sample axis (union across views, with metadata): %d", len(sample_axis))
    for name in view_samples:
        logger.debug("  view '%s': %d in file, %d with data (of %d)",
                     name, int(presence[f"has_{name}"].sum()),
                     int(observed[name].sum()), len(sample_axis))

    if min_views > 1:
        n_views = observed.sum(axis=1)
        keep = [s for s in sample_axis if n_views[s] >= min_views]
        if not keep:
            raise ValueError(
                f"No samples have data in at least {min_views} views "
                f"(the most any sample reaches is {int(n_views.max())}).\n"
                f"  Lower --min-views, or check that sample IDs match across views."
            )
        logger.info("min_views=%d dropped %d of %d samples",
                    min_views, len(sample_axis) - len(keep), len(sample_axis))
        sample_axis = keep
        presence = presence.loc[keep]
        observed = observed.loc[keep]

    aligned = {name: df.reindex(sample_axis) for name, df in views.items()}

    if shared_features and aligned:
        common: set | None = None
        for df in aligned.values():
            cols = set(df.columns)
            common = cols if common is None else common & cols
        shared = sorted(common or set())
        smallest = min(len(df.columns) for df in aligned.values())

        if not shared:
            sizes = {n: len(df.columns) for n, df in aligned.items()}
            raise ValueError(
                f"shared_features left no features: the views have no feature "
                f"names in common (view sizes: {sizes}).\n"
                f"  This option requires every view to use one identifier "
                f"namespace, e.g. all omics summarised to gene symbols."
            )
        if len(shared) < 0.1 * smallest:
            logger.warning(
                "shared_features kept only %d features, under 10%% of the "
                "smallest view (%d). Check that the views really share an "
                "identifier namespace.", len(shared), smallest,
            )
        logger.info("shared_features: %d features common to all %d views",
                    len(shared), len(aligned))
        aligned = {name: df[shared] for name, df in aligned.items()}

    return aligned, metadata.loc[sample_axis], presence


def _print_conversion_report(
    n_metadata: int,
    n_union_views: int,
    n_matched: int,
    min_views: int,
    presence: pd.DataFrame,
    omics: dict[str, pd.DataFrame],
) -> None:
    """Print what came in, what went out, and why anything was dropped."""
    n_final = len(presence)

    def row(label: str, value: int) -> None:
        print(f"    {label:<28}{value:>8}")

    print("\nConversion report")
    print("  Samples")
    row("in metadata:", n_metadata)
    row("in at least one view:", n_union_views)
    row("matched on both:", n_matched)
    row("dropped, no metadata:", n_union_views - n_matched)
    row("dropped, no view data:", n_metadata - n_matched)
    if min_views > 1:
        row(f"dropped, min_views={min_views}:", n_matched - n_final)
    row("final:", n_final)

    print("  Views")
    for name, df in omics.items():
        n_present = int(presence[f"has_{name}"].sum())
        print(f"    {name:<16}{df.shape[1]:>7} features"
              f"{n_present:>8} / {n_final} samples present")
        n_dup = int(df.columns.duplicated().sum())
        if n_dup:
            print(f"    {'':<16}{n_dup:>7} duplicate feature name(s), "
                  f"suffixed at load")

    print("  Samples by view count")
    n_views = presence.sum(axis=1)
    for k in range(1, len(omics) + 1):
        row(f"{k} view{'s' if k > 1 else ''}:", int((n_views == k).sum()))
    print()


# Conversion

def csv_to_mudata(
    conditionals_path: str,
    view_specs: list[tuple[str, str]],
    output_path: str,
    mutations_path: str | None = None,
    format: str = "h5mu",
    id_map_path: str | None = None,
    on_collision: str = "error",
    min_views: int = 1,
    metadata_filters: dict[str, list[str]] | None = None,
    shared_features: bool = False,
) -> None:
    """Convert tabular omic files to MuData format.

    Parameters
    ----------
    conditionals_path : str
        Path to the sample metadata table (requires model_id, model_type;
        tissue optional).
    view_specs : list of tuple
        (view_name, path) tuples; tables must be features x samples. Repeating
        a view name assembles that omic from several files.
    output_path : str
        Output path for MuData file.
    mutations_path : str or None
        Optional mutations table (features x samples, binary).
    format : str
        Output format: "h5mu" or "zarr".
    id_map_path : str or None
        Optional sample-ID crosswalk (columns: source_id, model_id) applied to
        every view before alignment.
    on_collision : str
        What to do when two columns resolve to one sample: "error" or "first".
    min_views : int
        Drop samples with data in fewer than this many views (default 1, which
        keeps every sample).
    metadata_filters : dict or None
        {column: [allowed values]} applied to the metadata before alignment,
        restricting which samples are eligible at all.
    shared_features : bool
        Reduce every view to the features they all share (default off).
    """
    import anndata

    if on_collision not in ("error", "first"):
        raise ValueError(
            f"on_collision must be 'error' or 'first', got '{on_collision}'"
        )

    anndata.settings.allow_write_nullable_strings = True

    logger.info("Converting dataset to MuData format")

    # Pre-flight validation
    _check_format_extension(output_path, format)
    conditionals = _validate_conditionals(conditionals_path)
    if metadata_filters:
        conditionals = _filter_metadata(conditionals, metadata_filters, conditionals_path)
    conditionals_ids = set(conditionals.index)

    id_map = _load_id_map(id_map_path) if id_map_path else {}
    # A source ID mapping into the metadata is a known sample pre-mapping too.
    orientation_ids = conditionals_ids | {
        src for src, dst in id_map.items() if dst in conditionals_ids
    }

    if mutations_path:
        _require_exists(mutations_path, what="Mutations CSV")

    # Validation and load share one read; these files reach gigabytes. Repeating
    # a view name in view_specs adds another file to that same omic.
    logger.debug("Loading view tables")
    view_frames: dict[str, list[pd.DataFrame]] = {}
    for view_name, csv_path in view_specs:
        _require_exists(csv_path, what=f"View '{view_name}': file")
        try:
            df_raw = _read_table(csv_path)
        except Exception as e:
            raise ValueError(
                f"View '{view_name}': cannot read '{csv_path}': {e}"
            ) from e

        _check_view_orientation(df_raw, orientation_ids, view_name, csv_path)
        _validate_view_numeric(df_raw, view_name, csv_path)

        df = df_raw.T.astype(np.float32)
        # astype copied if the source was float64; if it was already float32 the
        # two share memory and this just drops the name.
        del df_raw
        if id_map:
            df = df.rename(index=id_map)  # unmapped IDs pass through unchanged
        view_frames.setdefault(view_name, []).append(df)
        logger.debug("  view '%s' <- %s: %d samples x %d features",
                     view_name, csv_path, *df.shape)

    omics = {
        name: _combine_view_files(frames, name, on_collision)
        for name, frames in view_frames.items()
    }

    # Captured pre-alignment so the report can attribute every dropped sample.
    metadata_ids_pre = set(conditionals.index)
    union_pre: set[str] = set()
    for df in omics.values():
        union_pre |= set(df.index)
    n_matched = len(union_pre & metadata_ids_pre)

    omics, conditionals, presence = align_views(
        omics, conditionals, min_views, shared_features
    )
    sample_axis = list(conditionals.index)

    _print_conversion_report(
        n_metadata=len(metadata_ids_pre),
        n_union_views=len(union_pre),
        n_matched=n_matched,
        min_views=min_views,
        presence=presence,
        omics=omics,
    )

    mutations_df = None
    if mutations_path:
        mutations_df = _read_table(mutations_path).T.reindex(sample_axis).fillna(0)

    # Build AnnData objects
    logger.debug("Creating AnnData objects")
    adatas = {}
    for view_name, df in omics.items():
        X = df.values.astype(np.float32)
        mask = ~np.isnan(X)
        # Do NOT impute here; keep NaN for z-score in datamodule
        # Only mask layer records which values are missing
        adata = AnnData(X=X, var=pd.DataFrame(index=df.columns))
        adata.obs_names = conditionals.index
        adata.layers["mask"] = mask
        adatas[view_name] = adata

    # Build MuData
    import mudata

    logger.debug("Creating MuData object")
    with mudata.set_options(pull_on_update=False):
        mdata = MuData(adatas)

    mdata.obs = conditionals.copy()

    # A sample absent from a view and one whose values are all NaN share a mask,
    # so presence is stored beside it.
    for col in presence.columns:
        mdata.obs[col] = presence[col].values

    if mutations_df is not None:
        mutations_df = mutations_df.add_prefix("mutation_")
        for col in mutations_df.columns:
            mdata.obs[col] = mutations_df[col].values

    _dearrow_mudata(mdata)

    output_path_obj = Path(output_path)
    ensure_dir(output_path_obj.parent)

    logger.info("Saving MuData (%s) to %s", format, output_path)
    # Writing runs update(), which pulls per-modality obs/var unless disabled.
    with mudata.set_options(pull_on_update=False):
        if format == "zarr":
            mdata.write_zarr(str(output_path_obj))
        else:
            mdata.write(str(output_path_obj))
    logger.info("Conversion complete: %d samples, %d modalities",
                len(sample_axis), len(adatas))


# Inspection

def inspect_mudata(path: str) -> None:
    """Print a human-readable summary of a MuData file for post-conversion verification."""
    import mudata

    _require_exists(path, what="File")

    with mudata.set_options(pull_on_update=False):
        mdata = mudata.read_zarr(path) if is_zarr_path(path) else mudata.read(path)

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
            if (isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object
                    or pd.api.types.is_bool_dtype(s)):
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

    print("\nPer-view sample presence:")
    for key, adata in mdata.mod.items():
        if "mask" not in adata.layers:
            continue
        n = int(adata.layers["mask"].any(axis=1).sum())
        print(f"  {key}: {n}/{n_obs} samples")

    print()

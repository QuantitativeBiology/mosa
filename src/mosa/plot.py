"""Plotting utilities for MOSA training diagnostics.

Generates UMAP visualizations, loss curves, reconstruction scatter plots,
and clustering quality metrics from training outputs.
"""

import colorsys
import logging
import warnings
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

from mosa.errors import DataError
from mosa.utils import ensure_dir, mudata_set_options

logger = logging.getLogger(__name__)

DEFAULT_PALETTE = {
    "Lung": "#007fff",
    "TCGA-LUAD": "#007fff",
    "TCGA-LUSC": "#007fff",
    "Prostate": "#665d1e",
    "TCGA-PRAD": "#665d1e",
    "Stomach": "#ffbf00",
    "TCGA-STAD": "#ffbf00",
    "Central Nervous System": "#fbceb1",
    "TCGA-GBM": "#fbceb1",
    "TCGA-LGG": "#fbceb1",
    "Skin": "#ff033e",
    "TCGA-SKCM": "#ff033e",
    "Bladder": "#ab274f",
    "TCGA-BLCA": "#ab274f",
    "Haematopoietic and Lymphoid": "#d5e6f7",
    "TCGA-LAML": "#d5e6f7",
    "TCGA-DLBC": "#d5e6f7",
    "TCGA-THYM": "#d5e6f7",
    "Kidney": "#7cb9e8",
    "TCGA-KIRC": "#7cb9e8",
    "TCGA-KIRP": "#7cb9e8",
    "TCGA-KICH": "#7cb9e8",
    "Thyroid": "#efdecd",
    "TCGA-THCA": "#efdecd",
    "Soft Tissue": "#8db600",
    "TCGA-SARC": "#8db600",
    "Head and Neck": "#e9d66b",
    "TCGA-HNSC": "#e9d66b",
    "Ovary": "#b284be",
    "TCGA-OV": "#b284be",
    "Bone": "#b2beb5",
    "Endometrium": "#10b36f",
    "Breast": "#6e7f80",
    "TCGA-BRCA": "#6e7f80",
    "Pancreas": "#ff7e00",
    "TCGA-PAAD": "#ff7e00",
    "Peripheral Nervous System": "#87a96b",
    "TCGA-PCPG": "#87a96b",
    "Cervix": "#c9ffe5",
    "TCGA-CESC": "#c9ffe5",
    "Large Intestine": "#9f2b68",
    "TCGA-COAD": "#9f2b68",
    "TCGA-READ": "#9f2b68",
    "Liver": "#00ffff",
    "TCGA-LIHC": "#00ffff",
    "Vulva": "#008000",
    "Esophagus": "#cd9575",
    "TCGA-ESCA": "#cd9575",
    "Biliary Tract": "#72a0c1",
    "TCGA-CHOL": "#72a0c1",
    "Other tissue": "#a32638",
    "Small Intestine": "#9966cc",
    "Placenta": "#f19cbb",
    "Testis": "#e32636",
    "TCGA-TGCT": "#e32636",
    "Adrenal Gland": "#3b7a57",
    "TCGA-ACC": "#3b7a57",
    "Uterus": "#7a3b5e",
    "TCGA-UCEC": "#7a3b5e",
    "TCGA-UCS": "#7a3b5e",
    "Unknown": "#a32638",
    "Eye": "#ff1493",
    "TCGA-UVM": "#ff1493",
    "TCGA-MESO": "#002e5c",
    "Cell Line": "darkorange",
    "Organoid": "firebrick",
    "Broad": "#32CD32",
    "Sanger": "#FF8D00",
    "Tumor": "darkgray",
}
"""Default tissue/model_type color palette for DepMap data."""

# Seed for deterministic color generation for unknown categories.
_PALETTE_COLOR_SEED = 0xA3F1B2C4

# Marker shapes cycled across model_type layers, most-common first.
_MARKER_CYCLE = ["o", "^", "s", "D", "P", "X", "v", "*", "<", ">"]


def _name_to_color(name):
    """Map a category name to a deterministic HLS color via CRC32."""
    hue = ((zlib.crc32(name.encode()) ^ _PALETTE_COLOR_SEED) & 0xFFFFFFFF) / 0xFFFFFFFF
    return colorsys.hls_to_rgb(hue, 0.5, 0.7)


def build_palette(categories, base_palette=None):
    """Build a complete color palette covering all categories.

    Known categories use colors from base_palette. Unknown categories receive
    deterministic colors derived from their name.
    """
    base = base_palette or DEFAULT_PALETTE
    return {c: base[c] if c in base else _name_to_color(c) for c in categories}


def _infer_layer_styles(plot_df, model_type_col="model_type"):
    """Infer per-category plotting styles directly from the data.

    Draws the most common category first (background) and the rarest last
    (foreground), giving rarer categories more visual emphasis (higher alpha,
    larger markers, an edge outline) instead of relying on hardcoded label
    names, so this works for any category scheme.
    """
    counts = plot_df[model_type_col].dropna().value_counts()
    if counts.empty:
        return []

    ordered = counts.sort_values(ascending=False).index.tolist()
    total = counts.sum()

    layers = []
    for i, mt in enumerate(ordered):
        share = counts[mt] / total  # 0..1, larger = more common
        rarity = 1.0 - share  # larger = rarer -> more emphasis
        layers.append(
            {
                "model_type": mt,
                "marker": _MARKER_CYCLE[i % len(_MARKER_CYCLE)],
                "alpha": float(np.clip(0.35 + 0.5 * rarity, 0.35, 0.9)),
                "size": float(np.clip(4 + 6 * rarity, 4, 10)),
                "zorder": i + 1,
                "edgecolor": "black" if rarity > 0.5 else None,
                "linewidth": 0.2 if rarity > 0.5 else 0.1,
            }
        )
    return layers


def configure_plot_style():
    """Apply matplotlib styling for publication-ready figures."""
    plt.rcParams.update(
        {
            "figure.figsize": [2.5, 2.5],
            "figure.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.titlesize": 7,
            "legend.fontsize": 6,
            "legend.title_fontsize": 6,
            "axes.labelsize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "grid.linewidth": 0.15,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.color": "black",
            "grid.alpha": 0.5,
            "legend.frameon": False,
            "legend.loc": "best",
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def compute_umap_embedding(
    df,
    n_neighbors=25,
    min_dist=0.25,
    metric="euclidean",
    n_components=2,
    random_state=42,
    pca_components=None,
):
    """Compute UMAP embedding with optional PCA pre-reduction.

    Parameters
    ----------
    df : DataFrame or array-like
        Data matrix (samples x features).
    pca_components : int or None
        Optional PCA reduction before UMAP.

    Returns
    -------
    DataFrame
        Embedding with columns UMAP1, UMAP2, etc.
    """
    X = df.values if isinstance(df, pd.DataFrame) else np.asarray(df)
    index = df.index if isinstance(df, pd.DataFrame) else None

    if pca_components is not None:
        # PCA cannot produce more components than min(n_samples, n_features);
        # small cohorts would otherwise fail on the default of 50.
        n_components_pca = min(pca_components, *X.shape)
        X = PCA(n_components=n_components_pca).fit_transform(X)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        embedding = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            n_components=n_components,
            random_state=random_state,
        ).fit_transform(X)

    return pd.DataFrame(
        embedding, index=index, columns=[f"UMAP{i + 1}" for i in range(n_components)]
    )


def plot_umap(plot_df, palette, title=None, model_type_col="model_type"):
    """Plot UMAP embedding colored by tissue, shaped by model_type.

    Parameters
    ----------
    plot_df : DataFrame
        Must have columns: UMAP1, UMAP2, tissue, and `model_type_col`.
    palette : dict
        Color mapping for tissues and model types.
    title : str or None
        Plot title.
    model_type_col : str
        Column used to derive marker/opacity/zorder styling. Styles are
        inferred from the data (see `_infer_layer_styles`) rather than
        hardcoded label names, so this works for any category scheme.

    Returns
    -------
    (fig, ax) tuple.
    """
    fig, ax = plt.subplots()

    layers = _infer_layer_styles(plot_df, model_type_col=model_type_col)
    sizes = {layer["model_type"]: layer["size"] for layer in layers}
    markers = {layer["model_type"]: layer["marker"] for layer in layers}

    for layer in layers:
        subset = plot_df[plot_df[model_type_col] == layer["model_type"]]
        if subset.empty:
            continue
        scatter_kw = dict(
            data=subset,
            x="UMAP1",
            y="UMAP2",
            hue="tissue",
            palette=palette,
            style=model_type_col,
            markers=markers,
            size=model_type_col,
            sizes=sizes,
            alpha=layer["alpha"],
            zorder=layer["zorder"],
            linewidth=layer["linewidth"],
            legend=False,
            ax=ax,
        )
        if layer["edgecolor"]:
            scatter_kw["edgecolor"] = layer["edgecolor"]
        sns.scatterplot(**scatter_kw)

    type_handles = [
        Line2D(
            [0],
            [0],
            marker=layer["marker"],
            color="w",
            label=layer["model_type"],
            markerfacecolor="gray",
            markersize=6,
            **(
                {"markeredgecolor": layer["edgecolor"], "markeredgewidth": 0.6}
                if layer["edgecolor"]
                else {}
            ),
        )
        for layer in layers
    ]
    legend_markers = ax.legend(
        handles=type_handles,
        title="Sample Type",
        loc="upper left",
        bbox_to_anchor=(1.05, 1.0),
        frameon=False,
    )
    ax.add_artist(legend_markers)

    tissues_present = plot_df["tissue"].dropna().unique()
    color_handles = [
        Line2D(
            [0], [0], marker="o", color=palette[t], label=t, linestyle="", markersize=6
        )
        for t in tissues_present
    ]
    ax.legend(
        handles=color_handles,
        title="Tissue",
        loc="upper left",
        bbox_to_anchor=(1.5, 1.0),
        ncol=2,
        columnspacing=0.5,
        handletextpad=0.3,
        frameon=False,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    return fig, ax


# Helpers


def _try_read(parquet_path, csv_path):
    """Try reading parquet first, fall back to CSV."""
    if Path(parquet_path).exists():
        return pd.read_parquet(parquet_path)
    elif Path(csv_path).exists():
        return pd.read_csv(csv_path, index_col=0)
    return None


def _read_split(output_dir, splits, filename):
    """Try reading filename.parquet/csv from each split subdir in order, first hit wins."""
    for split in splits:
        data = _try_read(
            output_dir / split / f"{filename}.parquet",
            output_dir / split / f"{filename}.csv",
        )
        if data is not None:
            return data
    return None


_LATENT_SPLITS = ("full", "train")


def _require_training_artifacts(output_dir):
    """Fail before plotting when output_dir holds no training outputs.

    Without this the loaders return empty and every plot silently no-ops,
    so the command reports success having written nothing.
    """
    if not output_dir.exists():
        raise DataError(
            f"Output directory not found: {output_dir}. "
            f"Run 'mosa train' first, or point --output-dir at a completed run."
        )

    candidates = [
        output_dir / split / f"latent{ext}"
        for split in _LATENT_SPLITS
        for ext in (".parquet", ".csv")
    ]
    if not any(p.exists() for p in candidates):
        looked_for = " or ".join(f"{split}/latent.parquet" for split in _LATENT_SPLITS)
        raise DataError(
            f"No latent representations found under {output_dir} "
            f"(looked for {looked_for}). "
            f"Run 'mosa train' first, or point --output-dir at a completed run."
        )


def _load_conditionals(path):
    """Load conditionals CSV with model_id as index."""
    conditionals = pd.read_csv(path, index_col=0)
    if "model_id" in conditionals.columns:
        conditionals = conditionals.set_index("model_id")
    return conditionals


def _align_to_conditionals(df, conditionals):
    """Keep only samples present in both DataFrames."""
    common = df.index.intersection(conditionals.index)
    if common.empty:
        return df, conditionals
    return df.loc[common], conditionals.loc[common]


def _save_fig(fig, out_path):
    """Save figure and close it."""
    ensure_dir(out_path.parent)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Saved plot: %s", out_path)


# Loss plots


def _load_lightning_metrics(output_dir):
    """Load metrics from latest Lightning log version.

    Returns
    -------
    dict
        Metric name to DataFrame with columns [epoch, value].
    """
    log_dir = Path(output_dir) / "lightning_logs"
    if not log_dir.exists():
        return {}
    versions = sorted(
        log_dir.glob("version_*"), key=lambda p: int(p.name.split("_")[1])
    )
    if not versions:
        return {}
    metrics_path = versions[-1] / "metrics.csv"
    if not metrics_path.exists():
        return {}

    df = pd.read_csv(metrics_path)
    result = {}
    for col in df.columns:
        if col in ("epoch", "step"):
            continue
        subset = df[["epoch", col]].dropna(subset=[col])
        grouped = subset.groupby("epoch")[col].mean().reset_index()
        grouped.columns = ["epoch", "value"]
        result[col] = grouped
    return result


def _plot_single_loss(metric_df, title, out_path, color):
    """Plot loss metric over epochs."""
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot(metric_df["epoch"], metric_df["value"], color=color, linewidth=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    _save_fig(fig, out_path)


def _plot_composite_loss(metrics, out_path):
    """Plot total loss with component breakdown."""
    total = metrics.get("train/loss")
    if total is None:
        return

    cmap = plt.get_cmap("tab20")
    components = [
        ("train/loss", "Total VAE Loss", cmap(0)),
        ("train/adv_loss", "Adversarial", cmap(2)),
        ("train/kl", "KL Divergence", cmap(4)),
        ("train/recon", "MSE", cmap(6)),
    ]

    fig, ax = plt.subplots(figsize=(3, 2))
    for key, label, color in components:
        df = metrics.get(key)
        if df is not None:
            ax.plot(df["epoch"], df["value"], label=label, color=color, linewidth=2)

    ax.set_xlabel("epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Total Loss")
    ax.legend()
    _save_fig(fig, out_path)


def _plot_omic_mse(metrics, omic, out_path):
    """Plot per-omic MSE loss with train/val and per-model-type breakdown."""
    cmap = plt.get_cmap("tab20")

    total_key = f"train/recon_{omic}"
    val_key = f"val/recon_{omic}"

    # Auto-discover per-model_type keys (e.g. train/recon_gexp_Tumor)
    group_prefix = f"train/recon_{omic}_"
    group_keys = sorted(k for k in metrics if k.startswith(group_prefix))

    has_total = total_key in metrics
    has_val = val_key in metrics

    if not has_total and not has_val:
        return

    fig, ax = plt.subplots(figsize=(3, 2))
    color_idx = 0

    if has_total:
        ax.plot(
            metrics[total_key]["epoch"],
            metrics[total_key]["value"],
            label="Total",
            color=cmap(color_idx),
            linewidth=2,
        )
        color_idx += 2

    for key in group_keys:
        group_name = key[len(group_prefix) :]
        ax.plot(
            metrics[key]["epoch"],
            metrics[key]["value"],
            label=group_name,
            color=cmap(color_idx),
            linewidth=2,
        )
        color_idx += 2

    if has_val:
        ax.plot(
            metrics[val_key]["epoch"],
            metrics[val_key]["value"],
            label="Val",
            color=cmap(color_idx),
            linewidth=2,
            linestyle="--",
        )

    ax.set_xlabel("epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{omic.upper()} MSE Loss")
    ax.legend()
    _save_fig(fig, out_path)


def _generate_loss_plots(output_dir, views, plots_dir):
    """Generate loss curve plots from Lightning metrics."""
    metrics = _load_lightning_metrics(output_dir)
    if not metrics:
        return

    cmap = plt.get_cmap("tab20")
    individual_losses = [
        ("train/kl", "KL Divergence Loss", "loss_kl.png", cmap(4)),
        ("train/adv_loss", "Adversarial Loss for VAE", "loss_adv.png", cmap(2)),
        ("train/disc_loss", "Discriminator Loss", "loss_disc.png", cmap(2)),
    ]
    for key, title, filename, color in individual_losses:
        df = metrics.get(key)
        if df is not None:
            _plot_single_loss(df, title, plots_dir / filename, color)

    _plot_composite_loss(metrics, plots_dir / "loss_total.png")

    for omic in views:
        _plot_omic_mse(metrics, omic, plots_dir / f"mse_{omic}.png")


# Reconstruction scatter plots


def _scatter_with_identity(ax, x, y, **scatter_kw):
    """Plot scatter with y=x identity line."""
    ax.scatter(x, y, **scatter_kw)
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.5, label="Identity")


def _plot_sample_scatter(plot_df, out_path, xlabel, ylabel):
    """Scatter of per-sample means, colored by model type."""
    fig, ax = plt.subplots(figsize=(3, 3))
    for model_type, group in plot_df.groupby("model_type"):
        _scatter_with_identity(
            ax,
            group["input_mean"],
            group["recon_mean"],
            label=model_type,
            alpha=0.6,
            s=20,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    _save_fig(fig, out_path)


def _plot_feature_scatter(plot_df, out_path, xlabel, ylabel):
    """Scatter of per-feature means."""
    fig, ax = plt.subplots(figsize=(3, 3))
    _scatter_with_identity(
        ax,
        plot_df["input_mean"],
        plot_df["recon_mean"],
        alpha=0.5,
        s=10,
        color="steelblue",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    _save_fig(fig, out_path)


def _generate_reconstruction_plots(data, views, plots_dir):
    """Generate input vs reconstruction scatter plots."""
    conditionals = data["conditionals"]

    for name in views:
        omic_data = data["omics"].get(name, {})

        recon_variants = [
            ("recon", ""),
            ("recon_inf", " (corrected)"),
        ]

        for recon_key, suffix in recon_variants:
            if "input" not in omic_data or recon_key not in omic_data:
                continue

            input_df = omic_data["input"]
            recon_df = omic_data[recon_key]

            common = input_df.index.intersection(recon_df.index)
            if common.empty:
                continue
            inp, rec = input_df.loc[common], recon_df.loc[common]

            sample_df = pd.DataFrame(
                {
                    "input_mean": inp.mean(axis=1),
                    "recon_mean": rec.mean(axis=1),
                }
            )
            common_conditionals = sample_df.index.intersection(conditionals.index)
            if not common_conditionals.empty:
                sample_df = sample_df.loc[common_conditionals]
                sample_df["model_type"] = conditionals.loc[
                    common_conditionals, "model_type"
                ]

                tag = recon_key.replace("recon_inf", "corrected").replace(
                    "recon", "recon"
                )
                _plot_sample_scatter(
                    sample_df,
                    plots_dir / f"input_recon_sample_{name}_{tag}.png",
                    f"Sample mean {name}{suffix} (original)",
                    f"Sample mean {name}{suffix} (reconstructed)",
                )

            common_feats = inp.columns.intersection(rec.columns)
            if not common_feats.empty:
                feat_df = pd.DataFrame(
                    {
                        "input_mean": inp[common_feats].mean(axis=0).values,
                        "recon_mean": rec[common_feats].mean(axis=0).values,
                    }
                )

                tag = recon_key.replace("recon_inf", "corrected").replace(
                    "recon", "recon"
                )
                _plot_feature_scatter(
                    feat_df,
                    plots_dir / f"input_recon_feature_{name}_{tag}.png",
                    f"Feature mean {name}{suffix} (original)",
                    f"Feature mean {name}{suffix} (reconstructed)",
                )


# Clustering metrics


def _compute_clustering_metrics(X, labels, dataset_name, label_type):
    """Compute Calinski-Harabasz and Davies-Bouldin scores."""
    n_samples = len(labels)
    n_unique = len(np.unique(labels))
    # sklearn's check_number_of_labels requires 2 <= n_labels <= n_samples - 1.
    if n_unique < 2 or n_unique > n_samples - 1:
        return {
            "dataset": dataset_name,
            "label_type": label_type,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
        }

    X_scaled = StandardScaler().fit_transform(X)
    return {
        "dataset": dataset_name,
        "label_type": label_type,
        "calinski_harabasz": calinski_harabasz_score(X_scaled, labels),
        "davies_bouldin": davies_bouldin_score(X_scaled, labels),
    }


def _try_compute_metrics(df, labels_series, dataset_name, label_type):
    """Compute clustering metrics if data is available and aligned."""
    if df is None or (hasattr(df, "empty") and df.empty):
        return None
    aligned = df.loc[df.index.intersection(labels_series.index)]
    aligned = aligned.dropna(axis=0, how="any").dropna(axis=1, how="any")
    if aligned.empty:
        return None
    labels = labels_series.loc[aligned.index]
    return _compute_clustering_metrics(
        aligned.values,
        pd.factorize(labels)[0],
        dataset_name,
        label_type,
    )


def _compute_all_clustering_metrics(data, views, conditionals):
    """Compute clustering metrics for all datasets and label types."""
    label_cols = [c for c in ("tissue", "model_type") if c in conditionals.columns]
    rows = []

    for label_col in label_cols:
        labels = conditionals[label_col].dropna()

        for key in ("z", "z_inf"):
            result = _try_compute_metrics(data.get(key), labels, key, label_col)
            if result:
                rows.append(result)

        for name in views:
            omic_data = data["omics"].get(name, {})
            for prefix in ("input", "recon", "recon_inf"):
                result = _try_compute_metrics(
                    omic_data.get(prefix),
                    labels,
                    f"{prefix}_{name}",
                    label_col,
                )
                if result:
                    rows.append(result)

    return rows


# Data loading


def _load_data_files(output_dir, views, data_path):
    """Load latents, reconstructions, and inputs for plotting.

    Conditionals and input data are from the MuData file at data_path.
    """
    import mudata
    from scipy.sparse import issparse

    from mosa.data.io import is_zarr_path

    output_dir = Path(output_dir)

    data_path_obj = Path(data_path)
    with mudata_set_options(pull_on_update=False):
        if is_zarr_path(data_path_obj):
            mdata = mudata.read_zarr(str(data_path))
        else:
            mdata = mudata.read(str(data_path))
    data = {"omics": {}, "conditionals": mdata.obs}

    z_data = _read_split(output_dir, ["full", "train"], "latent")
    if z_data is not None:
        data["z"] = z_data

    z_inf_data = _read_split(output_dir, ["inference"], "latent")
    if z_inf_data is not None:
        data["z_inf"] = z_inf_data

    for name in views:
        omic_data = {}

        if name in mdata.mod:
            X = mdata.mod[name].X
            if issparse(X):
                X = X.toarray()
            omic_data["input"] = pd.DataFrame(
                X,
                index=mdata.mod[name].obs_names,
                columns=mdata.mod[name].var_names,
            )

        recon_data = _read_split(output_dir, ["full", "train"], f"recon_{name}")
        if recon_data is not None:
            omic_data["recon"] = recon_data

        recon_inf_data = _read_split(output_dir, ["inference"], f"recon_{name}")
        if recon_inf_data is not None:
            omic_data["recon_inf"] = recon_inf_data

        data["omics"][name] = omic_data

    return data


# UMAP plot generation


def _make_umap_plot(df, conditionals, palette, title, out_path, pca_components):
    """Compute UMAP and save scatter plot."""
    logger.debug("Computing UMAP: %s (%d samples x %d features)", title, *df.shape)
    df, conditionals = _align_to_conditionals(df, conditionals)
    pca_comp = pca_components if df.shape[1] > pca_components else None
    embedding = compute_umap_embedding(df, pca_components=pca_comp)
    plot_df = pd.concat([embedding, conditionals], axis=1)
    fig, _ = plot_umap(plot_df, palette, title=title)
    _save_fig(fig, out_path)


def _generate_umap_plots(data, views, plots_dir, palette, pca_components):
    """Generate UMAP plots for latent and per-view reconstructions."""
    conditionals = data["conditionals"]

    if "z" in data:
        _make_umap_plot(
            data["z"],
            conditionals,
            palette,
            "Latent UMAP",
            plots_dir / "umap_z.png",
            pca_components,
        )

    for name in views:
        omic_data = data["omics"].get(name, {})

        if "recon" in omic_data:
            _make_umap_plot(
                omic_data["recon"],
                conditionals,
                palette,
                f"Reconstructed {name.upper()} UMAP",
                plots_dir / f"umap_recon_{name}.png",
                pca_components,
            )

        if "recon_inf" in omic_data:
            _make_umap_plot(
                omic_data["recon_inf"],
                conditionals,
                palette,
                f"Reconstructed corrected {name.upper()} UMAP",
                plots_dir / f"umap_recon_corrected_{name}.png",
                pca_components,
            )


def generate_all_plots(
    output_dir, data_cfg, model_cfg=None, palette=None, pca_components=50
):
    """Generate diagnostic plots and clustering metrics.

    Parameters
    ----------
    output_dir : str or Path
        Root output directory with training artifacts.
    data_cfg : DataConfig
        Data configuration (path, views).
    model_cfg : ModelConfig or None
        Model configuration (currently unused; reserved for model-specific plots).
    palette : dict or None
        Base color mapping for known tissues.
    pca_components : int
        PCA dimensions before UMAP.

    Returns
    -------
    Path
        Path to plots directory.
    """
    configure_plot_style()
    output_dir = Path(output_dir)
    _require_training_artifacts(output_dir)
    plots_dir = output_dir / "plots"

    views = list(data_cfg.views)

    logger.debug("Loading data files")
    data = _load_data_files(output_dir, views, data_cfg.path)

    # 'tissue' is optional everywhere else (training warns and carries on), but
    # every plot here is coloured by it.
    conditionals = data["conditionals"]
    if "tissue" not in conditionals.columns:
        raise DataError(
            f"Plotting requires a 'tissue' column in the data's .obs; "
            f"{data_cfg.path} has {sorted(conditionals.columns)}. "
            f"Training and cross-validation do not need it."
        )

    tissues = conditionals["tissue"].dropna().unique()
    palette = build_palette(tissues, base_palette=palette)

    logger.debug("Generating UMAP plots")
    _generate_umap_plots(data, views, plots_dir, palette, pca_components)
    logger.debug("Generating loss plots")
    _generate_loss_plots(output_dir, views, plots_dir)
    logger.debug("Generating reconstruction plots")
    _generate_reconstruction_plots(data, views, plots_dir)

    metrics_rows = _compute_all_clustering_metrics(data, views, data["conditionals"])
    if metrics_rows:
        metrics_out = ensure_dir(output_dir / "metrics")
        pd.DataFrame(metrics_rows).to_csv(
            metrics_out / "clustering_metrics.csv", index=False
        )
    else:
        warnings.warn(
            "Clustering metrics not computed: missing labels or data.", stacklevel=2
        )

    return plots_dir

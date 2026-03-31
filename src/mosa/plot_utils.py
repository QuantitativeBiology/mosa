"""Plotting utilities for MOSA training diagnostics.

Generates UMAP visualizations, loss curves, reconstruction scatter plots,
and clustering quality metrics from training outputs.
"""
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import umap
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

logger = logging.getLogger(__name__)

DEFAULT_PALETTE = {
    "Lung": "#007fff",
    "TCGA-LUAD": "#007fff",
    "TCGA-LUSC": "#007fff",
    "Prostate": "#665d1e",
    "Stomach": "#ffbf00",
    "Central Nervous System": "#fbceb1",
    "TCGA-GBM": "#fbceb1",
    "TCGA-LGG": "#fbceb1",
    "Skin": "#ff033e",
    "TCGA-SKCM": "#ff033e",
    "Bladder": "#ab274f",
    "Haematopoietic and Lymphoid": "#d5e6f7",
    "TCGA-LAML": "#d5e6f7",
    "TCGA-DLBC": "#d5e6f7",
    "Kidney": "#7cb9e8",
    "Thyroid": "#efdecd",
    "Soft Tissue": "#8db600",
    "Head and Neck": "#e9d66b",
    "Ovary": "#b284be",
    "Bone": "#b2beb5",
    "Endometrium": "#10b36f",
    "Breast": "#6e7f80",
    "Pancreas": "#ff7e00",
    "TCGA-PAAD": "#ff7e00",
    "Peripheral Nervous System": "#87a96b",
    "Cervix": "#c9ffe5",
    "Large Intestine": "#9f2b68",
    "TCGA-COAD": "#9f2b68",
    "TCGA-READ": "#9f2b68",
    "Liver": "#00ffff",
    "Vulva": "#008000",
    "Esophagus": "#cd9575",
    "TCGA-ESCA": "#cd9575",
    "Biliary Tract": "#72a0c1",
    "Other tissue": "#a32638",
    "Small Intestine": "#9966cc",
    "Placenta": "#f19cbb",
    "Testis": "#e32636",
    "Adrenal Gland": "#3b7a57",
    "Uterus": "#7a3b5e",
    "Unknown": "#a32638",
    "Eye": "#ff1493",
    "Cell Line": "darkorange",
    "Organoid": "firebrick",
    "Broad": "#32CD32",
    "Sanger": "#FF8D00",
    "Tumor": "darkgray",
}
"""Default tissue/model_type color palette for DepMap data."""

# Visual style per model_type layer in UMAP plots. Tumors are drawn first
# with low alpha as background; cell lines and organoids are drawn on top
# with higher contrast for visibility.
_UMAP_LAYERS = [
    {"model_type": "Tumor",     "marker": "o", "alpha": 0.4, "size": 5,  "zorder": 1, "edgecolor": None,    "linewidth": 0.1},
    {"model_type": "Cell Line", "marker": "o", "alpha": 0.6, "size": 5,  "zorder": 2, "edgecolor": "black", "linewidth": 0.2},
    {"model_type": "Organoid",  "marker": "^", "alpha": 0.9, "size": 10, "zorder": 2, "edgecolor": "black", "linewidth": 0.2},
]


def configure_plot_style():
    """Apply matplotlib styling for publication-ready figures."""
    plt.rcParams.update({
        "figure.figsize": [2.5, 2.5],
        "figure.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": "Arial",
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
    })


def compute_umap_embedding(df, n_neighbors=25, min_dist=0.25, metric="euclidean",
                           n_components=2, random_state=42, pca_components=None):
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
        X = PCA(n_components=pca_components).fit_transform(X)

    embedding = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist,
        metric=metric, n_components=n_components, random_state=random_state,
    ).fit_transform(X)

    return pd.DataFrame(embedding, index=index,
                        columns=[f"UMAP{i+1}" for i in range(n_components)])


def plot_umap(plot_df, palette, title=None):
    """Plot UMAP embedding colored by tissue, shaped by model_type.

    Parameters
    ----------
    plot_df : DataFrame
        Must have columns: UMAP1, UMAP2, tissue, model_type.
    palette : dict
        Color mapping for tissues and model types.
    title : str or None
        Plot title.

    Returns
    -------
    (fig, ax) tuple.
    """
    fig, ax = plt.subplots()

    # Build complete mappings for all model types
    sizes = {layer["model_type"]: layer["size"] for layer in _UMAP_LAYERS}
    markers = {layer["model_type"]: layer["marker"] for layer in _UMAP_LAYERS}

    for layer in _UMAP_LAYERS:
        subset = plot_df[plot_df["model_type"] == layer["model_type"]]
        if subset.empty:
            continue
        scatter_kw = dict(
            data=subset, x="UMAP1", y="UMAP2",
            hue="tissue", palette=palette,
            style="model_type", markers=markers,
            size="model_type", sizes=sizes,
            alpha=layer["alpha"], zorder=layer["zorder"],
            linewidth=layer["linewidth"], legend=False, ax=ax,
        )
        if layer["edgecolor"]:
            scatter_kw["edgecolor"] = layer["edgecolor"]
        sns.scatterplot(**scatter_kw)

    # Model type legend
    legend_specs = [
        ("Tumor",     "o", None),
        ("Cell Line", "o", "black"),
        ("Organoid",  "^", None),
    ]
    type_handles = [
        Line2D([0], [0], marker=m, color="w", label=label,
               markerfacecolor="gray", markersize=6,
               **({"markeredgecolor": ec, "markeredgewidth": 0.6} if ec else {}))
        for label, m, ec in legend_specs
    ]
    legend_markers = ax.legend(
        handles=type_handles, title="Sample Type",
        loc="upper left", bbox_to_anchor=(1.05, 1.0), frameon=False,
    )
    ax.add_artist(legend_markers)

    # Tissue legend
    tissues_present = plot_df["tissue"].unique()
    color_handles = [
        Line2D([0], [0], marker="o", color=palette[t], label=t, linestyle="", markersize=6)
        for t in tissues_present if t in palette
    ]
    ax.legend(
        handles=color_handles, title="Tissue",
        loc="upper left", bbox_to_anchor=(1.5, 1.0),
        ncol=2, columnspacing=0.5, handletextpad=0.3, frameon=False,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    return fig, ax


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_read(parquet_path, csv_path):
    """Try reading parquet first, fall back to CSV."""
    if Path(parquet_path).exists():
        return pd.read_parquet(parquet_path)
    elif Path(csv_path).exists():
        return pd.read_csv(csv_path, index_col=0)
    return None


def _load_samplesheet(path):
    """Load samplesheet CSV with model_id as index."""
    ss = pd.read_csv(path, index_col=0)
    if "model_id" in ss.columns:
        ss = ss.set_index("model_id")
    return ss


def _align_to_samplesheet(df, samplesheet):
    """Keep only samples present in both DataFrames."""
    common = df.index.intersection(samplesheet.index)
    if common.empty:
        return df, samplesheet
    return df.loc[common], samplesheet.loc[common]


def _save_fig(fig, out_path):
    """Save figure and close it."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Saved plot: %s", out_path)


# ---------------------------------------------------------------------------
# Loss plots
# ---------------------------------------------------------------------------

def _load_lightning_metrics(output_dir):
    """Load metrics from latest Lightning log version.

    Returns
    -------
    dict
        Metric name -> DataFrame with columns [epoch, value].
    """
    log_dir = Path(output_dir) / "lightning_logs"
    if not log_dir.exists():
        return {}
    versions = sorted(log_dir.glob("version_*"), key=lambda p: int(p.name.split("_")[1]))
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
        ("train/loss",     "Total VAE Loss", cmap(0)),
        ("train/adv_loss", "Adversarial",    cmap(2)),
        ("train/kl",       "KL Divergence",  cmap(4)),
        ("train/recon",    "MSE",            cmap(6)),
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
        ax.plot(metrics[total_key]["epoch"], metrics[total_key]["value"],
                label="Total", color=cmap(color_idx), linewidth=2)
        color_idx += 2

    for key in group_keys:
        group_name = key[len(group_prefix):]
        ax.plot(metrics[key]["epoch"], metrics[key]["value"],
                label=group_name, color=cmap(color_idx), linewidth=2)
        color_idx += 2

    if has_val:
        ax.plot(metrics[val_key]["epoch"], metrics[val_key]["value"],
                label="Val", color=cmap(color_idx), linewidth=2, linestyle="--")

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
        ("train/kl",        "KL Divergence Loss",       "loss_kl.png",   cmap(4)),
        ("train/adv_loss",  "Adversarial Loss for VAE",  "loss_adv.png",  cmap(2)),
        ("train/disc_loss", "Discriminator Loss",        "loss_disc.png", cmap(2)),
    ]
    for key, title, filename, color in individual_losses:
        df = metrics.get(key)
        if df is not None:
            _plot_single_loss(df, title, plots_dir / filename, color)

    _plot_composite_loss(metrics, plots_dir / "loss_total.png")

    for omic in views:
        _plot_omic_mse(metrics, omic, plots_dir / f"mse_{omic}.png")


# ---------------------------------------------------------------------------
# Reconstruction scatter plots
# ---------------------------------------------------------------------------

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
        _scatter_with_identity(ax, group["input_mean"], group["recon_mean"],
                               label=model_type, alpha=0.6, s=20)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    _save_fig(fig, out_path)


def _plot_feature_scatter(plot_df, out_path, xlabel, ylabel):
    """Scatter of per-feature means."""
    fig, ax = plt.subplots(figsize=(3, 3))
    _scatter_with_identity(ax, plot_df["input_mean"], plot_df["recon_mean"],
                           alpha=0.5, s=10, color="steelblue")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    _save_fig(fig, out_path)


def _generate_reconstruction_plots(data, views, plots_dir):
    """Generate input vs reconstruction scatter plots."""
    samplesheet = data["samplesheet"]

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

            # Sample means (mean across features per sample)
            sample_df = pd.DataFrame({
                "input_mean": inp.mean(axis=1),
                "recon_mean": rec.mean(axis=1),
            })
            common_ss = sample_df.index.intersection(samplesheet.index)
            if not common_ss.empty:
                sample_df = sample_df.loc[common_ss]
                sample_df["model_type"] = samplesheet.loc[common_ss, "model_type"]

                tag = recon_key.replace("recon_inf", "corrected").replace("recon", "recon")
                _plot_sample_scatter(
                    sample_df, plots_dir / f"input_recon_sample_{name}_{tag}.png",
                    f"Sample mean {name}{suffix} (original)",
                    f"Sample mean {name}{suffix} (reconstructed)",
                )

            # Feature means (mean across samples per feature)
            common_feats = inp.columns.intersection(rec.columns)
            if not common_feats.empty:
                feat_df = pd.DataFrame({
                    "input_mean": inp[common_feats].mean(axis=0).values,
                    "recon_mean": rec[common_feats].mean(axis=0).values,
                })

                tag = recon_key.replace("recon_inf", "corrected").replace("recon", "recon")
                _plot_feature_scatter(
                    feat_df, plots_dir / f"input_recon_feature_{name}_{tag}.png",
                    f"Feature mean {name}{suffix} (original)",
                    f"Feature mean {name}{suffix} (reconstructed)",
                )


# ---------------------------------------------------------------------------
# Clustering metrics
# ---------------------------------------------------------------------------

def _compute_clustering_metrics(X, labels, dataset_name, label_type):
    """Compute Calinski-Harabasz and Davies-Bouldin scores."""
    n_unique = len(np.unique(labels))
    if n_unique < 2:
        return {"dataset": dataset_name, "label_type": label_type,
                "calinski_harabasz": np.nan, "davies_bouldin": np.nan}

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
        aligned.values, pd.factorize(labels)[0], dataset_name, label_type,
    )


def _compute_all_clustering_metrics(data, views, samplesheet):
    """Compute clustering metrics for all datasets and label types."""
    label_cols = [c for c in ("tissue", "model_type") if c in samplesheet.columns]
    rows = []

    for label_col in label_cols:
        labels = samplesheet[label_col].dropna()

        for key in ("z", "z_inf"):
            result = _try_compute_metrics(data.get(key), labels, key, label_col)
            if result:
                rows.append(result)

        for name in views:
            omic_data = data["omics"].get(name, {})
            for prefix in ("input", "recon", "recon_inf"):
                result = _try_compute_metrics(
                    omic_data.get(prefix), labels, f"{prefix}_{name}", label_col,
                )
                if result:
                    rows.append(result)

    return rows


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_data_files(output_dir, views, data_path):
    """Load latents, reconstructions, and inputs for plotting.

    Samplesheet and input data are from the MuData file at data_path.
    """
    import mudata
    from scipy.sparse import issparse

    output_dir = Path(output_dir)

    # Load samplesheet from MuData .obs
    mdata = mudata.read(data_path)
    data = {"omics": {}, "samplesheet": mdata.obs}

    # Prefer full pass latents when available; fallback to train split latents.
    z_full_pq = output_dir / "full" / "latent.parquet"
    z_full_csv = output_dir / "full" / "latent.csv"
    z_train_pq = output_dir / "train" / "latent.parquet"
    z_train_csv = output_dir / "train" / "latent.csv"

    z_data = _try_read(z_full_pq, z_full_csv)
    if z_data is None:
        z_data = _try_read(z_train_pq, z_train_csv)
    if z_data is not None:
        data["z"] = z_data

    z_inf_pq = output_dir / "inference" / "latent.parquet"
    z_inf_csv = output_dir / "inference" / "latent.csv"
    z_inf_data = _try_read(z_inf_pq, z_inf_csv)
    if z_inf_data is not None:
        data["z_inf"] = z_inf_data

    for name in views:
        omic_data = {}

        # Load input data from MuData modality
        if name in mdata.mod:
            X = mdata.mod[name].X
            if issparse(X):
                X = X.toarray()
            omic_data["input"] = pd.DataFrame(
                X, index=mdata.mod[name].obs_names, columns=mdata.mod[name].var_names,
            )

        recon_full_pq = output_dir / "full" / f"recon_{name}.parquet"
        recon_full_csv = output_dir / "full" / f"recon_{name}.csv"
        recon_train_pq = output_dir / "train" / f"recon_{name}.parquet"
        recon_train_csv = output_dir / "train" / f"recon_{name}.csv"

        recon_data = _try_read(recon_full_pq, recon_full_csv)
        if recon_data is None:
            recon_data = _try_read(recon_train_pq, recon_train_csv)
        if recon_data is not None:
            omic_data["recon"] = recon_data

        recon_inf_pq = output_dir / "inference" / f"recon_{name}.parquet"
        recon_inf_csv = output_dir / "inference" / f"recon_{name}.csv"
        recon_inf_data = _try_read(recon_inf_pq, recon_inf_csv)
        if recon_inf_data is not None:
            omic_data["recon_inf"] = recon_inf_data

        data["omics"][name] = omic_data

    return data


# ---------------------------------------------------------------------------
# UMAP plot generation
# ---------------------------------------------------------------------------

def _make_umap_plot(df, samplesheet, palette, title, out_path, pca_components):
    """Compute UMAP and save scatter plot."""
    logger.debug("Computing UMAP: %s (%d samples x %d features)", title, *df.shape)
    df, ss = _align_to_samplesheet(df, samplesheet)
    pca_comp = pca_components if df.shape[1] > pca_components else None
    embedding = compute_umap_embedding(df, pca_components=pca_comp)
    plot_df = pd.concat([embedding, ss], axis=1)
    fig, _ = plot_umap(plot_df, palette, title=title)
    _save_fig(fig, out_path)


def _generate_umap_plots(data, views, plots_dir, palette, pca_components):
    """Generate UMAP plots for latent and per-view reconstructions."""
    samplesheet = data["samplesheet"]

    if "z" in data:
        _make_umap_plot(data["z"], samplesheet, palette,
                        "Latent UMAP", plots_dir / "umap_z.png", pca_components)

    for name in views:
        omic_data = data["omics"].get(name, {})

        if "recon" in omic_data:
            _make_umap_plot(omic_data["recon"], samplesheet, palette,
                            f"Reconstructed {name.upper()} UMAP",
                            plots_dir / f"umap_recon_{name}.png", pca_components)

        if "recon_inf" in omic_data:
            _make_umap_plot(omic_data["recon_inf"], samplesheet, palette,
                            f"Reconstructed corrected {name.upper()} UMAP",
                            plots_dir / f"umap_recon_corrected_{name}.png", pca_components)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_all_plots(output_dir, config, palette=None, pca_components=50):
    """Generate diagnostic plots and clustering metrics.

    Parameters
    ----------
    output_dir : str or Path
        Root output directory with training artifacts.
    config : MOSAConfig
        Experiment configuration.
    palette : dict or None
        Color mapping for tissues/model types. Defaults to DEFAULT_PALETTE.
    pca_components : int
        PCA dimensions before UMAP.

    Returns
    -------
    Path
        Path to plots directory.
    """
    configure_plot_style()
    palette = palette or DEFAULT_PALETTE
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"

    logger.debug("Loading data files")
    data = _load_data_files(output_dir, config.views, config.data_path)

    logger.debug("Generating UMAP plots")
    _generate_umap_plots(data, config.views, plots_dir, palette, pca_components)
    logger.debug("Generating loss plots")
    _generate_loss_plots(output_dir, config.views, plots_dir)
    logger.debug("Generating reconstruction plots")
    _generate_reconstruction_plots(data, config.views, plots_dir)

    metrics_rows = _compute_all_clustering_metrics(data, config.views, data["samplesheet"])
    if metrics_rows:
        metrics_out = output_dir / "metrics"
        metrics_out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(metrics_rows).to_csv(metrics_out / "clustering_metrics.csv", index=False)
    else:
        warnings.warn("Clustering metrics not computed: missing labels or data.")

    return plots_dir

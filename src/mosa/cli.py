from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback
import warnings

from mosa.config import CV_STRATEGIES
from mosa.errors import (
    ConfigError,
    DataError,
    MissingFileError,
    MosaError,
)

logger = logging.getLogger(__name__)


def _setup_logging(debug: bool) -> logging.Handler:
    """Configure logging: debug enables detailed logs, suppresses noisy third-party loggers."""
    warnings.filterwarnings(
        "ignore", message="Cannot join columns with the same name", module="mudata"
    )
    warnings.filterwarnings(
        "ignore", message=".*LeafSpec.*is deprecated", module="pytorch_lightning"
    )
    warnings.filterwarnings(
        "ignore", message=".*tensorboardX.*", module="pytorch_lightning"
    )

    # Replace only our handler: repeated main() calls may use a new stderr,
    # while test runners and embedding applications own their own handlers.
    root = logging.getLogger()
    for handler in root.handlers[:]:
        if handler.name == "mosa.cli":
            root.removeHandler(handler)
            handler.close()
    handler = logging.StreamHandler(sys.stderr)
    handler.name = "mosa.cli"
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            if debug
            else "[%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(logging.WARNING)
    level = logging.DEBUG if debug else logging.INFO
    if int(os.environ.get("LOCAL_RANK", 0)) != 0:
        level = logging.WARNING
    logging.getLogger("mosa").setLevel(level)
    logging.getLogger("pytorch_lightning").setLevel(
        logging.INFO if debug else logging.WARNING
    )
    for name in (
        "torch",
        "matplotlib",
        "numba",
        "fsspec",
        "numcodecs",
        "h5py",
        "zarr",
        "asyncio",
        "optuna",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
    logger.debug("Debug logging enabled")
    return handler


def _load_config_and_data(config_path):
    """Load a config and its MuData, emitting data-validation warnings."""
    from mosa.data.io import load_mudata
    from mosa.utils import load_config, validate_config_against_data

    cfg = load_config(config_path)
    for w in validate_config_against_data(cfg):
        logger.warning(w)
    dataset = load_mudata(cfg.data.path, cfg.data.views, cfg.data.mask_layer_name)
    return cfg, dataset


def _train(args):
    """Load config and data, split into train/val, and fit the model."""
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.model_selection import train_test_split

    from mosa.models.registry import build_model
    from mosa.utils import seed_everything

    torch.set_float32_matmul_precision("high")
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

    cfg, dataset = _load_config_and_data(args.config)
    seed_everything(cfg.model.random_seed)

    train_data = dataset
    val_data = None
    if cfg.evaluation.test_size > 0:
        classes, counts = np.unique(dataset.metadata["model_type"], return_counts=True)
        if counts.min() < 2:
            smallest = classes[np.argmin(counts)]
            raise DataError(
                f"Cannot create a stratified train/val split: model_type class "
                f"'{smallest}' has only {counts.min()} sample(s); need at least 2."
            )

        label_codes = pd.Categorical(
            dataset.metadata["model_type"],
            categories=sorted(dataset.metadata["model_type"].unique()),
            ordered=True,
        ).codes

        train_idx, val_idx = train_test_split(
            np.arange(dataset.n_samples),
            test_size=cfg.evaluation.test_size,
            random_state=cfg.model.random_seed,
            stratify=label_codes,
        )
        train_data = dataset.subset(train_idx)
        val_data = dataset.subset(val_idx)

    logger.info(
        "Split: %d training samples, %d validation samples%s",
        train_data.n_samples,
        val_data.n_samples if val_data is not None else 0,
        " (stratified by model_type)"
        if val_data is not None
        else " (using all samples)",
    )
    model = build_model(cfg.data, cfg.model)
    logger.info("Selected model: %s", model.registered_name)
    model.fit(train_data, val_data, resume_from=args.resume)

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        model.save_outputs()


def _transform(args):
    """Load a saved model and project data into the latent space."""
    from pathlib import Path

    import pandas as pd

    from mosa.data.io import load_mudata, summarize_structure
    from mosa.models.registry import load_model
    from mosa.utils import ensure_dir

    for label, path in (("Checkpoint", args.checkpoint), ("Input data", args.input)):
        p = Path(path)
        if not (p.is_file() or p.is_dir()):
            raise MissingFileError(f"{label} not found: {path}")

    logger.info("Loading saved model from %s", args.checkpoint)
    model = load_model(args.checkpoint)
    logger.info("Checking input views against the saved model")

    # Name the mismatch here: load_mudata would report only the first missing
    # view, without saying what the checkpoint was trained on.
    available = list(summarize_structure(args.input).get("modalities", {}))
    missing = [v for v in model.data_cfg.views if v not in available]
    if missing:
        raise DataError(
            f"Input {args.input} is missing view(s) {missing} required by the "
            f"checkpoint. Checkpoint was trained on {list(model.data_cfg.views)}; "
            f"input has {available}."
        )

    dataset = load_mudata(
        args.input,
        model.data_cfg.views,
        model.data_cfg.mask_layer_name,
    )

    # Compute everything first: a failed run must leave a previous run's
    # outputs untouched.
    logger.info("Projecting %d samples into the latent space", dataset.n_samples)
    z = model.transform(dataset)
    if args.reconstruct:
        logger.info("Reconstructing %d views", len(dataset.view_names))
    recon = model.reconstruct(dataset) if args.reconstruct else {}

    logger.info(
        "Writing outputs to %s (replacing existing projection files)", args.output
    )
    out_dir = ensure_dir(args.output)
    # A previous transform into this directory may have written views this
    # model does not have, or reconstructions this run does not ask for.
    for stale in out_dir.glob("recon_*.parquet"):
        stale.unlink()
    pd.DataFrame(z, index=dataset.sample_names).to_parquet(out_dir / "latent.parquet")
    print(f"Latent representations saved to {out_dir / 'latent.parquet'}")

    for omic, arr in recon.items():
        df = pd.DataFrame(
            arr, index=dataset.sample_names, columns=dataset.feature_names[omic]
        )
        df.to_parquet(out_dir / f"recon_{omic}.parquet")
    if recon:
        print(f"Reconstructions saved to {out_dir}/")


def _eval_cfg_from_args(cfg, args):
    """Apply the cross-validation CLI flags that were given onto cfg.evaluation."""
    import dataclasses

    overrides = {}
    if args.folds is not None:
        overrides["n_folds"] = args.folds
    if args.strategy is not None:
        overrides["strategy"] = args.strategy
    if args.no_shuffle:
        overrides["shuffle"] = False
    if overrides:
        logger.debug("Evaluation overrides from CLI: %s", overrides)
    return dataclasses.replace(cfg.evaluation, **overrides)


def _save_cv_outputs(results, dataset, output_dir):
    """Write per-fold-per-epoch train/val loss and out-of-sample reconstructions.

    history.csv has one row per (fold, epoch) with train_loss/val_loss, so it
    can be grouped by epoch and averaged across folds to plot learning
    curves. recon_<view>.parquet holds the out-of-sample reconstruction for
    every sample (see cross_validate's docstring), aligned to dataset order,
    ready to compare against the original data.
    """
    import pandas as pd

    from mosa.utils import ensure_dir

    output_dir = ensure_dir(output_dir)
    logger.info(
        "Writing cross-validation history and reconstructions to %s", output_dir
    )

    history_rows = [
        {"fold": fold_idx, **epoch_entry}
        for fold_idx, fold in enumerate(results["per_fold"])
        for epoch_entry in fold["epoch_history"]
    ]
    if history_rows:
        pd.DataFrame(history_rows).to_csv(output_dir / "history.csv", index=False)

    for view, recon in results["reconstructions"].items():
        cols = dataset.feature_names.get(view)
        df = pd.DataFrame(recon, index=dataset.sample_names, columns=cols)
        df.to_parquet(output_dir / f"recon_{view}.parquet")


def _cross_validate(args):
    """Load config and data, run k-fold cross-validation, and print scores."""
    import torch

    from mosa.models.evaluation import cross_validate
    from mosa.utils import seed_everything

    torch.set_float32_matmul_precision("high")

    cfg, dataset = _load_config_and_data(args.config)
    seed_everything(cfg.model.random_seed)
    eval_cfg = _eval_cfg_from_args(cfg, args)
    results = cross_validate(dataset, cfg.data, cfg.model, eval_cfg)

    if getattr(cfg.model, "output_dir", None):
        _save_cv_outputs(results, dataset, cfg.model.output_dir)
        print(
            f"Per-epoch history and out-of-sample reconstructions written to {cfg.model.output_dir}\n"
        )

    views = list(results["per_view"].keys())
    n_views = len(views)

    def cell(value):
        return f"{'n/a':<20}" if value != value else f"{value:<20.4f}"

    print(f"{'fold':<7}" + "".join(f"{v + ' (NMSE)':<20}" for v in views) + "aggregate")
    for i, fold in enumerate(results["per_fold"]):
        row = f"{i:<7}" + "".join(cell(fold["per_view"][v]["nmse"]) for v in views)
        row += f"{fold['aggregate']:<12.4f}"
        if fold["n_views"] < n_views:
            row += f"({fold['n_views']}/{n_views} views)"
        print(row)

    print(
        f"{'pooled':<7}"
        + "".join(cell(results["per_view"][v]["nmse"]) for v in views)
        + f"{results['mean']:<12.4f}"
    )

    if any(f["n_views"] < n_views for f in results["per_fold"]):
        print(
            "\nSome folds observed no entries for a view (n/a above). Their "
            "aggregates cover fewer views and are not comparable to each other."
        )
    print(f"\nAggregate NMSE (pooled over folds): {results['mean']:.4f}")
    print(
        f"Per-fold spread (diagnostic, not an error bar on the above): ±{results['std']:.4f}"
    )


def _optimize(args):
    """Load config and data, run Optuna hyperparameter search, and print the best trial."""
    import torch

    from mosa.models.optimize import load_search_space, optimize
    from mosa.utils import seed_everything

    torch.set_float32_matmul_precision("high")

    # A bad search space should not cost a full MuData load to discover.
    logger.info("Loading search space from %s", args.search_space)
    search_space = load_search_space(args.search_space)

    cfg, dataset = _load_config_and_data(args.config)
    seed_everything(cfg.model.random_seed)

    results = optimize(
        dataset,
        cfg.data,
        cfg.model,
        search_space,
        n_trials=args.trials,
        eval_cfg=_eval_cfg_from_args(cfg, args),
    )
    study = results["study"]

    print(f"Trials: {len(study.trials)} (n_trials={args.trials})")
    pruned = sum(1 for t in study.trials if t.state.name == "PRUNED")
    completed = sum(1 for t in study.trials if t.state.name == "COMPLETE")
    print(f"  completed: {completed}, pruned: {pruned}")
    print(f"\nBest value (mean aggregate NMSE): {results['best_value']:.4f}")
    print("Best params:")
    for name, value in results["best_params"].items():
        print(f"  {name}: {value}")


def _plot(args):
    """Generate diagnostic plots from a completed training run."""
    from mosa.plot import generate_all_plots
    from mosa.utils import load_config

    cfg = load_config(args.config)
    output_dir = args.output_dir or cfg.model.output_dir
    logger.info("Generating plots from %s", output_dir)

    plots_dir = generate_all_plots(output_dir, cfg.data, cfg.model)
    print(f"Plots saved to {plots_dir}")


def _convert(args):
    """Convert CSV dataset to MuData (.h5mu) format."""
    from mosa.data.io import csv_to_mudata

    view_specs = []
    for spec in args.view:
        if ":" not in spec:
            raise ConfigError(
                f"Invalid --view format: '{spec}'. Expected 'name:path' "
                f"(e.g. 'gexp_voom:data/gexp_voom.csv')"
            )
        name, path = spec.split(":", 1)
        view_specs.append((name, path))

    metadata_filters: dict[str, list[str]] = {}
    for spec in args.filter or []:
        if "=" not in spec:
            raise ConfigError(
                f"Invalid --filter format: '{spec}'. Expected 'COLUMN=VAL[,VAL...]' "
                f"(e.g. 'model_type=Cell_Line,Organoid')"
            )
        column, values = spec.split("=", 1)
        metadata_filters[column] = [v.strip() for v in values.split(",") if v.strip()]

    logger.debug(
        "Conversion settings: format=%s, min_views=%d, shared_features=%s, on_collision=%s, filters=%s",
        args.format,
        args.min_views,
        args.shared_features,
        args.on_collision,
        metadata_filters,
    )

    csv_to_mudata(
        conditionals_path=args.conditionals,
        view_specs=view_specs,
        output_path=args.output,
        mutations_path=args.mutations,
        format=args.format,
        id_map_path=args.id_map,
        on_collision=args.on_collision,
        min_views=args.min_views,
        metadata_filters=metadata_filters,
        shared_features=args.shared_features,
    )
    print(f"MuData file saved to {args.output}")


def _inspect(args):
    """Print a summary of a MuData file."""
    from mosa.data.io import inspect_mudata

    inspect_mudata(args.input)


def _validate(args):
    """Validate a YAML config, including that the data satisfies the model's requirements."""
    from mosa.utils import load_config, validate_config_against_data

    cfg = load_config(args.config)
    data_warnings = validate_config_against_data(cfg)

    for w in data_warnings:
        logger.warning(w)

    print("Config OK")
    print(f"  data:    {cfg.data.path}")
    print(f"  views:   {cfg.data.views} (discrete: {sorted(cfg.data.discrete_views)})")
    print(f"  model:   {type(cfg.model).__name__}")
    print(f"  output:  {cfg.model.output_dir}")
    print(f"  seed:    {cfg.model.random_seed}")
    print(
        f"  eval:    test_size={cfg.evaluation.test_size}, "
        f"n_folds={cfg.evaluation.n_folds}, strategy={cfg.evaluation.strategy}, "
        f"shuffle={cfg.evaluation.shuffle}"
    )

    from mosa.models.mosa.config import MOSAConfig

    if isinstance(cfg.model, MOSAConfig):
        print(
            f"  arch:    fusion={cfg.model.fusion_method}, latent={cfg.model.joint_latent_dim}"
        )
        print(
            f"  train:   epochs={cfg.model.num_epochs}, batch_size={cfg.model.batch_size}"
        )


def _add_eval_args(parser):
    """Add the cross-validation flags that override a config's evaluation block.

    Defaults are None/False so an unset flag leaves the config value alone.
    """
    parser.add_argument(
        "-k",
        "--folds",
        type=int,
        default=None,
        metavar="N",
        help="Number of held-out folds per evaluation (default: "
        "evaluation.n_folds in YAML)",
    )
    parser.add_argument(
        "-s",
        "--strategy",
        choices=CV_STRATEGIES,
        default=None,
        help="Fold assignment: stratified preserves model_type source "
        "proportions; kfold ignores source labels (overrides "
        "evaluation.strategy)",
    )
    parser.add_argument(
        "-N",
        "--no-shuffle",
        action="store_true",
        help="Keep input order for fold assignment (overrides "
        "evaluation.shuffle); stratified folds still balance model_type",
    )


def _add_config_arg(parser):
    parser.add_argument_group("required inputs").add_argument(
        "-c",
        "--config",
        required=True,
        metavar="PATH",
        help="Experiment YAML; file paths are relative to the working directory",
    )


def _add_debug_arg(parser):
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Include effective settings, preprocessing details, timings, and "
        "error tracebacks",
    )


def _add_command(subparsers, name, summary, description, example, doc_page):
    """Add concise command help with an example and its documentation page."""
    return subparsers.add_parser(
        name,
        help=summary,
        description=description,
        epilog=f"example:\n  {example}\n\nDocumentation:\n"
        f"  docs/reference/commands/{doc_page}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the mosa argument parser with every subcommand."""
    parser = argparse.ArgumentParser(
        prog="mosa",
        description="MOSA is a Python library for multi-omics integration, with commands for\n"
        "preparing datasets, training and evaluating models, optimizing hyperparameters,\n"
        "and analysing results.\n\n"
        "Its main model, the MOSA VAE, is a conditional multiview variational\n"
        "autoencoder for bulk multi-omics data, with optional adversarial training.\n"
        "Select it, the optional MOFA integration, or registered custom models with\n"
        "model.type in a YAML configuration.",
        epilog="Run 'mosa <command> -h' for command details and options.\n\n"
        "Documentation:\n  docs/index.md",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = _add_command(
        subparsers,
        "train",
        "Train a model and save its results",
        "Fit the model selected by model.type to the configured dataset, using\n"
        "evaluation.test_size to reserve validation samples. Save model outputs to\n"
        "model.output_dir. The selected model determines its preprocessing, use of\n"
        "validation, and saved-model format.",
        "mosa train -c config.yaml",
        "train.md",
    )
    _add_config_arg(train_parser)
    train_parser.add_argument(
        "-r",
        "--resume",
        default=None,
        metavar="CKPT",
        help="Continue MOSA VAE training from a Lightning checkpoint (.ckpt)",
    )
    _add_debug_arg(train_parser)

    transform_parser = _add_command(
        subparsers,
        "transform",
        "Project samples with a saved model",
        "Apply a saved model to new samples to obtain coordinates in its learned\n"
        "latent space, with --reconstruct to estimate omic profiles. Reuse fitted\n"
        "preprocessing without updating the model; views and feature order must match\n"
        "training. Write predictions to --output.",
        "mosa transform -m run/checkpoints/last.ckpt -i data.h5mu -o projected/ -r",
        "transform.md",
    )
    transform_inputs = transform_parser.add_argument_group("required inputs")
    transform_inputs.add_argument(
        "-m",
        "--checkpoint",
        required=True,
        metavar="PATH",
        help="Saved model file, such as VAE .ckpt/.pt or MOFA .hdf5",
    )
    transform_inputs.add_argument(
        "-i",
        "--input",
        required=True,
        metavar="PATH",
        help="Prepared MuData dataset (.h5mu file or .zarr directory)",
    )
    transform_inputs.add_argument(
        "-o",
        "--output",
        required=True,
        metavar="DIR",
        help="Directory for latent.parquet and optional reconstruction files",
    )
    transform_parser.add_argument(
        "-r",
        "--reconstruct",
        action="store_true",
        help="Also save predicted measurements for each view as recon_<view>.parquet",
    )
    _add_debug_arg(transform_parser)

    cv_parser = _add_command(
        subparsers,
        "cross-validate",
        "Evaluate reconstruction on held-out samples",
        "Compare configurations by reconstructing samples excluded from fitting.\n"
        "Train a fresh model per fold and report observed-entry normalized mean\n"
        "squared error (NMSE), where lower is better. Save held-out reconstructions\n"
        "and available epoch histories to model.output_dir; fold models are temporary.",
        "mosa cross-validate -c config.yaml -k 5",
        "cross-validate.md",
    )
    _add_config_arg(cv_parser)
    _add_eval_args(cv_parser)
    _add_debug_arg(cv_parser)

    optimize_parser = _add_command(
        subparsers,
        "optimize",
        "Optimize model parameters by cross-validation",
        "Use Optuna to sample model parameters from a search-space YAML file.\n"
        "Evaluate each combination by cross-validation and print the best parameters\n"
        "and pooled aggregate NMSE. Apply the selected values to your configuration,\n"
        "then run mosa train to save a fitted model.",
        "mosa optimize -c config.yaml -p search_space.yaml -n 50",
        "optimize.md",
    )
    _add_config_arg(optimize_parser)
    optimize_parser.add_argument_group("search input").add_argument(
        "-p",
        "--search-space",
        required=True,
        metavar="PATH",
        help="YAML distributions for direct model fields (e.g. "
        "configs/search_space.yaml)",
    )
    optimize_parser.add_argument(
        "-n",
        "--trials",
        type=int,
        default=20,
        metavar="N",
        help="Parameter combinations to evaluate; each runs cross-validation "
        "(default: 20)",
    )
    _add_eval_args(optimize_parser)
    _add_debug_arg(optimize_parser)

    plot_parser = _add_command(
        subparsers,
        "plot",
        "Plot sample representations, reconstruction, and training losses",
        "Examine a completed run's sample representations, reconstruction, and\n"
        "training behaviour. Read the configured dataset and saved outputs, then\n"
        "write figures to plots/ and label-separation scores to metrics/ within the\n"
        "run directory. Requires tissue metadata; available figures depend on the\n"
        "saved outputs and logs.",
        "mosa plot -c config.yaml",
        "plot.md",
    )
    _add_config_arg(plot_parser)
    plot_parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Run directory to read outputs from and save plots to (default: "
        "model.output_dir)",
    )
    _add_debug_arg(plot_parser)

    convert_parser = _add_command(
        subparsers,
        "convert",
        "Align omic tables and sample metadata into a dataset",
        "Prepare a dataset from omic tables and sample metadata by aligning sample\n"
        "identifiers and recording missing measurements. Each view table has features\n"
        "as rows and samples as columns. Write MuData to --output; supplied values\n"
        "are preserved without normalization, scaling, or imputation.",
        "mosa convert -m meta.csv -v gexp:gexp.csv -v meth:meth.csv -o data.h5mu",
        "convert.md",
    )
    convert_inputs = convert_parser.add_argument_group("required inputs and output")
    convert_inputs.add_argument(
        "-m",
        "--conditionals",
        required=True,
        metavar="PATH",
        help="Sample metadata table with model_id and model_type columns; "
        "tissue is optional",
    )
    convert_inputs.add_argument(
        "-v",
        "--view",
        required=True,
        action="append",
        metavar="NAME:PATH",
        help="Omic table as name:path (e.g. gexp:data/gexp.csv). Repeat for "
        "each view. Repeating a name combines samples and retains the "
        "union of features. Formats: .csv, .tsv, .txt, .tab, .parquet, "
        ".pq; delimited files may be compressed.",
    )
    convert_parser.add_argument(
        "-M",
        "--mutations",
        default=None,
        metavar="PATH",
        help="Binary mutation table (features x samples); rows become "
        "mutation_* metadata columns",
    )
    convert_parser.add_argument(
        "-I",
        "--id-map",
        default=None,
        metavar="PATH",
        help="Table mapping source_id to model_id in omic tables before "
        "alignment; metadata must already use model_id identifiers",
    )
    convert_parser.add_argument(
        "-x",
        "--on-collision",
        choices=["error", "first"],
        default="error",
        help="Duplicate sample IDs within a view: reject them or keep the first"
        " occurrence (default: error)",
    )
    convert_parser.add_argument(
        "-n",
        "--min-views",
        type=int,
        default=1,
        metavar="N",
        help="For N > 1, require observed measurements in at least N views; 1 "
        "keeps all matched samples (default: 1)",
    )
    convert_parser.add_argument(
        "-F",
        "--filter",
        action="append",
        default=None,
        metavar="COLUMN=VAL[,VAL...]",
        help="Keep samples whose metadata matches COLUMN=VAL[,VAL...] (e.g. "
        "tissue=Lung,Breast). Filters on different columns combine with "
        "AND; repeating a column replaces its earlier filter.",
    )
    convert_parser.add_argument(
        "-s",
        "--shared-features",
        action="store_true",
        help="Keep feature identifiers present in every view; requires a shared"
        " identifier namespace, such as genes",
    )
    convert_inputs.add_argument(
        "-o",
        "--output",
        required=True,
        metavar="PATH",
        help="Destination .h5mu file or .zarr directory; use --format zarr for Zarr",
    )
    convert_parser.add_argument(
        "-f",
        "--format",
        choices=["h5mu", "zarr"],
        default="h5mu",
        help="Output format (default: h5mu)",
    )
    _add_debug_arg(convert_parser)

    inspect_parser = _add_command(
        subparsers,
        "inspect",
        "Inspect dataset dimensions, metadata, and missingness",
        "Check a prepared MuData dataset before configuring a model. Print sample\n"
        "and feature counts, value ranges, missingness, and metadata so you can\n"
        "compare the saved dataset with your input tables. The input is not modified.",
        "mosa inspect -i data.h5mu",
        "inspect.md",
    )
    inspect_parser.add_argument_group("required inputs").add_argument(
        "-i",
        "--input",
        required=True,
        metavar="PATH",
        help="Prepared MuData dataset (.h5mu file or .zarr directory)",
    )
    _add_debug_arg(inspect_parser)

    validate_parser = _add_command(
        subparsers,
        "validate",
        "Check configuration and dataset compatibility",
        "Check a YAML configuration against its dataset before fitting. Validate\n"
        "paths, view names, observed-value masks, metadata, and model-specific\n"
        "requirements. Print a configuration summary and warnings, or report an\n"
        "error with exit status 1, without training or writing model outputs.",
        "mosa validate -c config.yaml",
        "validate.md",
    )
    _add_config_arg(validate_parser)
    _add_debug_arg(validate_parser)

    return parser


def main(argv=None):
    """CLI entry point. argv defaults to sys.argv[1:]; tests pass it explicitly."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = _setup_logging(args.debug)

    handlers = {
        "train": _train,
        "transform": _transform,
        "cross-validate": _cross_validate,
        "optimize": _optimize,
        "plot": _plot,
        "convert": _convert,
        "inspect": _inspect,
        "validate": _validate,
    }

    try:
        started = time.perf_counter()
        logger.info("Starting %s", args.command)
        handlers[args.command](args)
    except MosaError as e:
        # Anything not deriving from MosaError is a bug: let it traceback.
        print(f"Error: {e}", file=sys.stderr)
        # getattr, not args.debug: a subcommand without the flag must not
        # raise AttributeError from inside the error handler.
        if getattr(args, "debug", False):
            traceback.print_exc()
        sys.exit(1)
    except KeyboardInterrupt:
        # Only reached outside Lightning (data loading, convert, inspect,
        # transform). Lightning traps SIGINT inside fit() and raises
        # SystemExit(1) itself, which passes straight through here.
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
    else:
        logger.info(
            "%s completed in %.1fs", args.command, time.perf_counter() - started
        )
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


if __name__ == "__main__":
    main()

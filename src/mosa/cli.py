from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings

from mosa.config import CV_STRATEGIES

logger = logging.getLogger(__name__)


def _setup_logging(debug: bool):
    """Configure logging: debug enables detailed logs, suppresses noisy third-party loggers."""
    warnings.filterwarnings("ignore", message="Cannot join columns with the same name", module="mudata")
    warnings.filterwarnings("ignore", message=".*LeafSpec.*is deprecated", module="pytorch_lightning")
    warnings.filterwarnings("ignore", message=".*tensorboardX.*", module="pytorch_lightning")

    if debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
        logging.getLogger("torch").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("numba").setLevel(logging.WARNING)
        logging.getLogger("fsspec").setLevel(logging.WARNING)
        logging.getLogger("numcodecs").setLevel(logging.WARNING)
        logging.getLogger("h5py").setLevel(logging.WARNING)
        logging.getLogger("zarr").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)
        logger.debug("Debug logging enabled")
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format="[%(levelname)s] %(message)s",
        )
        logging.getLogger("mosa").setLevel(logging.INFO)
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


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

    if int(os.environ.get("LOCAL_RANK", 0)) != 0:
        logging.getLogger("mosa").setLevel(logging.WARNING)

    cfg, dataset = _load_config_and_data(args.config)
    seed_everything(cfg.model.random_seed)

    train_data = dataset
    val_data = None
    if cfg.evaluation.test_size > 0:
        classes, counts = np.unique(dataset.metadata["model_type"], return_counts=True)
        if counts.min() < 2:
            smallest = classes[np.argmin(counts)]
            raise ValueError(
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

    model = build_model(cfg.data, cfg.model)
    model.fit(train_data, val_data, resume_from=args.resume)

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        model.save_outputs()
        logger.debug("Training complete")


def _transform(args):
    """Load a saved model and project data into the latent space."""
    import pandas as pd

    from mosa.data.io import load_mudata
    from mosa.models.registry import load_model
    from mosa.utils import ensure_dir

    model = load_model(args.checkpoint)
    dataset = load_mudata(
        args.input,
        model.data_cfg.views,
        model.data_cfg.mask_layer_name,
    )

    out_dir = ensure_dir(args.output)

    z = model.transform(dataset)
    pd.DataFrame(z, index=dataset.sample_names).to_parquet(out_dir / "latent.parquet")
    print(f"Latent representations saved to {out_dir / 'latent.parquet'}")

    if args.reconstruct:
        recon = model.reconstruct(dataset)
        for omic, arr in recon.items():
            df = pd.DataFrame(arr, index=dataset.sample_names)
            df.to_parquet(out_dir / f"recon_{omic}.parquet")
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
    return dataclasses.replace(cfg.evaluation, **overrides)


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

    views = list(results["per_fold"][0]["per_view"].keys())
    header = f"{'fold':<6}" + "".join(f"{v + ' (NMSE)':<20}" for v in views) + f"{'aggregate':<12}"
    print(header)
    for i, fold in enumerate(results["per_fold"]):
        row = f"{i:<6}"
        for v in views:
            row += f"{fold['per_view'][v]['nmse']:<20.4f}"
        row += f"{fold['aggregate']:<12.4f}"
        print(row)
    print(f"\nmean ± std (aggregate NMSE): {results['mean']:.4f} ± {results['std']:.4f}")


def _optimize(args):
    """Load config and data, run Optuna hyperparameter search, and print the best trial."""
    import torch

    from mosa.models.optimize import load_search_space, optimize
    from mosa.utils import seed_everything

    torch.set_float32_matmul_precision("high")

    cfg, dataset = _load_config_and_data(args.config)
    seed_everything(cfg.model.random_seed)
    search_space = load_search_space(args.search_space)

    results = optimize(
        dataset, cfg.data, cfg.model, search_space,
        n_trials=args.trials, eval_cfg=_eval_cfg_from_args(cfg, args),
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
    logger.debug("Generating plots from %s", output_dir)

    plots_dir = generate_all_plots(output_dir, cfg.data, cfg.model)
    print(f"Plots saved to {plots_dir}")


def _convert(args):
    """Convert CSV dataset to MuData (.h5mu) format."""
    from mosa.data.io import csv_to_mudata

    view_specs = []
    for spec in args.view:
        if ":" not in spec:
            raise ValueError(
                f"Invalid --view format: '{spec}'. Expected 'name:path' "
                f"(e.g. 'gexp_voom:data/gexp_voom.csv')"
            )
        name, path = spec.split(":", 1)
        view_specs.append((name, path))

    metadata_filters: dict[str, list[str]] = {}
    for spec in args.filter or []:
        if "=" not in spec:
            raise ValueError(
                f"Invalid --filter format: '{spec}'. Expected 'COLUMN=VAL[,VAL...]' "
                f"(e.g. 'model_type=Cell_Line,Organoid')"
            )
        column, values = spec.split("=", 1)
        metadata_filters[column] = [v.strip() for v in values.split(",") if v.strip()]

    logger.debug("Converting CSV dataset to MuData format")
    logger.debug("Output file: %s", args.output)

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

    try:
        cfg = load_config(args.config)
        data_warnings = validate_config_against_data(cfg)
    except (ValueError, FileNotFoundError, KeyError, TypeError) as e:
        print(f"Config invalid: {e}")
        sys.exit(1)

    for w in data_warnings:
        print(f"Warning: {w}")

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
        print(f"  arch:    fusion={cfg.model.fusion_method}, latent={cfg.model.joint_latent_dim}")
        print(f"  train:   epochs={cfg.model.num_epochs}, batch_size={cfg.model.batch_size}")


def _add_eval_args(parser):
    """Add the cross-validation flags that override a config's evaluation block.

    Defaults are None/False so an unset flag leaves the config value alone.
    """
    parser.add_argument(
        "--folds", type=int, default=None,
        help="Number of folds, per trial for optimize (overrides evaluation.n_folds)",
    )
    parser.add_argument(
        "--strategy", choices=CV_STRATEGIES, default=None,
        help="Fold assignment: stratified balances model_type, kfold ignores it "
             "(overrides evaluation.strategy)",
    )
    parser.add_argument(
        "--no-shuffle", action="store_true",
        help="Assign folds as contiguous blocks of sample order instead of shuffling",
    )


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MOSA: Multi-Omic Synthetic Augmentation",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config", required=True, help="Path to YAML config file")
    train_parser.add_argument("--resume", default=None, metavar="CKPT",
                              help="Resume training from a Lightning checkpoint (.ckpt)")
    train_parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")

    transform_parser = subparsers.add_parser(
        "transform", help="Project data into the latent space using a saved model",
    )
    transform_parser.add_argument("--checkpoint", required=True, help="Path to saved model checkpoint (.ckpt)")
    transform_parser.add_argument("--input", required=True, help="Path to .h5mu or .zarr input data")
    transform_parser.add_argument("--output", required=True, help="Directory to write latent.parquet (and reconstructions)")
    transform_parser.add_argument("--reconstruct", action="store_true",
                                  help="Also write per-omic reconstruction parquets")
    transform_parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")

    cv_parser = subparsers.add_parser(
        "cross-validate", help="Run k-fold cross-validation",
    )
    cv_parser.add_argument("--config", required=True, help="Path to YAML config file")
    _add_eval_args(cv_parser)
    cv_parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")

    optimize_parser = subparsers.add_parser(
        "optimize", help="Run Optuna hyperparameter search over a config",
    )
    optimize_parser.add_argument("--config", required=True, help="Path to YAML config file")
    optimize_parser.add_argument(
        "--search-space", required=True, help="Path to search-space YAML (see configs/search_space.yaml)",
    )
    optimize_parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials (default: 20)")
    _add_eval_args(optimize_parser)
    optimize_parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")

    plot_parser = subparsers.add_parser("plot", help="Generate diagnostic plots from training outputs")
    plot_parser.add_argument("--config", required=True, help="Path to YAML config file")
    plot_parser.add_argument(
        "--output-dir",
        default=None,
        help="Path to training output directory (defaults to model.output_dir in config)",
    )
    plot_parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")

    convert_parser = subparsers.add_parser(
        "convert", help="Convert CSV files to MuData (.h5mu or .zarr)",
    )
    convert_parser.add_argument(
        "--conditionals", required=True,
        help="Path to conditionals CSV (required columns: model_id, model_type; optional: tissue)",
    )
    convert_parser.add_argument(
        "--view", required=True, action="append",
        help="View spec as 'name:path' (e.g. 'gexp:data/gexp.parquet'). Repeat for each "
             "modality. Repeating the same name assembles that omic from several files: "
             "samples concatenate, features union. Formats: .csv, .tsv, .txt, .parquet "
             "(delimited ones may be .gz).",
    )
    convert_parser.add_argument(
        "--mutations", default=None,
        help="Path to mutations CSV (features x samples, binary). Columns become mutation_* in .obs.",
    )
    convert_parser.add_argument(
        "--id-map", default=None,
        help="Sample-ID crosswalk table (columns: source_id, model_id) applied to every "
             "view before alignment. Use it when providers name the same sample differently.",
    )
    convert_parser.add_argument(
        "--on-collision", choices=["error", "first"], default="error",
        help="What to do when two columns resolve to one sample ID (default: error).",
    )
    convert_parser.add_argument(
        "--min-views", type=int, default=1, metavar="N",
        help="Keep only samples with data in at least N views (default: 1, keep all).",
    )
    convert_parser.add_argument(
        "--filter", action="append", default=None, metavar="COLUMN=VAL[,VAL...]",
        help="Restrict samples to metadata rows whose COLUMN is one of the listed values "
             "(e.g. 'model_type=Cell_Line,Organoid'). Repeat for several columns.",
    )
    convert_parser.add_argument(
        "--shared-features", action="store_true",
        help="Reduce every view to the features they all share. Only valid when all "
             "views use one identifier namespace (e.g. every omic at gene level).",
    )
    convert_parser.add_argument("--output", required=True, help="Output file path (.h5mu or .zarr)")
    convert_parser.add_argument(
        "--format", choices=["h5mu", "zarr"], default="h5mu",
        help="Output format (default: h5mu)",
    )
    convert_parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")

    inspect_parser = subparsers.add_parser(
        "inspect", help="Print a summary of a MuData file (.h5mu or .zarr)",
    )
    inspect_parser.add_argument("--input", required=True, help="Path to .h5mu or .zarr file")
    inspect_parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")

    validate_parser = subparsers.add_parser(
        "validate", help="Validate a YAML config without training",
    )
    validate_parser.add_argument("--config", required=True, help="Path to YAML config file")
    validate_parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")

    args = parser.parse_args()
    _setup_logging(args.debug)

    if args.command == "train":
        _train(args)
    elif args.command == "transform":
        _transform(args)
    elif args.command == "cross-validate":
        _cross_validate(args)
    elif args.command == "optimize":
        _optimize(args)
    elif args.command == "plot":
        _plot(args)
    elif args.command == "convert":
        _convert(args)
    elif args.command == "inspect":
        _inspect(args)
    elif args.command == "validate":
        _validate(args)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)


def _setup_logging(debug: bool):
    """Configure logging: debug enables detailed logs, suppresses noisy third-party loggers."""
    warnings.filterwarnings("ignore", category=FutureWarning, module="mudata")
    warnings.filterwarnings("ignore", message="Cannot join columns with the same name", module="mudata")
    warnings.filterwarnings("ignore", message=".*LeafSpec.*is deprecated", module="pytorch_lightning")
    warnings.filterwarnings("ignore", message=".*batch_size.*ambiguous collection", module="pytorch_lightning")
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


def _build_model(data_cfg, model_cfg):
    """Instantiate the model class matching the parsed model_cfg type."""
    from mosa.models.mofa.config import MOFAConfig
    from mosa.models.mosa.config import MOSAVAEConfig

    if isinstance(model_cfg, MOSAVAEConfig):
        from mosa.models.mosa import MOSAVAEModel
        return MOSAVAEModel(data_cfg, model_cfg)
    if isinstance(model_cfg, MOFAConfig):
        from mosa.models.mofa import MOFAModel
        return MOFAModel(data_cfg, model_cfg)
    raise TypeError(f"Unsupported model_cfg type: {type(model_cfg).__name__}")


def _train(args):
    """Load config and data, split into train/val, and fit the model."""
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.model_selection import train_test_split

    from mosa.data.io import load_mudata
    from mosa.utils import load_config, seed_everything

    torch.set_float32_matmul_precision("high")
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

    if int(os.environ.get("LOCAL_RANK", 0)) != 0:
        logging.getLogger("mosa").setLevel(logging.WARNING)

    cfg = load_config(args.config)
    cfg.data.validate_paths()
    logger.debug("Config loaded from %s", args.config)

    seed_everything(cfg.model.random_seed)

    dataset = load_mudata(
        cfg.data.path,
        cfg.data.views,
        cfg.data.mask_layer_name,
    )

    train_data = dataset
    val_data = None
    if cfg.model.test_size > 0:
        label_codes = pd.Categorical(
            dataset.metadata["model_type"],
            categories=sorted(dataset.metadata["model_type"].unique()),
            ordered=True,
        ).codes

        train_idx, val_idx = train_test_split(
            np.arange(dataset.n_samples),
            test_size=cfg.model.test_size,
            random_state=cfg.model.random_seed,
            stratify=label_codes,
        )
        train_data = dataset.subset(train_idx)
        val_data = dataset.subset(val_idx)

    model = _build_model(cfg.data, cfg.model)
    if "resume_from" in model.fit.__code__.co_varnames:
        model.fit(train_data, val_data, resume_from=args.resume)
    else:
        model.fit(train_data, val_data)

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        logger.debug("Training complete")


def _transform(args):
    """Load a saved model and project data into the latent space."""
    import pandas as pd

    from mosa.data.io import load_mudata
    from mosa.models.mosa import MOSAVAEModel

    model = MOSAVAEModel.load(args.checkpoint)
    dataset = load_mudata(
        args.input,
        model.data_cfg.views,
        model.data_cfg.mask_layer_name,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    z = model.transform(dataset)
    pd.DataFrame(z, index=dataset.sample_names).to_parquet(out_dir / "latent.parquet")
    print(f"Latent representations saved to {out_dir / 'latent.parquet'}")

    if args.reconstruct:
        recon = model.reconstruct(dataset)
        for omic, arr in recon.items():
            df = pd.DataFrame(arr, index=dataset.sample_names)
            df.to_parquet(out_dir / f"recon_{omic}.parquet")
        print(f"Reconstructions saved to {out_dir}/")


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

    logger.debug("Converting CSV dataset to MuData format")
    logger.debug("Output file: %s", args.output)

    csv_to_mudata(
        samplesheet_path=args.samplesheet,
        view_specs=view_specs,
        output_path=args.output,
        mutations_path=args.mutations,
        format=args.format,
    )
    print(f"MuData file saved to {args.output}")


def _inspect(args):
    """Print a summary of a MuData file."""
    from mosa.data.io import inspect_mudata

    inspect_mudata(args.input)


def _validate(args):
    """Validate a YAML config without training."""
    from mosa.utils import load_config

    try:
        cfg = load_config(args.config)
        cfg.data.validate_paths()
    except (ValueError, FileNotFoundError, KeyError, TypeError) as e:
        print(f"Config invalid: {e}")
        sys.exit(1)

    print("Config OK")
    print(f"  data:    {cfg.data.path}")
    print(f"  views:   {cfg.data.views} (discrete: {sorted(cfg.data.discrete_views)})")
    print(f"  model:   {type(cfg.model).__name__}")
    print(f"  output:  {cfg.model.output_dir}")
    print(f"  seed:    {cfg.model.random_seed}, test_size={cfg.model.test_size}")

    from mosa.models.mosa.config import MOSAVAEConfig
    if isinstance(cfg.model, MOSAVAEConfig):
        print(f"  arch:    fusion={cfg.model.fusion_method}, latent={cfg.model.joint_latent_dim}")
        print(f"  train:   epochs={cfg.model.num_epochs}, batch_size={cfg.model.batch_size}")


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
        "--samplesheet", required=True,
        help="Path to samplesheet CSV (required columns: model_id, model_type; optional: tissue)",
    )
    convert_parser.add_argument(
        "--view", required=True, action="append",
        help="View spec as 'name:path' (e.g. 'gexp_voom:data/gexp_voom.csv'). Repeat for each modality.",
    )
    convert_parser.add_argument(
        "--mutations", default=None,
        help="Path to mutations CSV (features x samples, binary). Columns become mutation_* in .obs.",
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

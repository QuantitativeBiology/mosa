from __future__ import annotations

import argparse
import logging
import os
import warnings

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
        logging.basicConfig(level=logging.WARNING)
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


def _train(args):
    """Load config and data, split into train/val, and fit the model."""
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.model_selection import train_test_split

    from mosa.data.io import load_mudata
    from mosa.model.mosa_model import MOSAVAEModel
    from mosa.utils import load_config, seed_everything

    torch.set_float32_matmul_precision("high")
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

    if int(os.environ.get("LOCAL_RANK", 0)) != 0:
        logging.getLogger("mosa").setLevel(logging.WARNING)

    config = load_config(args.config)
    config.validate_paths()
    logger.debug("Config loaded from %s", args.config)

    seed_everything(config.random_seed)

    dataset = load_mudata(
        config.data_path,
        list(config.views.keys()),
        config.mask_layer_name,
    )

    train_data = dataset
    val_data = None
    if config.test_size > 0:
        label_codes = pd.Categorical(
            dataset.metadata["model_type"],
            categories=sorted(dataset.metadata["model_type"].unique()),
            ordered=True,
        ).codes

        train_idx, val_idx = train_test_split(
            np.arange(dataset.n_samples),
            test_size=config.test_size,
            random_state=config.random_seed,
            stratify=label_codes,
        )
        train_data = dataset.subset(train_idx)
        val_data = dataset.subset(val_idx)

    model = MOSAVAEModel(config)
    model.fit(train_data, val_data)

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        logger.debug("Training complete")


def _plot(args):
    """Generate diagnostic plots from a completed training run."""
    from mosa.plot import generate_all_plots
    from mosa.utils import load_config

    config = load_config(args.config)
    output_dir = args.output_dir or config.output_dir
    logger.debug("Generating plots from %s", output_dir)

    plots_dir = generate_all_plots(output_dir, config)
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


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MOSA: Multi-Omic Synthetic Augmentation",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a MOSA model")
    train_parser.add_argument("--config", required=True, help="Path to YAML config file")
    train_parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")

    plot_parser = subparsers.add_parser("plot", help="Generate diagnostic plots from training outputs")
    plot_parser.add_argument("--config", required=True, help="Path to YAML config file")
    plot_parser.add_argument(
        "--output-dir",
        default=None,
        help="Path to training output directory (defaults to output_dir in config)",
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

    args = parser.parse_args()
    _setup_logging(args.debug)

    if args.command == "train":
        _train(args)
    elif args.command == "plot":
        _plot(args)
    elif args.command == "convert":
        _convert(args)
    elif args.command == "inspect":
        _inspect(args)


if __name__ == "__main__":
    main()

"""Command line: `python -m benchmark_forecasting sync|bounds`."""

import argparse
import logging
from pathlib import Path

from benchmark_forecasting import config
from benchmark_forecasting.data import asymptote_bounds, load_dataset
from benchmark_forecasting.sync import sync


def main() -> None:
    parser = argparse.ArgumentParser(prog="benchmark_forecasting")
    sub = parser.add_subparsers(dest="command", required=True)
    p_sync = sub.add_parser("sync", help="copy the pipeline's views into 0_input/")
    p_sync.add_argument(
        "--pipeline",
        type=Path,
        default=config.PIPELINE_DIR,
        help="benchmark-data-pipeline checkout (default: sibling directory)",
    )
    sub.add_parser("bounds", help="print the asymptote floor and pinned value per benchmark")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.command == "sync":
        provenance = sync(args.pipeline)
        print(
            f"0_input/ now mirrors pipeline commit {provenance['pipeline_commit'][:8]} "
            f"built {provenance['pipeline_built_at']}"
        )
    elif args.command == "bounds":
        bounds = asymptote_bounds(load_dataset(), config.ModelConfig())
        print(bounds.sort_values(["reason", "L_floor"]).to_string())


if __name__ == "__main__":
    main()

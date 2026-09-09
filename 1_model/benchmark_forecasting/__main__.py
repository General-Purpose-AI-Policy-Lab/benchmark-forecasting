"""Command line: `python -m benchmark_forecasting sync|bounds`."""

import argparse
import logging
from pathlib import Path

from benchmark_forecasting import config
from benchmark_forecasting.data import asymptote_bounds, load_dataset, prepare_dataset
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
    p_bounds = sub.add_parser("bounds",
                              help="print the asymptote floor and pinned value per benchmark")
    p_bounds.add_argument(
        "--cutoff",
        default=config.DATA_CUTOFF,
        help="data cutoff (inclusive) applied before the frontier, as the scripts do "
             "(default: config.DATA_CUTOFF; 'none' for every score)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.command == "sync":
        provenance = sync(args.pipeline)
        print(
            f"0_input/ now mirrors pipeline commit {provenance['pipeline_commit'][:8]} "
            f"built {provenance['pipeline_built_at']}"
        )
    elif args.command == "bounds":
        import pandas as pd
        raw = load_dataset()
        if args.cutoff.lower() != "none":
            raw = raw[raw["release_date"] <= pd.Timestamp(args.cutoff)]
        cfg = config.ModelConfig()
        bounds = asymptote_bounds(prepare_dataset(raw, top_n=cfg.top_n), cfg)
        print(bounds.sort_values(["reason", "L_floor"]).to_string())


if __name__ == "__main__":
    main()

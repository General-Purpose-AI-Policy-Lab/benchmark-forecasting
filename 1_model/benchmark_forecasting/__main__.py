"""Command line: `python -m benchmark_forecasting sync|bounds|prune-fits|thin-fits`."""

import argparse
import logging
from pathlib import Path

from benchmark_forecasting import config
from benchmark_forecasting.data import asymptote_bounds, load_dataset, prepare_dataset
from benchmark_forecasting.sync import sync


def _fits_dirs(cutoff: str | None) -> list[Path]:
    """The cache folder of one cutoff tag, or of every run folder."""
    if cutoff:
        return [config.cutoff_dir(cutoff) / config.FITS_SUBDIR]
    return sorted(config.OUTPUTS_DIR.glob(f"*/{config.FITS_SUBDIR}"))


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
    p_prune = sub.add_parser("prune-fits",
                             help="delete the caches of older data (all but each fit's last used)")
    p_prune.add_argument("--cutoff", default=None,
                         help="cutoff folder tag, e.g. cutoff20261001 (default: every cutoff)")
    p_prune.add_argument("--dry-run", action="store_true", help="list without deleting")
    p_thin = sub.add_parser("thin-fits",
                            help="keep one draw in four in the caches sampled before thinning")
    p_thin.add_argument("--cutoff", default=None,
                        help="cutoff folder tag, e.g. cutoff20261001 (default: every cutoff)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.command == "sync":
        provenance = sync(args.pipeline)
        print(
            f"0_input/ now mirrors pipeline commit {provenance['pipeline_commit'][:8]} "
            f"run {provenance['pipeline_run_date']} (built {provenance['pipeline_built_at']}); "
            f"set config.DATA_CUTOFF to that day"
        )
    elif args.command == "bounds":
        import pandas as pd
        raw = load_dataset()
        if args.cutoff.lower() != "none":
            raw = raw[raw["release_date"] <= pd.Timestamp(args.cutoff)]
        cfg = config.ModelConfig()
        bounds = asymptote_bounds(prepare_dataset(raw, top_n=cfg.top_n), cfg)
        print(bounds.sort_values(["reason", "L_floor"]).to_string())
    elif args.command == "thin-fits":
        from benchmark_forecasting.fit import SAVE_THIN, thin_cached_fits
        for fits_dir in _fits_dirs(args.cutoff):
            done = thin_cached_fits(fits_dir)
            print(f"{fits_dir}: {len(done)} cache(s) thinned to one draw in {SAVE_THIN}")
    elif args.command == "prune-fits":
        from benchmark_forecasting.fit import prune_stale_fits
        for fits_dir in _fits_dirs(args.cutoff):
            stale = prune_stale_fits(fits_dir, dry_run=args.dry_run)
            size = sum(p.stat().st_size for p in stale) if args.dry_run else None
            verb = "would delete" if args.dry_run else "deleted"
            extra = f" ({size / 1e9:.1f} GB)" if size is not None else ""
            print(f"{fits_dir}: {verb} {len(stale)} stale fit(s){extra}")
            for path in stale:
                print(f"  {path.name}")


if __name__ == "__main__":
    main()

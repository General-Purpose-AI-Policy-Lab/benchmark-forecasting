"""`sync` copies the pipeline's latest dated run; the main runs' cutoff must be that run's date."""

import json

import pytest

from benchmark_forecasting import config
from benchmark_forecasting.sync import SOURCES, latest_run, sync


def _pipeline(root, dates):
    for date in dates:
        for folder, name in SOURCES:
            d = root / folder / date
            d.mkdir(parents=True, exist_ok=True)
            manifest = {"git_commit": f"c{date}", "built_at": f"{date}T0900+0200"}
            body = json.dumps(manifest) if name.endswith(".json") else f"run,{date}\n"
            (d / name).write_text(body)
    (root / "2_database" / "notes").mkdir()  # not a run
    return root


def test_sync_takes_the_latest_run_and_records_its_date(tmp_path):
    pipe = _pipeline(tmp_path / "pipe", ["20260908", "20261001", "20260930"])
    assert latest_run(pipe) == "20261001"
    prov = sync(pipe, tmp_path / "in")
    assert prov["pipeline_run_date"] == "20261001" and prov["pipeline_commit"] == "c20261001"
    assert (tmp_path / "in" / config.SCORES_FILE).read_text() == "run,20261001\n"


def test_cutoff_must_be_the_synced_run_date(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INPUT_DIR", tmp_path)
    (tmp_path / config.PROVENANCE_FILE).write_text(json.dumps({"pipeline_run_date": "20261001"}))
    assert config.checked_data_cutoff("2026-10-01") == "2026-10-01"
    with pytest.raises(ValueError, match='DATA_CUTOFF = "2026-10-01"'):
        config.checked_data_cutoff("2026-09-29")

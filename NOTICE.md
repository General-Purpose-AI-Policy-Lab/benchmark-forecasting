# NOTICE: third-party data

The `LICENSE` (CC-BY-4.0) covers what this project produces: the code, the analysis scripts, the fitted models and the figures and tables under `3_outputs/`.

It does **not** cover the benchmark scores and human baselines those outputs are built from. `0_input/dated_scores_flat.csv` and `0_input/human_baselines.csv` are copies of the views of [`benchmark-data-pipeline`](https://github.com/General-Purpose-AI-Policy-Lab/benchmark-data-pipeline) (run and commit in `0_input/provenance.json`), which aggregates published results from the sources below. Each row keeps its origin in the `feed`, `source` and `source_url` columns. Anyone republishing these files, or an analysis built on them, must credit the sources and respect their terms.

## Sources of the scores

| Feed (`feed` column) | Source | Terms |
|---|---|---|
| `epoch`, `epoch_live` | Epoch AI Benchmarking Hub: [`epoch.ai/data/benchmark_data.zip`](https://epoch.ai/data/benchmark_data.zip) and [`epoch.ai/data/benchmarks.csv`](https://epoch.ai/data/benchmarks.csv) | **CC-BY 4.0** (Epoch AI), attribution required ([terms](https://epoch.ai/benchmarks/use-this-data)). Rows whose `source` is not Epoch's own evaluation are results Epoch compiles from papers and leaderboards; per Epoch, they keep their original licensing, and the original source is given in `source` / `source_url`. |
| `epoch_cyber` | Epoch AI's compilation of cybersecurity results, [`epoch.ai/data/cyber/processed_data_for_eci.csv`](https://epoch.ai/data/cyber/processed_data_for_eci.csv), described in Chauvin et al., ["Are Mythos' cyber capabilities overhyped?"](https://epoch.ai/gradient-updates/are-mythos-cyber-capabilities-overhyped) (Epoch AI, 2026) | **CC-BY 4.0** (Epoch AI), attribution required. Some values were read off plots published by the UK AI Security Institute and others; cite the underlying benchmarks as well. |
| `kaggle` | Kaggle Open Benchmarks leaderboards: [`open-benchmarks/mmlu`](https://www.kaggle.com/benchmarks/open-benchmarks/mmlu), `open-benchmarks/mmlu-pro`, `deepmind/simpleqa-verified` | Released under the **Apache 2.0** license (stated on each board); cite the board and the benchmark paper. |
| `rand` | Digitized from RAND report **RR-A3797-1**: Dev et al., *Toward Comprehensive Benchmarking of the Biological Knowledge of Frontier Large Language Models*, RAND Corporation, 2025, [doi:10.7249/RRA3797-1](https://www.rand.org/pubs/research_reports/RRA3797-1.html) | Copyright RAND Corporation. Only the numeric scores are reproduced, for non-commercial research and with citation; neither the report nor its figures are redistributed. Commercial reuse of these rows requires RAND's permission. |
| `seal` | Scale AI SEAL leaderboards, [`labs.scale.com/leaderboard`](https://labs.scale.com/leaderboard) | **No open license**; Scale AI's [terms](https://scale.com/legal/terms) apply. See below. |
| `cybench_site` | [Cybench](https://cybench.github.io) leaderboard (Zhang et al., ICLR 2025), frozen snapshot | No license stated; cite the Cybench paper. |

## Scale SEAL

Scale AI publishes the SEAL leaderboards without an open license. This repository does not redistribute the leaderboard pages, their layout or any other content from them. It keeps only the numeric scores (one score per model and benchmark, with the model name and release date), which are factual results, reproduced for non-commercial research with attribution to Scale AI: every such row carries `feed = seal` and the URL of its board in `source_url`. Anyone redistributing these rows should attribute Scale AI and check its current terms first.

## Human baselines

`0_input/human_baselines.csv` lists human performance figures taken from published papers, reports and leaderboards (one figure per row, with its source in the `source` column). They are cited in the papers that use them; credit the original source when reusing a figure.

## Attribution when citing

> Benchmark scores aggregated from Epoch AI (CC-BY 4.0), Kaggle Open Benchmarks (Apache 2.0), RAND RR-A3797-1 (Dev et al. 2025), Scale AI SEAL and the Cybench leaderboard, via the General-Purpose AI Policy Lab `benchmark-data-pipeline`; forecasts by General-Purpose AI Policy Lab, `benchmark-forecasting` (CC-BY 4.0).

# AGENTS.md

## Goal
Q3 of the Advanced Programming 2026 brief, options A and B: forecast the PMI contraction regime P(d_{t+1}=1) using only F_t = {x_{i,s}: s + L_i <= t}. Option A is a classical probit/PCA model; Option B is an LSTM with leakage ablations.

## Environment
Python 3.13, managed with uv, dependencies pinned.
uv sync --python 3.13 && source .venv/bin/activate
./reproduce.sh --quick   # tests + compare only
./reproduce.sh           # offline data check, tests, Q3 approaches, compare
python code/check_reproduction.py

## Approach map
- approach1_q3a_classical.py: Q3A classical probit (formerly approach5)
- approach1_q3a_posthoc.py: Q3A post-hoc probit robustness
- approach2_q3b_sequence.py: Q3B LSTM and leakage ablation (formerly approach6)
- approach2_q3b_posthoc.py: Q3B post-hoc diagnostics and retraining

## Data
data/clean/macro_panel.csv, data/clean/macro_meta.csv and data/clean/shibor.csv are committed snapshots. data/metadata.json is filtered to Q3 artifacts; data/fetch.py --offline verifies every checksum. Raw Q3 interface responses were gitignored caches in the parent and are not part of this repository.

## Conventions
Seed: --seed 42 by default, CLI-overridable; every stochastic step is seeded.
Each code/approach*.py: argparse CLI, load -> verify -> split -> fit -> evaluate -> write.
Output: results/approachN/<dataset>/metrics.json = {params, metrics, fit_seconds, history}.
Terminal output via rich; figures via plotly HTML with include_plotlyjs=True.
Q3B imports Q3A code and recomputes the probit; it does not read Q3A result files.

## Reproduction
`./reproduce.sh` is the full offline path; `./reproduce.sh --quick` runs tests and compare only.
`python code/check_reproduction.py` compares `.cache/clean_clone_full/results` against the committed `results`, ignores timing fields, and uses relative tolerance 1e-6.

## Provenance
This repository was split from the combined AP 2026 repository at parent tag `submission`, commit `4521da3`. See `SPLIT_HISTORY.md` for the old-to-new commit map, renamed paths and the split disclosure.

## Agent rules
Write nothing outside the repo root; tool caches stay in the repo and are gitignored.
Propose first and wait for approval before any destructive operation.
Commit after each milestone (data:, feat:, exp:, refactor:, docs:).
Never commit credentials. Never invent results: every reported number comes from a metrics.json.

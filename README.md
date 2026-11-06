# AP 2026 Q3: PMI regime forecasting

Q3 of the Advanced Programming 2026 coursework: forecast the PMI contraction regime P(d_{t+1}=1), with a classical probit/PCA model (Q3A) and an LSTM with leakage ablations (Q3B).

## Run

`uv sync --python 3.13 && source .venv/bin/activate`

Full reproduction: `./reproduce.sh`
Quick checks (tests + comparison): `./reproduce.sh --quick`
Clean-clone comparison: `python code/check_reproduction.py`

## Results

- `results/approach1/macro/` holds the Q3A probit, nowcast and clustering outputs, including the post-hoc robustness block.
- `results/approach2/macro/` holds the Q3B LSTM, leakage ablation and post-hoc diagnostics/retraining outputs.

## Provenance

Split from the combined AP 2026 repository at parent tag `submission`, commit `4521da3`. Former `approach5` is now `approach1`; former `approach6` is now `approach2`. See `SPLIT_HISTORY.md` for the commit map and disclosure. `SESSION.md` is the full transcript for all questions and uses the old file names and hashes.

# Advanced Programming 2026 Project Report — Q3

Hao Lu

## Provenance and split disclosure

The work was done in one repository (parent tag `submission`, commit `4521da3`) and split afterwards into one repository per question following the instructor's guidance. Commit hashes changed and the old-to-new commit map is included in `SPLIT_HISTORY.md`. Dates in rewritten commits are the original ones; the split commits are dated when they were made. Shared-file commits appear in more than one repository. `SESSION.md` is the full transcript for all questions and refers to old file names and hashes. Agent work on the split ran with `ap2026-split` as its root.

Section and table numbers below follow the parent report; only this question's sections are included.

### Old-to-new mapping

| Parent | This repository |
|---|---|
| `code/approach5_q3a_classical.py` | `code/approach1_q3a_classical.py` |
| `code/approach5_q3a_posthoc.py` | `code/approach1_q3a_posthoc.py` |
| `code/approach6_q3b_sequence.py` | `code/approach2_q3b_sequence.py` |
| `code/approach6_q3b_posthoc.py` | `code/approach2_q3b_posthoc.py` |
| `results/approach5/` | `results/approach1/` |
| `results/approach6/` | `results/approach2/` |

Some code comments, labels and JSON keys keep the old names, such as approach3 or approach5_members. They were left unchanged so that results stay directly comparable with the parent.

## 1. Summary

The repository implements Q3 of the Advanced Programming 2026 brief, options A and B. Q3 forecasts the PMI contraction regime with a classical probit/PCA model (Q3A) and an LSTM with leakage ablations (Q3B).

Q3A beats the expanding prior on Brier score but not persistence (AUC 0.6928 vs 0.7390), and its nowcast beats a random-walk-with-drift benchmark in all periods, including ex-COVID. Q3B does not beat persistence either; the retrained LSTM never predicts a probability above 0.5 on the test block and its test AUC is 0.6422 versus 0.7390 for persistence.

What was not achieved: Q3A and Q3B do not reach the indicative AUC target of 0.75 or beat persistence. The data are monthly and use the current revised vintage, not real-time vintages.

## 2. Environment and one-command reproduction

The project uses Python 3.13 under `uv` with pinned dependencies. From the repository root:

```bash
uv sync --python 3.13
source .venv/bin/activate
./reproduce.sh --quick   # tests + compare only
./reproduce.sh           # offline data check, tests, Q3 approaches, compare
python code/check_reproduction.py
```

`./reproduce.sh` runs `data/fetch.py --offline`, the Q3 tests, Q3A classical and post-hoc, Q3B sequence and post-hoc, and `compare.py`. The offline check verifies the three committed clean artifacts against `data/metadata.json` and reports that all checksums match. The test suite ran 13 tests and passed.

The clean-clone check compares `.cache/clean_clone_full/results` with the committed results. The full clean-clone run completed in 209 s with HTTP_PROXY and HTTPS_PROXY pointed at `http://127.0.0.1:9`. The checker reports 10,180 shared numeric values, all exact, with maximum absolute and relative differences 0. It also reports 216 numeric keys only in the reproduced run (`history.refits[*].min_epochs` config echoes), the same known key-set difference documented in the parent; there are zero numeric differences among shared keys.

The sum of committed `fit_seconds` is 52.5 s, about 0.9 minutes; the clean-clone wall time was 3.5 minutes including the post-hoc retraining.

## 3. Data

### 3.1 Sources, coverage and snapshots

Table 1: Committed clean artifacts in this repository. Source: `data/metadata.json`.

| artifact | source | rows | retrieved |
|---|---|---|---|
| `data/clean/shibor.csv` | China Shibor | 2398 | 2026-11-04 |
| `data/clean/macro_panel.csv` | Q3 monthly macro panel from akshare macro_china interfaces and committed SHIBOR | 202 | 2026-11-05 |
| `data/clean/macro_meta.csv` | Q3 series metadata and conservative availability lags | 34 | 2026-11-05 |

The raw Q3 interface responses were local, gitignored caches in the parent repository; the cleaned panel and metadata are committed here. `data/clean/shibor.csv` is the committed SHIBOR source used to build `shibor_3m`. `data/metadata.json` contains only the Q3 artifacts, and `data/fetch.py --offline` verifies every checksum.

### 3.3 Q3 panel, lags and revisions

The macro panel has 202 month-end observations and 34 series. Publication lags follow `F_t = {x_{i,s}: s + L_i <= t}`. Thirteen series have `L_i = 0` (market rates, yields, PMI and market-based series); twenty-one have `L_i = 1` (CPI, PPI, money, credit, trade, activity, GDP and similar). Most lags are conservative assumptions because the raw interfaces carry no genuine release date; lag zero is definitional only for market rates and yields. Jin10 announcement-date interfaces were saved but mostly stop in 2025 and are not used for the panel, except `macro_china_gdp_yearly` as a cross-check. GDP is derived from cumulative real levels and the current revised vintage: 79 quarters derived, 59 validation quarters, maximum absolute difference 0.83 percentage points and mean absolute difference 0.17 points. The data are the current revised vintage, not real-time vintages.

January and February are missing for: `retail_yoy` (Jan/Feb missing share 0.8824), `industrial_production_yoy` (Jan 1.000, Feb 0.7059), `fai_ytd_yoy` (Jan 1.000, Feb 0.1765), `electricity_yoy` (Jan 0.9412, Feb 0.000) and `industrial_profit_yoy` (Jan 1.000, Feb 0.000). These values are left missing rather than imputed.

Largest missing shares are `gdp_yoy` 0.6683, `lpr_5y` 0.5693, `unemployment` 0.4802, `tsf_12m_yoy` 0.4406 and `shibor_3m` 0.4257. At threshold 50, the full-sample PMI-below share is 0.3069 and the two-month regime share is 0.2079 (42 months); at 49.5 the regime share is 0.0743 (15 months).

## 8. Q3 Option A: classical probit

### 8.1 Model and protocol

The information set is implemented by shifting each series forward by its lag `L_i`: row `t` contains only values public at the end of month `t`. The label is `d_t = 1{PMI_t < 50 and PMI_{t-1} < 50}`. The model is `P(d_{t+1}=1) = Phi(beta' z_t)`, where `z_t` contains an intercept, the first `r` principal components of the standardized aligned factor panel (excluding `gdp_yoy`, `cgb10y`, `spread_10y_2y`, `shibor_3m`) and the controls `cgb10y` and `spread_10y_2y`. PCA and standardization use only rows public at the origin; missing values become zero after standardization; series with under 50% training-window coverage are dropped. The probit is fitted by maximum likelihood with a ridge of `1e-4` and a Newton/BFGS fallback. Validation is expanding walk-forward, refit monthly, minimum 60 training months, a 36-month tuning block and a test block of 105 origins from 2018-01-31 to 2026-09-30. The PCA rank is chosen on the tuning block by cross-validated nowcast RMSE at month 3 of the quarter.

The nowcast is `y_q = b0 + b' fbar_q`, where `fbar_q` averages the aligned factors over the months of the quarter elapsed by `t`; training quarters are those with GDP public by `t`. The benchmark is a random walk with drift estimated from the same information set. Clustering uses rolling 36-month correlations, `d_ij = sqrt(2(1-rho_ij))`, Ward linkage, `k` chosen by silhouette, adjacent-window adjusted Rand indices, and lead-lag correlations with GDP.

### 8.2 Results

Table 25: Q3A test-block classification (105 origins, 34 positives, class share 0.3238). Source: `results/approach1/macro/metrics.json`.

| model | Brier | AUC | accuracy |
|---|---|---|---|
| probit | 0.2396 | 0.6928 | 0.6381 |
| prior | 0.2482 | 0.604 | 0.6762 |
| persistence | 0.2286 | 0.739 | 0.7714 |

The chosen PCA rank is 6, with tuning CV RMSE 0.3565 over 12 quarterly observations; rank 5 is 0.3838, rank 4 is 0.4313, rank 3 is 0.6175, rank 2 is 0.8006, rank 1 is 0.6774, rank 7 is 0.4564 and rank 8 is 0.5271. The first principal component explains 0.3098 of variance. No slope coefficient is significant at 5% with Newey-West standard errors; the largest are PC1 (`z=-1.892`) and `spread_10y_2y` (`z=-1.678`). The only strongly significant term is the intercept (`z=-8.180`).

Table 26: Q3A label sensitivity, including the prior. Source: `results/approach1/macro/metrics.json`.

| specification | probit Brier | probit AUC | probit acc | prior Brier | prior AUC | prior acc | persistence Brier | persistence AUC | persistence acc |
|---|---|---|---|---|---|---|---|---|---|
| threshold 50, two-month (main) | 0.2396 | 0.6928 | 0.6381 | 0.2482 | 0.604 | 0.6762 | 0.2286 | 0.739 | 0.7714 |
| threshold 49.5, two-month | 0.1715 | 0.7159 | 0.7714 | 0.1235 | 0.6052 | 0.8667 | 0.1714 | 0.6291 | 0.8286 |
| threshold 50, one-month | 0.2468 | 0.6625 | 0.6571 | 0.3033 | 0.6441 | 0.5333 | 0.2762 | 0.7219 | 0.7238 |

The probit loses to the prior on accuracy at the main threshold (0.6381 vs 0.6762) and at 49.5 (0.7714 vs 0.8667), and on Brier at 49.5 (0.1715 vs 0.1235). It wins on Brier and AUC at the main threshold against the prior, and on all three one-month metrics.

Table 27: Q3A SHIBOR sensitivity. Source: `results/approach1/macro/metrics.json`.

| model | Brier | AUC | accuracy |
|---|---|---|---|
| probit | 0.2408 | 0.6912 | 0.6381 |
| prior | 0.2482 | 0.604 | 0.6762 |
| persistence | 0.2286 | 0.739 | 0.7714 |

Table 28: Q3A nowcast RMSE. Source: `results/approach1/macro/metrics.json`.

| sample | n | month 1 | month 2 | month 3 | RW drift |
|---|---|---|---|---|---|
| all quarters | 35 | 3.626 | 3.411 | 3.314 | 4.163 |
| ex-COVID | 27 | 0.7095 | 0.8065 | 0.8784 | 1.258 |

The bridge beats the random walk in all periods, including ex-COVID: 0.71 to 0.88 versus 1.26. Ex-COVID the month-3 RMSE is higher than month-1, but the paired difference 0.1689 has interval `[-0.1071, 0.4390]`, so the degradation is not statistically resolved. The advantage over the random walk is not concentrated in COVID.

Table 29: Q3A full-sample descriptive probit coefficients. Source: `results/approach1/macro/metrics.json`.

| term | estimate | information SE | information z | Newey-West SE | Newey-West z |
|---|---|---|---|---|---|
| intercept | -1.547 | 0.2216 | -6.98 | 0.1891 | -8.18 |
| PC1 | -0.3636 | 0.1897 | -1.917 | 0.1922 | -1.892 |
| PC2 | 0.3264 | 0.1761 | 1.853 | 0.2125 | 1.536 |
| PC3 | 0.1725 | 0.1178 | 1.464 | 0.1334 | 1.293 |
| PC4 | -0.33 | 0.16 | -2.063 | 0.2189 | -1.507 |
| PC5 | -0.2055 | 0.137 | -1.5 | 0.1325 | -1.551 |
| PC6 | -0.1271 | 0.1263 | -1.006 | 0.1349 | -0.9424 |
| cgb10y | -0.5927 | 0.597 | -0.9929 | 0.7123 | -0.8321 |
| spread_10y_2y | -0.3375 | 0.2208 | -1.529 | 0.2011 | -1.678 |

Table 30: Q3A latest cluster membership (2023-01-31 to 2025-12-31, k=9). Source: `results/approach1/macro/metrics.json`.

| cluster | members |
|---|---|
| 1 | cgb10y, shibor_3m, lpr_1y, lpr_5y, cgb2y, cgb5y, cgb30y |
| 2 | pmi_non_man, m2_yoy, retail_yoy, real_estate_climate, fiscal_revenue_yoy |
| 3 | pmi, consumer_confidence, unemployment |
| 4 | new_credit_yoy |
| 5 | m1_yoy, tsf_12m_yoy, stock_mktcap_yoy, margin_balance_yoy |
| 6 | ppi_yoy, commodity_price_yoy, commodity_spot_yoy, industrial_profit_yoy |
| 7 | industrial_production_yoy, fai_ytd_yoy, exports_yoy, imports_yoy |
| 8 | fx_reserves_yoy, central_bank_assets_yoy |
| 9 | spread_10y_2y, cpi_yoy, electricity_yoy |

Table 31: Q3A adjacent-window ARI. Source: `results/approach1/macro/metrics.json`.

| left window | right window | common series | ARI |
|---|---|---|---|
| 2012-12-31 | 2013-12-31 | 26 | 0.09801 |
| 2013-12-31 | 2014-12-31 | 26 | 0.2098 |
| 2014-12-31 | 2015-12-31 | 28 | 0.08733 |
| 2015-12-31 | 2016-12-31 | 29 | 0.3273 |
| 2016-12-31 | 2017-12-31 | 29 | 0.3217 |
| 2017-12-31 | 2018-12-31 | 29 | 0.2424 |
| 2018-12-31 | 2019-12-31 | 31 | 0.0965 |
| 2019-12-31 | 2020-12-31 | 32 | 0.3533 |
| 2020-12-31 | 2021-12-31 | 32 | 0.09256 |
| 2021-12-31 | 2022-12-31 | 33 | 0.1148 |
| 2022-12-31 | 2023-12-31 | 33 | 0.5456 |
| 2023-12-31 | 2024-12-31 | 33 | 0.03484 |
| 2024-12-31 | 2025-12-31 | 33 | 0.1648 |

Table 32: Q3A lead-lag summary with GDP, latest window. Source: `results/approach1/macro/metrics.json`.

The lag sign is defined as `corr(cluster_t, GDP_{t+lag})`; a positive lag means the cluster leads GDP. Source: `results/approach1/macro/metrics.json`.

| cluster | largest absolute correlation | lag | n |
|---|---|---|---|
| 1 | 0.4131 | -1 | 67 |
| 2 | 0.7515 | 0 | 67 |
| 3 | 0.3671 | 3 | 64 |
| 4 | 0.2827 | 3 | 64 |
| 5 | 0.4092 | 0 | 67 |
| 6 | 0.4143 | -1 | 67 |
| 7 | 0.7771 | 0 | 67 |
| 8 | 0.4056 | 0 | 67 |
| 9 | 0.5491 | -3 | 65 |

Clustering has 14 rolling windows; the latest window selects `k=9` with silhouette 0.2675. Adjacent-window ARI has minimum 0.03484, median 0.1648 and maximum 0.5456, so cluster membership is unstable across windows.

Table 33: Q3A post-hoc probit robustness (test block already seen; not pre-registered). Source: `results/approach1/macro/metrics.json`.

| model | Brier | AUC | accuracy | share outside [0.01,0.99] |
|---|---|---|---|---|
| original_probit | 0.2396 | 0.6928 | 0.6381 | 0.2762 |
| ridge_selected_probit | 0.2183 | 0.6881 | 0.6381 | 0.181 |
| ridge_selected_probit_with_d_t | 0.2001 | 0.739 | 0.7048 | 0.181 |
| prior | 0.2482 | 0.604 | 0.6762 | 0 |
| persistence | 0.2286 | 0.739 | 0.7714 | 1 |

The post-hoc `d_t` extension improves Brier to 0.2001 and matches persistence's AUC exactly (0.7390), but its paired AUC difference is 0.0 with interval `[-0.1316, 0.0713]`, so there is no evidence that it beats persistence. The original probit's AUC difference versus persistence is -0.04619 with interval `[-0.2375, 0.03518]`.

Figures: `results/approach1/macro/regime.html`, `nowcast_revision.html`, `dendrogram.html` and `stability.html`.

### 8.3 Discussion and limitations

The probit improves on the prior in Brier and AUC at the main threshold, but not in accuracy, and no significance test against the prior was run; persistence is stronger because the two-month label is highly persistent and PMI itself is in the information set. The AUC is 0.6928, below the 0.75 indicative target. The rank is chosen on only 12 quarterly nowcast observations, so the rank choice is noisy. The test block is 105 months but only 35 quarters are available for the nowcast, and the AUC bootstrap intervals are wide. The nowcast beats the random walk in all periods, including ex-COVID; the month-3 ex-COVID RMSE being higher than month-1 is unexplained and is not attributed to a cause. Clustering is descriptive and unstable, and it is not used for forecasting.

## 9. Q3 Option B: LSTM and leakage

### 9.1 Model and protocol

Q3B uses 12-month windows of the standardized aligned panel, excluding `gdp_yoy`, as input to an LSTM with at most two layers and a linear logit head. Training pairs are `(window ending s, d_{s+1})` with `s+1 <= origin t`. The loss is `BCEWithLogits` with `pos_weight = n_neg/n_pos`, and probabilities are corrected by `p_cal = sigmoid(logit - log(pos_weight))`. The walk-forward scheme is expanding, with the LSTM refit every six months while the Q3A probit refits monthly. Hyperparameters are selected on the 36-month tuning block only by mean Brier over three seeds.

The original run used 18-month validation blocks. The retrained model, which is the primary Q3B result, uses the chosen setting hidden 8, one layer, dropout 0.2 and weight decay `1e-2`, an expanding split with the last 36 training-pair months held out for validation, minimum 30 epochs, patience 30 and maximum 300 epochs, with three seeds. Leakage arms are the disciplined aligned panel, a zero-lag panel (`L_i=0`), a zero-lag panel with full-sample standardization, and a zero-lag probit. Leakage premiums use a circular 12-month block bootstrap with 5,000 replicates; positive means the leaked arm looks better (lower Brier or higher AUC).

### 9.2 Results

Table 34: Q3B primary comparison, retrained LSTM (test block, 105 origins, 34 positives). Source: `results/approach2/macro/metrics.json`.

| arm | Brier | AUC | accuracy |
|---|---|---|---|
| probit | 0.2396 | 0.6928 | 0.6381 |
| lstm_disciplined | 0.2561 | 0.6422 | 0.6762 |
| lstm_zero_lag | 0.2566 | 0.6315 | 0.6762 |
| lstm_zero_lag_full_sample | 0.2522 | 0.6138 | 0.6762 |
| prior | 0.2482 | 0.604 | 0.6762 |
| persistence | 0.2286 | 0.739 | 0.7714 |

The retrained disciplined LSTM has Brier 0.2561, AUC 0.6422 and accuracy 0.6762; it does not beat persistence (0.2286, 0.7390, 0.7714) or the probit (0.2396, 0.6928, 0.6381) on Brier or AUC. Its accuracy equals the all-negative rate because it never predicts a probability above 0.5 on the test block. In `results/approach2/macro/post_hoc_predictions.csv`, the disciplined arm has 0 of 105 predictions above 0.5 for each seed; the maxima are 0.1767 (seed 42), 0.3172 (seed 43) and 0.2234 (seed 44).

Table 35: Q3B retrained leakage premiums (positive means leaked arm better). Source: `results/approach2/macro/metrics.json`.

| comparison | Brier premium | Brier 95% CI | AUC premium | AUC 95% CI |
|---|---|---|---|---|
| LSTM zero lag vs disciplined | -0.0008503 | [-0.003906, 0.001532] | 0.004557 | [-0.06256, 0.0745] |
| LSTM full-sample standardization vs disciplined | 0.00368 | [0.0001998, 0.007341] | -0.009528 | [-0.07817, 0.06286] |
| LSTM standardization only | 0.00453 | [-0.0005247, 0.01053] | -0.01408 | [-0.09826, 0.06635] |
| probit zero lag vs probit | 0.006345 | [-0.01956, 0.03238] | 0.001864 | [-0.04367, 0.05129] |

After retraining, the Brier premium is -0.00085 for zero-lag alignment (CI includes zero), 0.00368 for full-sample standardization (CI excludes zero) and 0.00453 for standardization only (CI includes zero); the AUC premiums are 0.00456, -0.00953 and -0.01408, all with intervals including zero. The disciplined and leaked arms are therefore mostly indistinguishable. The larger original premiums are attributed to the original 18-month validation design, which was unreliable: 4 of 24 refits had a single-class validation block, and the tuning AUC was far below the test AUC. The original LSTM is shown as a failed run, not as the primary result.

Table 36: Q3B original failed run and tuning/test gap. Source: `results/approach2/macro/metrics.json`.

| arm | tuning AUC | test AUC | test Brier | test accuracy |
|---|---|---|---|---|
| original LSTM disciplined | 0.2677 | 0.4925 | 0.2604 | 0.6762 |
| retrained LSTM disciplined | 0.9146 | 0.6422 | 0.2561 | 0.6762 |

The original tuning AUC is the chosen setting's seed-mean from `history.tuning` (setting 3); the retrained tuning AUC is the chosen setting's seed-mean from `post_hoc.retrain.tuning_table` (setting 3). The original validation design produced 4 single-class blocks out of 24 refits; the retrained scheme uses 36 validation months and a minimum of 30 epochs.

Table 37: Q3B latent bridge and null. Source: `results/approach2/macro/metrics.json`.

| quantity | value |
|---|---|
| latest cluster window | 2023-01-31 to 2025-12-31 |
| k | 9 |
| indicators | 33 |
| ARI | 0.5219 |
| NMI | 0.7354 |
| null ARI mean / p95 | 0.001082 / 0.09255 |
| null ARI p-value | 0.000999 |
| null NMI mean / p95 | 0.4389 / 0.5093 |
| null NMI p-value | 0.000999 |

The bridge maps each indicator to a correlation vector with the 16 hidden dimensions over 2023-01-31 to 2025-12-31, clusters those vectors with `k=9`, and compares with Q3A's latest indicator labels. Because `h_t` is computed from those indicators, the ARI significance may be partly mechanical; the null permutes indicator labels against fixed bridge clusters, not the full construction. The latent analysis is descriptive.

Figures: `results/approach2/macro/probabilities.html` and `latent.html`.

### 9.3 Discussion and limitations

The original LSTM is a failed run: test AUC 0.4925 is essentially random, and its accuracy is the all-negative rate. The diagnostics show that the training loop can fit (shifted-label train AUC 0.9962; overfit test train AUC 0.9895), so the failure is not a broken loss. The retrained model improves test AUC to 0.6422 but still does not beat persistence or the probit on Brier score. The leakage premiums after retraining are small and mostly include zero, so the disciplined and leaked arms are mostly indistinguishable. Because NBS PMI has `L_i=0` in both arms and the label is built from PMI, the measured lag leakage premium is structurally limited. The final LSTM is trained through 2026-09, so it embeds test months in-sample; the latent analysis is descriptive only.

## 10. Indicative benchmarks

The brief file is held outside the repository. The target column below is quoted from the project brief, not from `AGENTS.md` or a results file. Every number in the reported-result column is from the metrics files named in the caption.

Table 38: Indicative Q3 benchmarks. Source: `results/approach1/macro/metrics.json`, `results/approach2/macro/metrics.json`; target column from the brief/instructions.

| benchmark | target | reported result | met? |
|---|---|---|---|
| Q3 regime AUC, classical walk-forward | >= 0.75 | 0.6928 | not met |
| Q3 nowcast error vs random walk with drift | lower than benchmark | bridge 0.7095 to 0.8784 vs 1.258 ex-COVID | met |
| Q3 leakage ablation quantified | quantified | Brier and AUC premiums with 95% intervals in Table 35 | met |

The benchmark table contains only the brief's indicative targets.

## 11. Differences from the brief

1. **LSTM refit frequency.** Q3B refits the LSTM every six months and reuses the last refit between refits; Q3A refits the probit monthly. Source: `results/approach2/macro/metrics.json`.
2. **Cluster bridge.** Q3A clusters partition indicators, while the LSTM's month-level `h_t` clusters partition months, so a direct ARI is undefined. The bridge maps each indicator to a correlation vector with the 16 hidden dimensions and clusters those vectors with `k=9`; the resulting ARI may be partly mechanical because `h_t` is computed from those indicators. Source: `results/approach2/macro/metrics.json`.
3. **Q3B inputs.** The LSTM excludes `gdp_yoy` from the 12-month input panel to avoid a mixed-frequency quarterly series; the probit also excludes it from the factors. Source: `results/approach2/macro/metrics.json`.
4. **Individual submission.** This is an individual submission rather than a group submission.
5. **AI-drafted report.** The report was drafted by AI rather than written by a group, with instructor approval as stated by the author in Section 14.
6. **FastboxData not used.** FastboxData was not used as a data source.

## 12. Limitations

- Macro data are current revised vintages, not real-time vintages. Most publication lags are conservative assumptions, not observed release dates. GDP is derived from cumulative real levels (79 derived quarters; validation maximum absolute difference 0.83 points).
- The retrained LSTM never exceeds 0.5 on the test block (maximum 0.3172) and does not beat persistence (Brier 0.2561 vs 0.2286; AUC 0.6422 vs 0.7390). Its accuracy is the all-negative rate.
- The PCA rank is chosen on 12 quarterly nowcast observations; the test block is short for regime classification, and the AUC intervals are wide (for example, the original probit minus persistence interval is `[-0.2375, 0.03518]`).
- The Q3 nowcast month-3 ex-COVID error is higher than month-1; the paired interval includes zero, and no cause is asserted.
- The Q3 clustering is unstable across windows (adjacent ARI median 0.1648) and is descriptive only.

## 13. Session integrity

Known writes outside the repository root, from the session record:

- `/tmp/q1_online.log`, `/tmp/q1_online_resume.log`, `/tmp/q1_online_resume2.log`, `/tmp/q1_expiry_fix.log`: logs from the early Q1 data downloads.
- `/tmp/q1a_close.log`: log from the early Q1A close run (still present; not deleted).
- `/tmp/offline_check.log`, `/tmp/offline_check2.log`, `/tmp/uv_sync_final.log`: reproduction and environment checks.
- `~/.cache/uv`: possible writes from early `uv run` calls before `UV_CACHE_DIR` was pinned to the repository.
- `~/.local/share/opencode/tool-output/`: automatic tool-output files written by the harness.

Tool setup and other session facts:

- Before the first working session, the desktop app and CLI conflicted over one database, so the file in `~/.local/share/opencode` was moved by hand to a backup folder. This was tool setup, not a project change.
- An accidental one-word prompt was sent to a free model (session 01); it ran read-only commands and was stopped.
- Free quota ran out on 2026-11-04, so the session continued on OpenCode Go with the same model family.
- During reproduction the agent copied the uv-managed Python into `.cache` with read access to `~/.local/share/uv`, and deleted its own temporary clone under `.cache` without a separate approval.
- Claude also renamed two misnamed session files and read repository files, on the user's computer, without changing any results.

The report step itself wrote only `report.md` and a verification script under `.cache/`; it did not change code, data or results. The working tree already contained a modified `results/comparison.html` on entry; it was not touched.

The split work itself ran with `ap2026-split` as its root; `../ap2026` was read-only, and the split repositories were created afterwards from parent tag `submission`.

## 14. Statement of work

This is an individual submission by Hao Lu. The instructor approved individual completion with AI assistance. opencode with DeepSeek V4.1 Flash wrote the code, ran the experiments and drafted this report. Claude, in a separate chat that is not part of SESSION.md, drafted most of the prompts, proposed many methodology decisions and corrections, and reviewed outputs. Hao Lu directed the work, approved each step, and owns the final submission after reviewing it. The author does not claim to have independently derived the mathematics; the derivations above are a record of the implemented methods and their sources. The author has read this report.

## 15. Causal claims

Table 39: Prose claims about causes and their status. Source: `results/approach1/macro/metrics.json`, `results/approach2/macro/metrics.json`, and the report text.

| claim | status |
|---|---|
| The original LSTM failure is caused by the 18-month validation design | evidence, not isolated: 4 of 24 validation blocks were single-class and tuning AUC was far below test AUC, but retraining changed hyperparameters and validation together |
| The larger original leakage premiums are caused by the unreliable 18-month validation design | hypothesis/evidence: the original design had single-class blocks and a large tuning-to-test gap, but the premium change is not isolated from the model change |
| The Q3A low AUC is caused by persistence and PMI being in the information set | hypothesis; not tested |
| The Q3B bridge ARI significance is partly mechanical because `h_t` is computed from the indicators | hypothesis; the null permutes labels but not the full construction |
| The Q3B candidate-only keys are config echoes, not numeric differences | tested by the checker: 10,180 shared numeric values exact; 216 `history.refits[*].min_epochs` keys only in the reproduced run |

## 16. Result files

Every number in the tables is from one of the following result files unless the table caption says otherwise: `results/approach1/macro/metrics.json`, `results/approach2/macro/metrics.json`. Metadata and row counts come from `data/metadata.json`. The clean-clone totals in Section 2 are from the `code/check_reproduction.py` output, and the session-integrity facts are from the session record. Former `approach5` is now `approach1`; former `approach6` is now `approach2`. The reproduced `approach2/macro/metrics.json` has 216 `history.refits[*].min_epochs` config echoes not present in the committed file; all shared numeric values match.

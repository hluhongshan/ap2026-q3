# Split history for ap2026-q3

## Provenance and split disclosure

The work was done in one repository (parent tag `submission`, commit `4521da3`) and split afterwards into one repository per question following the instructor's guidance. Commit hashes changed and the old-to-new commit map is included below. Dates in rewritten commits are the original ones; the split commits are dated when they were made. Shared-file commits appear in more than one repository. `SESSION.md` is the full transcript for all questions and refers to old file names and hashes. Agent work on the split ran with `ap2026-split` as its root.

## Path mapping

| Old path or name | New path or name |
|---|---|
| `code/approach5_q3a_classical.py` | `code/approach1_q3a_classical.py` |
| `code/approach5_q3a_posthoc.py` | `code/approach1_q3a_posthoc.py` |
| `code/approach6_q3b_sequence.py` | `code/approach2_q3b_sequence.py` |
| `code/approach6_q3b_posthoc.py` | `code/approach2_q3b_posthoc.py` |
| `results/approach5/` | `results/approach1/` |
| `results/approach6/` | `results/approach2/` |

Some code comments, labels and JSON keys keep the old names, such as approach3 or approach5_members. They were left unchanged so that results stay directly comparable with the parent.

## Commit map

The map is the unedited `git filter-repo` output. `pruned` means the commit became empty for this path set. `same` means filter-repo produced the identical hash; `rewritten` means the hash changed. Shared commits are listed with every repository that retained them.

| Old commit | New commit | Subject | Status | Retained in |
|---|---|---|---|---|
| `7ee9a860ad8a` | `7ee9a860ad8a` | chore: add agent contract before first agent session | same | Q1, Q2, Q3 |
| `bc857ecc516a` | `bc857ecc516a` | chore: scaffold repository | same | Q1, Q2, Q3 |
| `867a243cd90d` | `165ee8cde0ca` | data: add Q1 fetch pipeline and snapshot | rewritten | Q1, Q2, Q3 |
| `278562b16b99` | `c339f374e301` | data: roll holiday option expiries to next trading day | rewritten | Q1, Q2, Q3 |
| `0b680be37ac1` | `41cfe2c03aa2` | feat: Q1A SVI pipeline | rewritten | Q1, Q2, Q3 |
| `c8c840c4b3e4` | pruned | exp: Q1A results on IO | pruned | Q1 |
| `317fe9c55a8a` | pruned | fix: Q1A repricing basis and SVI bounds | pruned | Q1 |
| `18b5cc2f237d` | pruned | exp: Q1A results, settle primary | pruned | Q1 |
| `0e0a6b91984a` | pruned | chore: remove stale Q1A run | pruned | Q1 |
| `7551ad29752f` | pruned | exp: like-for-like forward check | pruned | Q1 |
| `0af1dd09691f` | pruned | feat: Q1B differentiable SVI and neural surface | pruned | Q1 |
| `ab3313adfc8f` | pruned | fix: Q1B robustness coverage reporting | pruned | Q1 |
| `664a71673bbd` | pruned | exp: Q1B results on IO | pruned | Q1 |
| `84a746a51ab8` | pruned | docs: record Q1B open issues | pruned | Q1 |
| `5f007e65d2ca` | `95f772ea07ce` | data: add Q2 futures snapshot | rewritten | Q1, Q2, Q3 |
| `7950d6fe1852` | `212670e3c490` | feat: Q2A Monte Carlo engine and estimators | rewritten | Q1, Q2, Q3 |
| `4fd0ca365927` | pruned | exp: Q2A results | pruned | Q2 |
| `32bfbcdf9c77` | pruned | fix: Q2A study reference consistency | pruned | Q2 |
| `89a27c6db5ed` | pruned | exp: refresh Q2A study outputs | pruned | Q2 |
| `d67df3903f33` | pruned | feat: Q2B deep hedging | pruned | Q2 |
| `b2a71278ed7d` | pruned | fix: Q2B default output path | pruned | Q2 |
| `fcdaf1096e58` | pruned | exp: Q2B results | pruned | Q2 |
| `4fc1d1053860` | `59d86be54385` | data: add Q3 macro panel | rewritten | Q1, Q2, Q3 |
| `6255defa60cc` | `2fab6c25d30f` | feat: Q3A factors, nowcast, probit, clustering | rewritten | Q3 |
| `1922099796cc` | `0742ff9ba542` | exp: Q3A results | rewritten | Q3 |
| `d2cf1893d830` | `2f2297c80e65` | exp: Q3A post-hoc probit regularization check | rewritten | Q3 |
| `a87b9f3dfb18` | `7799a329ad31` | feat: Q3B LSTM and leakage ablation | rewritten | Q3 |
| `5131fe20204d` | `56ab6044580b` | exp: Q3B results | rewritten | Q3 |
| `9790ce9e52a3` | `aa28eb410e00` | fix: Q3B training loop diagnostics | rewritten | Q3 |
| `17bbfd75a41e` | `982feed7ef20` | exp: Q3B post-hoc retraining | rewritten | Q3 |
| `fb4781adbadd` | `0de2cc4555ea` | feat: compare layer and reproduction script | rewritten | Q1, Q2, Q3 |
| `b5c09a83baf1` | `76272e11905d` | docs: update agent contract and README | rewritten | Q1, Q2, Q3 |
| `7290853b7bbe` | `11e617249e4c` | fix: Q1B objective scaling and shared initialization | rewritten | Q1, Q2, Q3 |
| `d4a7f215effb` | `33751088ac9c` | exp: Q1B results after objective scaling | rewritten | Q1, Q2, Q3 |
| `80009968446d` | pruned | docs: correct Q1B tenor-bucket aggregation | pruned | Q1 |
| `076e9d43a48c` | `6c45c48cf6e8` | fix: use repo-relative paths in outputs | rewritten | Q1, Q2, Q3 |
| `e6ae1a473870` | `c102550c2c89` | docs: note historical approach2/io run | rewritten | Q1, Q2, Q3 |
| `ab12e7080c73` | `576bd7e016e7` | data: commit CFFEX raw snapshot for offline Q1 cross-check | rewritten | Q1, Q2, Q3 |
| `4a6164be766d` | `7cc52e3f021d` | fix: commit legacy SVI snapshots used by Q1 comparisons | rewritten | Q1, Q2, Q3 |
| `509ac9f68ea4` | `b0089e2c1240` | chore: add reproduction check | rewritten | Q1, Q2, Q3 |
| `311bade4fe68` | `77017bad1094` | docs: document reproduction caveats | rewritten | Q1, Q2, Q3 |
| `8105e5dfd0ec` | `d39b2fb18f3f` | exp: refresh Q1A close results from current code | rewritten | Q1, Q2, Q3 |
| `b226622810fd` | `8534a90b30b6` | fix: Q3B post-hoc origin_date label | rewritten | Q3 |
| `e27222bdeaf5` | `d4eec3c16f97` | feat: Q1B cold-start SVI solver comparison | rewritten | Q1, Q2, Q3 |
| `99210e96e740` | pruned | exp: Q1B cold-start solver results | pruned | Q1 |
| `3015c7822b0c` | `dff707076a7f` | report: add project report | rewritten | Q1, Q2, Q3 |
| `dad4ca07163c` | `f09d871884e6` | report: author review confirmation | rewritten | Q1, Q2, Q3 |
| `4521da3ed674` | `19f2f73ddf98` | report: add complete session export | rewritten | Q1, Q2, Q3 |

## Commit counts

- Original parent history: 48 commits.
- Commits retained by this repository after filtering: 31.
- Later split commits are not part of the filter-repo map; they are ordinary commits on top of the filtered history and are dated when made.
- No `submission` tag was created by the split; the parent tag is recorded above.

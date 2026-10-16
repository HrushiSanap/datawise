# DATAwise: final results report

Config hash: 03fa342b535795c1. All numbers are read from results/master_results.csv and the figure backing CSVs.

## Dataset and design
- PCam balanced eval splits: valid 32,738, test 32,754, each exactly 50/50, trimmed once with EVAL_BALANCE_SEED.
- Models: logistic, cnn, resnet18. Main seeds: [21, 22, 23] (data=init=loader tied). Step budget 5000, batch 128, fp32, from scratch.
- PCam main successful runs aggregated: 221; init-sensitivity runs held separate: 20.

## Per-protocol findings (observational)
- Protocol A (balanced, varying total N): see figure_02 CSV for per-model trends in bal_acc, auroc, brier, minority_ece and aurc_minority as total N grows.
- Protocol B (fixed total N_F=64000, varying ratio): the r=100 condition sits at actual ratio ~99.95 by the pinned count formula. See figure_03 CSV.
- Protocol C (minority fixed at 1000, growing majority): see figure_04 CSV for how minority metrics move as majority exposure increases at fixed minority availability.
- Protocol D (majority fixed at 32000, removing minority): see figure_05 CSV for minority metrics as minority availability falls at fixed majority exposure.

## Class-conditional calibration headline (Protocol B extreme, resnet18)
- minority_ece (raw): 0.8140 (sd 0.0021, n=3)
- minority_ece (prior): 0.1502 (sd 0.0116, n=3)
- minority_ece (TS): 0.5129 (sd 0.0019, n=3)
- majority_ece (raw): 0.0055 (sd 0.0002, n=3)
- recall_pos (raw): 0.0978 (sd 0.0028, n=3)
- recall_pos (prior-corrected): 0.6729 (sd 0.0091, n=3)
  Numbers reported; no conclusion drawn here on whether a remedy removes the gap.

## Decomposition
Decomposition (Figure 6): Protocol C isolates changes associated with increasing majority exposure at fixed minority availability; Protocol D isolates changes associated with decreasing minority availability at fixed majority exposure; Protocol B shows both moving together. No numeric subtraction between protocols is performed. Values in figure_06 and figure_06b CSVs.

## Model ladder
Model ladder (Figure 7): minority_ece vs ratio for logistic, cnn and resnet18 along Protocol B. Whether the residual net amplifies, attenuates or leaves the minority penalty unchanged is read from figure_07 CSV; three rungs is a coarse ladder.

## Recalibration diagnostic
Recalibration (Figure 11): minority_ece and classwise_ece in raw, prior and TS states across Protocol B ratios, for all three models; and recall_pos beside recall_pos_prior in figure_11b CSV. Numbers reported; no conclusion drawn here about whether any remedy removes the class-conditional gap.

## ISIC external check
ISIC external check (Figure 12): recall_pos, auroc, brier and aurc_minority with stratified percentile bootstrap CIs. lo/hi are per-run bootstrap bounds averaged over the three seeds, not an interval on the three-seed mean. ISIC temperature scaling is excluded (fit in-sample on the holdout). The claim is that the direction of the minority-reliability effect is consistent in a second domain, not that it generalises across medical imaging.

## Initialisation sensitivity
Initialisation sensitivity (Figure 13): the five-init-seed spread of minority_ece (data and loader seeds fixed at 21) is initialisation-only variability; the three-seed spread (seeds 21, 22, 23, with data, initialisation and loader order tied) bundles all three sources and is not pure data-sampling variance. Values and run counts in figure_13 CSV.

## Consistency and reproducibility
The (32000, 32000) point resolves to a single run_id per model-seed across Protocols A, B and D (9 rows), so its metrics are identical by construction wherever a protocol references it.
- Single config_hash across all runs: 03fa342b535795c1.
- Saved artefacts: class permutations, subset indices, balanced-eval indices, per-run test and validation logits, training curves, frozen config.
- Metric computation is isolated in metrics.py, a deliberate and documented departure from the original blueprint (which placed metrics in core.py), so the frozen run engine was never edited after the runs began. No scientific definition changed.
- Evaluation splits trimmed to exact 50/50 once at load by a fixed seeded rule, kept indices saved; every reported metric is per-class, rank-based or a per-sample proper score and is robust to that trim.

## Data quality
- See summaries/07_data_quality.md for the calibration-collapse check (ece vs minority_ece and classwise_ece), NaN/inf, temperature-bound flags, seed coverage and hash uniqueness.

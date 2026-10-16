# Notebook 05: ISIC external check session summary

Active config hash: 03fa342b535795c1

## Data construction
- deduped pool: 32,701 (malignant 581 / benign 32,120), dropping every image_name_2
- holdout: 184 / 184 = 368 images, seed 20200, image-disjoint after dedup
- training pool: malignant 397 / benign 31,936
- fixed training minority: 384 malignant
- ratios: [1, 5, 10, 25, 50], n_maj = 384 * r = [384, 1920, 3840, 9600, 19200]
- resize: 96x96 bilinear, decoded and cached once
- ISIC norm mean: 0.805821, 0.620952, 0.590992
- ISIC norm std: 0.147848, 0.174740, 0.200526

## Runs
- expected unique runs: 45 (5 conditions x 3 models x 3 seeds)
- success: 45
- failed: 0
- pending: 0

## Caveat
- the holdout is image-disjoint after duplicate removal; patient-level separation is not enforced by design. ISIC is a secondary consistency check, not a leakage-controlled benchmark.

## Failed runs
- none

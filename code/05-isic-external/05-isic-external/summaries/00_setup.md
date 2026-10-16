# Notebook 00: setup summary

Active config hash: 03fa342b535795c1

## Raw class counts
- train pos/neg: 131072 / 131072
- valid pos/neg: 16369 / 16399
- test pos/neg: 16377 / 16391

## Balanced evaluation splits
- valid: 16369 / 16369 (total 32738)
- test: 16377 / 16377 (total 32754)
- kept index files: valid_kept=32738, test_kept=32754

## Normalisation provenance (tolerance 1e-3, all channels within)
- mean recomputed: 0.700756, 0.538358, 0.691621
- mean reference: 0.700756, 0.538358, 0.691621
- std recomputed: 0.234978, 0.277409, 0.212895
- std reference: 0.234978, 0.277409, 0.212895
- gate result: PASSED

## Shared-condition unit test
- condition_id: m32000_M32000
- run_id (cnn, seeds 21/21/21): cnn__m32000_M32000__d21_i21_l21
- subset length: 64000
- A/B/D identical: PASS

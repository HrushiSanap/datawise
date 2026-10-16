# Notebook 06: initialisation-sensitivity check session summary

Active config hash: 03fa342b535795c1

- expected unique runs: 20 (2 models x 2 conditions x 5 init seeds)
- success: 20
- failed: 0
- pending: 0
- this check isolates initialisation noise: data_seed and loader_seed are fixed at 21 and only init_seed varies over 21-25. It does not measure data-sampling variance.
- the init_seed=21 runs share run_ids with main-experiment runs and are reused via resume, not retrained.

## Failed runs
- none

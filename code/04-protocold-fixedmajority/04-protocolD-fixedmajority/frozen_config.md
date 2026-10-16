# Frozen configuration

Written by notebook 00 at setup. Authoritative for the runs.

- step_budget: 5000
- n_f: 64000
- batch_size: 128
- warmup: 500
- use_amp: false
- eval_balance_seed: 12345
- seeds: 21, 22, 23
- hp_logistic: lr=0.0003, wd=0.01
- hp_cnn: lr=0.0003, wd=0.0001
- hp_resnet18: lr=0.0003, wd=0.0001
- norm_mean: 0.700756, 0.538358, 0.691621
- norm_std: 0.234978, 0.277409, 0.212895
- config_hash: 03fa342b535795c1

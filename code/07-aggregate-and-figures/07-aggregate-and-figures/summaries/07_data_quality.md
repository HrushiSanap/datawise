# Notebook 07: data-quality dashboard

- max|ece - minority_ece| across runs: 0.449413
- max|ece - classwise_ece| across runs: 0.067308
  calibration metrics are genuinely class-conditional (collapse fixed)
- logit files (test+val): 482; master rows x2 = 482
- NaN/inf: none in any metric column
- temperature: no runs near the optimiser bounds
- short runs (actual_steps != step_budget): 0
- config_hash single-valued: True (['03fa342b535795c1'])
- seed coverage per (protocol-tag, model, condition), expecting {21,22,23}:
    GAP protocolA cnn m2000_M2000: seeds [22, 23]
    GAP protocolA logistic m2000_M2000: seeds [22, 23]
    GAP protocolC cnn m1000_M1000: seeds [22, 23]
    GAP protocolC logistic m1000_M1000: seeds [22, 23]
- PCam main successful runs (non-init): 221
- init-sensitivity runs (held separate): 20
  (expected ~241 main; print actual rather than asserting a remembered number)

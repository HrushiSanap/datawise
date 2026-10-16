"""metrics.py: offline metric computation for the PCam data-distribution study.

Companion to the frozen core.py. Pure and path-free: computes from numpy
arrays and returns numbers or small dicts. No training, no CUDA, no dataset
reads. Imported only by the aggregation notebook (07), after the experiment
is frozen, so it is allowed to see test logits.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


ECE_BINS = 15
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 13579
BOOTSTRAP_CI = 0.95
RISK_COVERAGE_POINTS = 100


def _trapz(y, x):
    # np.trapz is deprecated in NumPy 2.x; prefer np.trapezoid where present.
    fn = getattr(np, "trapezoid", None)
    if fn is None:
        fn = np.trapz
    return float(fn(y, x))


# ============================================================
# Conversions
# ============================================================

def probs_from_logit_diff(logit_diff):
    """Positive-class probability from the logit difference via the sigmoid."""
    d = np.asarray(logit_diff, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-d))


def _predicted_positive(p_pos):
    return np.asarray(p_pos) >= 0.5


# ============================================================
# Discrimination
# ============================================================

def recall_per_class(y_true, p_pos):
    """Per-class recall at threshold 0.5, returned as (recall_pos, recall_neg)."""
    y = np.asarray(y_true).astype(int)
    pred = _predicted_positive(p_pos).astype(int)
    pos = y == 1
    neg = y == 0
    recall_pos = float((pred[pos] == 1).mean()) if pos.any() else float("nan")
    recall_neg = float((pred[neg] == 0).mean()) if neg.any() else float("nan")
    return recall_pos, recall_neg


def balanced_accuracy(y_true, p_pos):
    """Mean of the two per-class recalls at threshold 0.5."""
    recall_pos, recall_neg = recall_per_class(y_true, p_pos)
    return float(np.nanmean([recall_pos, recall_neg]))


def auroc(y_true, p_pos):
    """Area under the ROC curve, rank-based and threshold-free."""
    y = np.asarray(y_true).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(p_pos)))


# ============================================================
# Calibration (proper scores lead; ECE is secondary, bin-sensitive)
# ============================================================

def brier(y_true, p_pos):
    """Brier score, the mean squared error of the probability, no binning."""
    y = np.asarray(y_true).astype(np.float64)
    p = np.asarray(p_pos, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def nll(y_true, p_pos, eps=1e-7):
    """Binary negative log-likelihood with probabilities clamped, no binning."""
    y = np.asarray(y_true).astype(np.float64)
    p = np.clip(np.asarray(p_pos, dtype=np.float64), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _bin_edges(n_bins):
    return np.linspace(0.0, 1.0, n_bins + 1)


def _binned_gap(scores, correct, n_bins):
    # 15 equal-width bins on [0,1], population-weighted gap between a score
    # (confidence) and a target (accuracy); equal-width (not equal-count) is
    # pinned by the blueprint, and ECE is kept secondary to Brier/NLL because
    # it is bin-sensitive. scores and correct are already aligned 1-D arrays.
    edges = _bin_edges(n_bins)
    n = len(scores)
    if n == 0:
        return float("nan")
    total = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b == n_bins - 1:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        if not mask.any():
            continue
        count = int(mask.sum())
        conf = float(scores[mask].mean())
        acc = float(correct[mask].mean())
        total += (count / n) * abs(conf - acc)
    return total


def ece(y_true, p_pos, n_bins=ECE_BINS):
    """Top-label expected calibration error, population-weighted confidence-accuracy gap."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_pos, dtype=np.float64)
    pred = _predicted_positive(p).astype(int)
    correct = (pred == y).astype(np.float64)
    confidence = np.maximum(p, 1.0 - p)
    return float(_binned_gap(confidence, correct, n_bins))


def reliability_bins(y_true, p_pos, n_bins=ECE_BINS, mode="top_label"):
    """Per-bin edges, counts, mean confidence and accuracy for reliability diagrams.

    mode='top_label' (default, unchanged for existing callers): bins by top-label
    confidence max(p,1-p) with correct = 1[pred==y]. mode='as_given': bins by p_pos
    directly with accuracy = mean(y_true) in each bin, for a one-vs-rest or already
    subset-and-filtered (y, p) diagram supplied by the caller.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_pos, dtype=np.float64)
    if mode == "top_label":
        pred = _predicted_positive(p).astype(int)
        target = (pred == y).astype(np.float64)
        score = np.maximum(p, 1.0 - p)
    elif mode == "as_given":
        target = y.astype(np.float64)
        score = p
    else:
        raise ValueError(f"unknown reliability mode {mode}")
    edges = _bin_edges(n_bins)

    bin_lo, bin_hi, bin_count, bin_conf, bin_acc = [], [], [], [], []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b == n_bins - 1:
            mask = (score >= lo) & (score <= hi)
        else:
            mask = (score >= lo) & (score < hi)
        count = int(mask.sum())
        bin_lo.append(float(lo))
        bin_hi.append(float(hi))
        bin_count.append(count)
        bin_conf.append(float(score[mask].mean()) if count else float("nan"))
        bin_acc.append(float(target[mask].mean()) if count else float("nan"))
    return (np.array(bin_lo), np.array(bin_hi), np.array(bin_count),
            np.array(bin_conf), np.array(bin_acc))


def classwise_ece(y_true, p_pos, n_bins=ECE_BINS):
    """One-vs-rest ECE: for each class k bin ALL examples by p_k, averaged over the two classes."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_pos, dtype=np.float64)
    per_class = []
    for k in (0, 1):
        p_k = p if k == 1 else (1.0 - p)
        is_k = (y == k).astype(np.float64)
        per_class.append(_binned_gap(p_k, is_k, n_bins))
    return float(np.mean(per_class))


def _toplabel_ece_subset(y_true, p_pos, subset_mask, n_bins):
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_pos, dtype=np.float64)
    m = np.asarray(subset_mask).astype(bool)
    if not m.any():
        return float("nan")
    ys, ps = y[m], p[m]
    pred = (ps >= 0.5).astype(int)
    correct = (pred == ys).astype(np.float64)
    confidence = np.maximum(ps, 1.0 - ps)
    return _binned_gap(confidence, correct, n_bins)


def minority_ece(y_true, p_pos, n_bins=ECE_BINS):
    """Top-label ECE on the true-minority (y==1) subset; a class-conditional diagnostic."""
    y = np.asarray(y_true).astype(int)
    return float(_toplabel_ece_subset(y, p_pos, y == 1, n_bins))


def majority_ece(y_true, p_pos, n_bins=ECE_BINS):
    """Top-label ECE on the true-majority (y==0) subset; a class-conditional diagnostic."""
    y = np.asarray(y_true).astype(int)
    return float(_toplabel_ece_subset(y, p_pos, y == 0, n_bins))


# ============================================================
# Uncertainty usefulness
# ============================================================

def _binary_entropy_bits(p_pos, eps=1e-12):
    p = np.clip(np.asarray(p_pos, dtype=np.float64), eps, 1.0 - eps)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def confidence_entropy_split(y_true, p_pos):
    """Mean confidence and binary Shannon entropy (bits) split by correctness at 0.5."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_pos, dtype=np.float64)
    pred = _predicted_positive(p).astype(int)
    correct = pred == y
    incorrect = ~correct
    confidence = np.maximum(p, 1.0 - p)
    entropy = _binary_entropy_bits(p)

    def _m(a, mask):
        return float(a[mask].mean()) if mask.any() else float("nan")

    mean_conf_correct = _m(confidence, correct)
    mean_conf_incorrect = _m(confidence, incorrect)
    return {
        "mean_conf_correct": mean_conf_correct,
        "mean_conf_incorrect": mean_conf_incorrect,
        "mean_entropy_correct": _m(entropy, correct),
        "mean_entropy_incorrect": _m(entropy, incorrect),
        "delta_conf": mean_conf_correct - mean_conf_incorrect,
    }


def confidence_entropy_split_by_class(y_true, p_pos):
    """Mean confidence and entropy (bits) split by correctness within each true-class subset."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_pos, dtype=np.float64)
    pred = _predicted_positive(p).astype(int)
    correct = pred == y
    confidence = np.maximum(p, 1.0 - p)
    entropy = _binary_entropy_bits(p)

    def _m(a, mask):
        return float(a[mask].mean()) if mask.any() else float("nan")

    out = {}
    for tag, cls in (("min", 1), ("maj", 0)):
        sub = y == cls
        out[f"mean_conf_correct_{tag}"] = _m(confidence, sub & correct)
        out[f"mean_conf_incorrect_{tag}"] = _m(confidence, sub & ~correct)
        out[f"mean_entropy_correct_{tag}"] = _m(entropy, sub & correct)
        out[f"mean_entropy_incorrect_{tag}"] = _m(entropy, sub & ~correct)
    return out


def risk_coverage(y_true, p_pos, n_points=RISK_COVERAGE_POINTS, subset_mask=None):
    """Selective-risk curve and its area (AURC), sweeping coverage by confidence.

    Minority-AURC is a class-conditional DIAGNOSTIC, not an operational
    selective-prediction metric, since the model does not know the true class
    at deployment.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_pos, dtype=np.float64)
    pred = _predicted_positive(p).astype(int)
    error = (pred != y).astype(np.float64)
    confidence = np.maximum(p, 1.0 - p)

    if subset_mask is not None:
        subset_mask = np.asarray(subset_mask).astype(bool)
        error = error[subset_mask]
        confidence = confidence[subset_mask]

    n = len(error)
    if n == 0:
        return np.array([]), np.array([]), float("nan")

    order = np.argsort(-confidence)
    error_sorted = error[order]
    cum_error = np.cumsum(error_sorted)
    counts = np.arange(1, n + 1)
    risk_at_count = cum_error / counts

    coverage_grid = np.linspace(1.0 / n, 1.0, n_points)
    k = np.clip(np.round(coverage_grid * n).astype(int), 1, n)
    risk_grid = risk_at_count[k - 1]
    aurc = _trapz(risk_grid, coverage_grid)
    return coverage_grid, risk_grid, aurc


def aurc_overall_minority_majority(y_true, p_pos):
    """AURC overall, and class-conditional on the minority (1) and majority (0)."""
    y = np.asarray(y_true).astype(int)
    _, _, aurc_all = risk_coverage(y, p_pos)
    _, _, aurc_min = risk_coverage(y, p_pos, subset_mask=(y == 1))
    _, _, aurc_maj = risk_coverage(y, p_pos, subset_mask=(y == 0))
    return aurc_all, aurc_min, aurc_maj


def error_detection_auroc(y_true, p_pos):
    """AUROC separating incorrect from correct predictions using entropy as the score."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_pos, dtype=np.float64)
    pred = _predicted_positive(p).astype(int)
    incorrect = (pred != y).astype(int)
    if len(np.unique(incorrect)) < 2:
        return float("nan")
    entropy = _binary_entropy_bits(p)
    return float(roc_auc_score(incorrect, entropy))


# ============================================================
# Three calibration states
# ============================================================

def fit_temperature(y_true_val, logit_diff_val):
    """Scalar temperature T>0 minimising validation NLL of sigmoid(logit_diff/T)."""
    y = np.asarray(y_true_val).astype(np.float64)
    d = np.asarray(logit_diff_val, dtype=np.float64)

    def _obj(T):
        if T <= 0:
            return np.inf
        return nll(y, probs_from_logit_diff(d / T))

    try:
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(_obj, bounds=(0.05, 20.0), method="bounded")
        return float(res.x)
    except Exception:
        grid = np.linspace(0.05, 20.0, 400)
        losses = np.array([_obj(t) for t in grid])
        best = grid[int(np.argmin(losses))]
        refine = np.linspace(max(0.05, best - 0.1), best + 0.1, 200)
        rlosses = np.array([_obj(t) for t in refine])
        return float(refine[int(np.argmin(rlosses))])


def apply_temperature(logit_diff, T):
    """Scale the logit difference by the fitted temperature."""
    return np.asarray(logit_diff, dtype=np.float64) / float(T)


def _logit(p):
    return np.log(p / (1.0 - p))


def prior_correct(logit_diff, pi_train, pi_test=0.5):
    """Shift the logit by the difference in prior log-odds between train and test.

    d_corrected = d - (logit(pi_train) - logit(pi_test)); the balanced test set
    gives pi_test=0.5 so logit(pi_test)=0; pi_train = n_min/N from the ACTUAL
    condition counts, not the target ratio; this tests whether the minority gap
    is merely train/test prior mismatch, and the claim stays scoped to "does not
    remove the class-conditional gap".
    """
    d = np.asarray(logit_diff, dtype=np.float64)
    shift = _logit(pi_train) - _logit(pi_test)
    return d - shift


# ============================================================
# Bootstrap (ISIC reporting only)
# ============================================================

def bootstrap_ci(y_true, p_pos, metric_fn, n=BOOTSTRAP_N, ci=BOOTSTRAP_CI,
                 seed=BOOTSTRAP_SEED, stratified=True):
    """Percentile bootstrap confidence interval for a metric.

    Resampled STRATIFIED within each class so a balanced holdout stays balanced
    across resamples; used for ISIC only, PCam uncertainty comes from the three
    seeds and is never bootstrapped.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_pos, dtype=np.float64)
    rng = np.random.default_rng(seed)
    point = float(metric_fn(y, p))

    if stratified:
        idx_pos = np.where(y == 1)[0]
        idx_neg = np.where(y == 0)[0]

    samples = []
    for _ in range(n):
        if stratified:
            bp = rng.choice(idx_pos, size=len(idx_pos), replace=True)
            bn = rng.choice(idx_neg, size=len(idx_neg), replace=True)
            idx = np.concatenate([bp, bn])
        else:
            idx = rng.choice(len(y), size=len(y), replace=True)
        val = metric_fn(y[idx], p[idx])
        if np.isfinite(val):
            samples.append(val)

    if not samples:
        return point, float("nan"), float("nan")
    lo_pct = (1.0 - ci) / 2.0 * 100.0
    hi_pct = (1.0 + ci) / 2.0 * 100.0
    lo = float(np.percentile(samples, lo_pct))
    hi = float(np.percentile(samples, hi_pct))
    return point, lo, hi


# ============================================================
# Top-level assembly
# ============================================================

def compute_all_metrics(y_true_test, logit_diff_test, y_true_val, logit_diff_val,
                        pi_train, with_bootstrap=False):
    """Compute the full master_results.csv metric set across raw, TS and prior states."""
    y_test = np.asarray(y_true_test).astype(int)
    d_test = np.asarray(logit_diff_test, dtype=np.float64)
    p_raw = probs_from_logit_diff(d_test)

    bal_acc = balanced_accuracy(y_test, p_raw)
    recall_pos, recall_neg = recall_per_class(y_test, p_raw)
    au = auroc(y_test, p_raw)
    br = brier(y_test, p_raw)
    nl = nll(y_test, p_raw)
    e_raw = ece(y_test, p_raw)
    cw_raw = classwise_ece(y_test, p_raw)
    min_raw = minority_ece(y_test, p_raw)
    maj_raw = majority_ece(y_test, p_raw)

    ce = confidence_entropy_split(y_test, p_raw)
    ce_cls = confidence_entropy_split_by_class(y_test, p_raw)
    aurc_all, aurc_min, aurc_maj = aurc_overall_minority_majority(y_test, p_raw)
    err_au = error_detection_auroc(y_test, p_raw)

    T = fit_temperature(y_true_val, logit_diff_val)
    d_ts = apply_temperature(d_test, T)
    p_ts = probs_from_logit_diff(d_ts)
    nll_ts = nll(y_test, p_ts)
    e_ts = ece(y_test, p_ts)
    cw_ts = classwise_ece(y_test, p_ts)
    min_ts = minority_ece(y_test, p_ts)
    maj_ts = majority_ece(y_test, p_ts)

    d_prior = prior_correct(d_test, pi_train)
    p_prior = probs_from_logit_diff(d_prior)
    e_prior = ece(y_test, p_prior)
    cw_prior = classwise_ece(y_test, p_prior)
    min_prior = minority_ece(y_test, p_prior)
    maj_prior = majority_ece(y_test, p_prior)
    br_prior = brier(y_test, p_prior)
    nl_prior = nll(y_test, p_prior)
    recall_pos_prior, recall_neg_prior = recall_per_class(y_test, p_prior)

    out = {
        "bal_acc": bal_acc,
        "recall_pos": recall_pos,
        "recall_neg": recall_neg,
        "auroc": au,
        "brier": br,
        "nll": nl,
        "ece": e_raw,
        "classwise_ece": cw_raw,
        "minority_ece": min_raw,
        "majority_ece": maj_raw,
        "mean_conf_correct": ce["mean_conf_correct"],
        "mean_conf_incorrect": ce["mean_conf_incorrect"],
        "mean_entropy_correct": ce["mean_entropy_correct"],
        "mean_entropy_incorrect": ce["mean_entropy_incorrect"],
        "delta_conf": ce["delta_conf"],
        "aurc_overall": aurc_all,
        "aurc_minority": aurc_min,
        "aurc_majority": aurc_maj,
        "error_auroc": err_au,
        "temperature": T,
        "nll_ts": nll_ts,
        "ece_ts": e_ts,
        "classwise_ece_ts": cw_ts,
        "minority_ece_ts": min_ts,
        "majority_ece_ts": maj_ts,
        "ece_prior": e_prior,
        "classwise_ece_prior": cw_prior,
        "minority_ece_prior": min_prior,
        "majority_ece_prior": maj_prior,
        "brier_prior": br_prior,
        "nll_prior": nl_prior,
        "recall_pos_prior": recall_pos_prior,
        "recall_neg_prior": recall_neg_prior,
    }
    out.update(ce_cls)

    if with_bootstrap:
        _, rp_lo, rp_hi = bootstrap_ci(y_test, p_raw, lambda y, p: recall_per_class(y, p)[0])
        _, au_lo, au_hi = bootstrap_ci(y_test, p_raw, auroc)
        _, br_lo, br_hi = bootstrap_ci(y_test, p_raw, brier)
        _, am_lo, am_hi = bootstrap_ci(
            y_test, p_raw, lambda y, p: aurc_overall_minority_majority(y, p)[1])
        out.update({
            "recall_pos_lo": rp_lo, "recall_pos_hi": rp_hi,
            "auroc_lo": au_lo, "auroc_hi": au_hi,
            "brier_lo": br_lo, "brier_hi": br_hi,
            "aurc_minority_lo": am_lo, "aurc_minority_hi": am_hi,
        })

    return out
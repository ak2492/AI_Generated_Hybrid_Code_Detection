"""Clean-validation threshold calibration, mirroring the CPG folder.

The CPG model stores `optimal_threshold` in each checkpoint: F1-argmax on
clean validation, prior-reweighted to the evaluation prevalence, 5-fold
cross-validated, and adopted only when it beats 0.50 by >= 5e-4. The hybrid
model used to hardcode 0.50 everywhere, so an attack comparison mixed a
calibrated CPG cut with a fixed hybrid cut.

These helpers reproduce the CPG rule exactly. They are Opt-In: the hybrid
default stays fixed 0.50 (paper Table 9 protocol). Run the 5-seed protocol
with `--threshold-mode calibrated` to get the secondary, threshold-parity
table. Calibration uses clean validation only; attacked data is never used
to pick the cut.
"""
import os

import joblib
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve

from model import HybridCodeDetector


def _f1_at(thr, y, s, prior=None):
    """F1 of the hard decision s >= thr.

    With `prior` (positive prevalence) the class-conditional TPR/FPR of the
    cut are reweighted to that prevalence; without it this is the plain F1.
    Copied from cpg-based-ai-generated-code-detection/train.py so both models
    use the same rule rather than two lookalike rules.
    """
    pred = (s >= thr).astype(np.float64)
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    if prior is not None and 0.0 < prior < 1.0:
        tpr = tp / max(1.0, float((y == 1).sum()))
        fpr = fp / max(1.0, float((y == 0).sum()))
        tp_p, fp_p, fn_p = prior * tpr, (1.0 - prior) * fpr, prior * (1.0 - tpr)
        return 2.0 * tp_p / max(1e-9, 2.0 * tp_p + fp_p + fn_p)
    return 2 * tp / max(1e-9, 2 * tp + fp + fn)


def _best_threshold(scores, labels, prior=None):
    """F1-argmax decision threshold, reweighted to `prior` when given."""
    if prior is not None and 0.0 < prior < 1.0:
        order = np.argsort(-scores, kind="mergesort")
        s, y = scores[order], labels[order]
        n_pos = max(1.0, float(y.sum()))
        n_neg = max(1.0, float(len(y) - y.sum()))
        tpr, fpr = np.cumsum(y) / n_pos, np.cumsum(1.0 - y) / n_neg
        tp_p = prior * tpr
        fp_p = (1.0 - prior) * fpr
        fn_p = prior * (1.0 - tpr)
        f1 = 2.0 * tp_p / np.maximum(1e-9, 2.0 * tp_p + fp_p + fn_p)
        return float(s[int(np.argmax(f1))])
    prec, rec, thres = precision_recall_curve(labels, scores)
    if len(thres) == 0:
        return None
    f1_curve = 2 * (prec * rec) / np.maximum(1e-6, prec + rec)
    return float(thres[int(np.argmax(f1_curve[:-1]))])


def _select_threshold(probs, labels, fallback=0.50, prior=None, n_folds=5,
                      min_gain=5e-4):
    """Pick a decision threshold, but only if calibration beats the fixed cut.

    Cross-validates the calibration rule on the val scores and adopts the
    calibrated cut only when held-out F1 beats the fixed fallback by
    `min_gain`. This is the CPG rule verbatim.
    """
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return float(fallback)
    thr_full = _best_threshold(probs, labels, prior=prior)
    if thr_full is None:
        return float(fallback)

    from sklearn.model_selection import StratifiedKFold
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    rule_scores, fixed_scores = [], []
    for tr_idx, te_idx in splitter.split(probs, labels):
        y_tr, s_tr = labels[tr_idx], probs[tr_idx]
        y_te, s_te = labels[te_idx], probs[te_idx]
        if len(np.unique(y_tr)) < 2:
            continue
        thr_fold = _best_threshold(s_tr, y_tr, prior=prior)
        if thr_fold is None:
            continue
        rule_scores.append(_f1_at(thr_fold, y_te, s_te, prior=prior))
        fixed_scores.append(_f1_at(fallback, y_te, s_te, prior=prior))
    if len(rule_scores) >= 2:
        rule_mean = float(np.mean(rule_scores))
        fixed_mean = float(np.mean(fixed_scores))
        if rule_mean > fixed_mean + min_gain:
            print(f"[i] Threshold calibration kept: held-out F1 rule={rule_mean:.4f} "
                  f"vs fixed {fallback:.2f}={fixed_mean:.4f} -> threshold {thr_full:.4f}")
            return thr_full
        print(f"[i] Threshold calibration rejected: held-out F1 rule={rule_mean:.4f} "
              f"vs fixed {fallback:.2f}={fixed_mean:.4f} -> threshold {fallback:.2f}")
        return float(fallback)
    return thr_full


def clean_val_probs(language, adversarial=False, checkpoint=None,
                    scaler_file=None, batch_size=512, device=None):
    """Return (val_probs, val_labels) using the current seed's artifacts."""
    tag = "_adv" if adversarial else ""
    ckpt = checkpoint or f"{language}{tag}_best_model.pt"
    scaler_path = scaler_file or f"{language}{tag}_scaler.pkl"
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Missing {ckpt} in {os.getcwd()}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Missing {scaler_path} in {os.getcwd()}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X_val = np.load(f"{language}_validation_X.npy")
    y_val = np.load(f"{language}_validation_y.npy")
    scaler = joblib.load(scaler_path)
    X_val = scaler.transform(X_val)

    model = HybridCodeDetector().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    probs = []
    with torch.no_grad():
        for start in range(0, len(X_val), batch_size):
            batch = torch.FloatTensor(X_val[start:start + batch_size]).to(device)
            probs.append(model(batch).cpu().numpy().flatten())
    return np.concatenate(probs), y_val


def compute_seed_threshold(language, adversarial=False, checkpoint=None,
                           scaler_file=None, fallback=0.50, prior=None,
                           batch_size=512):
    """Clean-validation cut for one seed, using the CPG calibration rule.

    `prior` should be the evaluation-set positive prevalence, exactly as the
    CPG passes `eval_prior`; pass None to skip prior reweighting.
    """
    probs, labels = clean_val_probs(language, adversarial=adversarial,
                                    checkpoint=checkpoint,
                                    scaler_file=scaler_file,
                                    batch_size=batch_size)
    return _select_threshold(probs, labels, fallback=fallback, prior=prior)


def test_prior(language):
    """Positive prevalence of the clean test split (CPG `eval_prior`)."""
    return float(np.mean(np.load(f"{language}_test_y.npy")))

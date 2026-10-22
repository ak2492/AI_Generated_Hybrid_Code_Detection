"""Train the hybrid detector and save dual checkpoints from one trajectory.

One pass tracks two checkpoints, mirroring the CPG folder:

  fixed050  best validation F1 at the fixed 0.50 cut
  bestthr   best validation F1 over every threshold

The primary path keeps the historical filename so every single-seed CLI keeps
working; bestthr is a sibling with ``_bestthr`` before the extension. A small
JSON sidecar next to each checkpoint stores the selection rule, selected epoch,
validation-best cut, and the conservative calibrated cut. The state dicts stay
raw (no wrapper), so evaluate.py / attack_utils.py load them unchanged.

Per-epoch test metrics are written to CSV when ``epoch_metrics_path`` is given.
Test metrics never influence selection, gradients, or the optimizer.
"""
import argparse
import csv
import json
import os
import random
import time

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from model import HybridCodeDetector
from threshold import _select_threshold

try:
    import psutil as _psutil
    _psutil_proc = _psutil.Process()
except ImportError:  # I fall back to 0.0 here so Kaggle runs without psutil still work.
    _psutil = None
    _psutil_proc = None


def current_rss_mb():
    """Current process RSS in MB (0.0 if psutil unavailable). Identical to CPG helper."""
    if _psutil_proc is None:
        return 0.0
    return _psutil_proc.memory_info().rss / (1024 * 1024)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _binary_metrics(probs, labels, threshold):
    probs = np.nan_to_num(np.asarray(probs, dtype=np.float64),
                          nan=0.0, posinf=1.0, neginf=0.0)
    labels = np.asarray(labels, dtype=np.float64)
    if len(labels) == 0:
        return {"Acc": 0.0, "Prec": 0.0, "Rec": 0.0, "F1": 0.0,
                "FPR": 0.0, "ROC": 0.0, "N": 0}
    preds = (probs >= float(threshold)).astype(np.int64)
    y = labels.astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    return {
        "Acc": float(accuracy_score(y, preds)),
        "Prec": float(precision_score(y, preds, zero_division=0)),
        "Rec": float(recall_score(y, preds, zero_division=0)),
        "F1": float(f1_score(y, preds, zero_division=0)),
        "FPR": float(fp / max(1, fp + tn)),
        "ROC": float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else 0.0,
        "N": int(len(y)),
    }


def _best_threshold_and_f1(scores, labels):
    """Exact legacy selection rule: max F1 over the validation PR curve."""
    scores = np.nan_to_num(np.asarray(scores, dtype=np.float64),
                           nan=0.0, posinf=1.0, neginf=0.0)
    labels = np.asarray(labels, dtype=np.float64)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return 0.50, 0.0
    prec, rec, thres = precision_recall_curve(labels, scores)
    if len(thres) == 0:
        return 0.50, 0.0
    f1_curve = 2 * (prec * rec) / np.maximum(1e-6, prec + rec)
    idx = int(np.argmax(f1_curve[:-1]))
    return float(thres[idx]), float(f1_curve[idx])


def _collect_probs(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            probs.append(model(batch_X).cpu().numpy().flatten())
            labels.append(batch_y.cpu().numpy().flatten())
    if not probs:
        return np.array([]), np.array([])
    return np.concatenate(probs), np.concatenate(labels)


EPOCH_FIELDS = [
    "epoch", "epochs", "train_loss", "epoch_seconds",
    "val_thr_best", "val_f1_best", "val_f1_fixed",
    "val_acc_fixed", "val_prec_fixed", "val_rec_fixed", "val_fpr_fixed", "val_roc_fixed",
    "val_acc_best", "val_prec_best", "val_rec_best", "val_fpr_best", "val_roc_best",
    "test_acc_fixed", "test_prec_fixed", "test_rec_fixed", "test_f1_fixed",
    "test_fpr_fixed", "test_roc_fixed",
    "test_acc_best", "test_prec_best", "test_rec_best", "test_f1_best",
    "test_fpr_best", "test_roc_best",
    "selected_fixed050", "selected_bestthr",
]


def _write_epoch_rows(path, rows):
    if not path:
        return
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=EPOCH_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in EPOCH_FIELDS})
    os.replace(tmp, path)


def _bestthr_path(path):
    stem, ext = os.path.splitext(path)
    return f"{stem}_bestthr{ext}"


def _meta_path(path):
    return path + ".json"


def _write_metadata(path, metadata):
    tmp = _meta_path(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    os.replace(tmp, _meta_path(path))


def _load_split(language, split, adversarial=False):
    tag = "_adv" if (split == "train" and adversarial) else ""
    stem = f"{language}_{split}{tag}"
    x_path, y_path = f"{stem}_X.npy", f"{stem}_y.npy"
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            f"Missing hybrid features {x_path} / {y_path}. Run "
            f"`python main.py --language {language}"
            f"{' --adversarial' if adversarial else ''}` first.")
    return np.load(x_path), np.load(y_path)


def train_model(language="python", epochs=100, batch_size=64, learning_rate=1e-5,
                adversarial=False, seed=42, threshold=0.50,
                epoch_metrics_path=None, return_info=False, save_dual=True,
                checkpoint_path=None):
    # I expose `seed` here because I want 5-seed averages (42-46) over
    # identical .npy bundles, mirroring the CPG folder's --seed protocol.
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing {language.upper()} training on {device}...")

    X_train, y_train = _load_split(language, "train", adversarial=adversarial)
    X_val, y_val = _load_split(language, "validation")
    X_test, y_test = _load_split(language, "test")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    scaler_file = f"{language}_adv_scaler.pkl" if adversarial else f"{language}_scaler.pkl"
    joblib.dump(scaler, scaler_file)

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train)),
        batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val)),
        batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test)),
        batch_size=batch_size, shuffle=False)

    model = HybridCodeDetector().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_fixed_f1, best_fixed_roc = 0.0, 0.0
    best_fixed_state, best_fixed_epoch = None, -1
    best_thr_f1, best_thr_roc = 0.0, 0.0
    best_thr_state, best_thr_epoch = None, -1
    last_epoch = -1
    epoch_rows = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    peak_ram_mb = current_rss_mb()

    for epoch in range(epochs):
        ep_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_X)

        val_probs, val_labels = _collect_probs(model, val_loader, device)
        fixed_metrics = _binary_metrics(val_probs, val_labels, 0.50)
        val_thr, val_f1_best = _best_threshold_and_f1(val_probs, val_labels)
        best_metrics = _binary_metrics(val_probs, val_labels, val_thr)

        # Full held-out test metrics every epoch. The test split is read-only
        # and does not influence selection, gradients, or the optimizer.
        test_probs, test_labels = _collect_probs(model, test_loader, device)
        test_fixed = _binary_metrics(test_probs, test_labels, 0.50)
        test_best = _binary_metrics(test_probs, test_labels, val_thr)

        peak_ram_mb = max(peak_ram_mb, current_rss_mb())
        ep_duration = time.perf_counter() - ep_start
        last_epoch = epoch
        train_loss = total_loss / max(1, len(train_loader.dataset))
        print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {train_loss:.4f} | "
              f"Val F1@0.50: {fixed_metrics['F1']:.4f} | "
              f"Val F1@best: {val_f1_best:.4f} (thr {val_thr:.4f}) | "
              f"Time: {ep_duration:.1f}s")
        print(f"  Val  @0.50 -> Acc {fixed_metrics['Acc']:.4f} | Prec {fixed_metrics['Prec']:.4f} | "
              f"Rec {fixed_metrics['Rec']:.4f} | F1 {fixed_metrics['F1']:.4f} | "
              f"FPR {fixed_metrics['FPR']:.4f} | ROC {fixed_metrics['ROC']:.4f}")
        print(f"  Test @0.50 -> Acc {test_fixed['Acc']:.4f} | Prec {test_fixed['Prec']:.4f} | "
              f"Rec {test_fixed['Rec']:.4f} | F1 {test_fixed['F1']:.4f} | "
              f"FPR {test_fixed['FPR']:.4f} | ROC {test_fixed['ROC']:.4f}")
        print(f"  Test @best -> Acc {test_best['Acc']:.4f} | Prec {test_best['Prec']:.4f} | "
              f"Rec {test_best['Rec']:.4f} | F1 {test_best['F1']:.4f} | "
              f"FPR {test_best['FPR']:.4f} | ROC {test_best['ROC']:.4f}")

        selected_fixed = fixed_metrics["F1"] > best_fixed_f1
        if selected_fixed:
            best_fixed_f1 = fixed_metrics["F1"]
            best_fixed_roc = fixed_metrics["ROC"]
            best_fixed_state = {k: v.detach().cpu().clone()
                                for k, v in model.state_dict().items()}
            best_fixed_epoch = epoch

        selected_best = val_f1_best > best_thr_f1
        if selected_best:
            best_thr_f1 = val_f1_best
            best_thr_roc = best_metrics["ROC"]
            best_thr_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            best_thr_epoch = epoch

        row = {
            "epoch": epoch + 1, "epochs": epochs,
            "train_loss": train_loss, "epoch_seconds": ep_duration,
            "val_thr_best": val_thr, "val_f1_best": val_f1_best,
            "val_f1_fixed": fixed_metrics["F1"],
            "val_acc_fixed": fixed_metrics["Acc"], "val_prec_fixed": fixed_metrics["Prec"],
            "val_rec_fixed": fixed_metrics["Rec"], "val_fpr_fixed": fixed_metrics["FPR"],
            "val_roc_fixed": fixed_metrics["ROC"],
            "val_acc_best": best_metrics["Acc"], "val_prec_best": best_metrics["Prec"],
            "val_rec_best": best_metrics["Rec"], "val_fpr_best": best_metrics["FPR"],
            "val_roc_best": best_metrics["ROC"],
            "test_acc_fixed": test_fixed["Acc"], "test_prec_fixed": test_fixed["Prec"],
            "test_rec_fixed": test_fixed["Rec"], "test_f1_fixed": test_fixed["F1"],
            "test_fpr_fixed": test_fixed["FPR"], "test_roc_fixed": test_fixed["ROC"],
            "test_acc_best": test_best["Acc"], "test_prec_best": test_best["Prec"],
            "test_rec_best": test_best["Rec"], "test_f1_best": test_best["F1"],
            "test_fpr_best": test_best["FPR"], "test_roc_best": test_best["ROC"],
            "selected_fixed050": int(selected_fixed),
            "selected_bestthr": int(selected_best),
        }
        epoch_rows.append(row)
        _write_epoch_rows(epoch_metrics_path, epoch_rows)

    # Degenerate runs still leave two usable checkpoints behind. Recompute the
    # fallback's validation metrics rather than reporting zero ROC or the
    # nominal epoch count after an early stop.
    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    fixed_is_fallback = best_fixed_state is None
    best_is_fallback = best_thr_state is None
    if fixed_is_fallback:
        best_fixed_state = last_state
        best_fixed_epoch = last_epoch
    if best_is_fallback:
        best_thr_state = last_state
        best_thr_epoch = last_epoch

    eval_prior = float(np.mean(y_test)) if len(y_test) else None

    def _calibrate(state):
        model.load_state_dict(state)
        probs, labels = _collect_probs(model, val_loader, device)
        best_thr, _ = _best_threshold_and_f1(probs, labels)
        cal = _select_threshold(probs, labels, fallback=threshold,
                                prior=eval_prior)
        return (cal, best_thr, _binary_metrics(probs, labels, 0.50),
                _binary_metrics(probs, labels, best_thr))

    cal_fixed, best_thr_fixed, fixed_val, _ = _calibrate(best_fixed_state)
    cal_best, best_thr_best, _, best_val = _calibrate(best_thr_state)
    if fixed_is_fallback:
        best_fixed_f1 = fixed_val["F1"]
        best_fixed_roc = fixed_val["ROC"]
    if best_is_fallback:
        best_thr_f1 = best_val["F1"]
        best_thr_roc = best_val["ROC"]
    prior_txt = "n/a" if eval_prior is None else f"{eval_prior:.4f}"
    print(f"[i] Fixed050 calibrated cut: {cal_fixed:.4f} "
          f"(selected epoch {best_fixed_epoch + 1}, val F1@0.50 {best_fixed_f1:.4f})")
    print(f"[i] BestThr selection cut: {best_thr_best:.4f}; calibrated cut: "
          f"{cal_best:.4f} (selected epoch {best_thr_epoch + 1}, "
          f"val F1@best {best_thr_f1:.4f}; eval prior {prior_txt})")

    tag = "_adv_best_model.pt" if adversarial else "_best_model.pt"
    fixed_path = checkpoint_path or f"{language}{tag}"
    best_path = _bestthr_path(fixed_path)

    model.load_state_dict(best_fixed_state)
    torch.save(model.state_dict(), fixed_path)
    fixed_meta = {
        "format_version": 2,
        "selection_rule": "fixed050",
        "selected_epoch": best_fixed_epoch + 1,
        "selection_threshold": 0.50,
        "best_threshold": float(best_thr_fixed),
        "calibrated_threshold": float(cal_fixed),
        "val_f1": float(best_fixed_f1),
        "val_roc": float(best_fixed_roc),
    }
    _write_metadata(fixed_path, fixed_meta)
    print(f"[+] Checkpoint preserved at: {fixed_path} (fixed050)")

    if save_dual:
        model.load_state_dict(best_thr_state)
        torch.save(model.state_dict(), best_path)
        best_meta = {
            "format_version": 2,
            "selection_rule": "bestthr",
            "selected_epoch": best_thr_epoch + 1,
            "selection_threshold": float(best_thr_best),
            "best_threshold": float(best_thr_best),
            "calibrated_threshold": float(cal_best),
            "val_f1": float(best_thr_f1),
            "val_roc": float(best_thr_roc),
        }
        _write_metadata(best_path, best_meta)
        print(f"[+] Checkpoint preserved at: {best_path} (bestthr)")

    # Leave fixed050 loaded, preserving the historical single-model behavior.
    model.load_state_dict(best_fixed_state)

    total_time = time.perf_counter() - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    peak_ram_mb = max(peak_ram_mb, current_rss_mb())

    print(f"Training complete. Best F1: {best_fixed_f1:.4f}")
    print(f"\n[DIAGNOSTICS - {'MODEL ADVERSARIAL' if adversarial else 'MODEL CLEAN'}]")
    print(f"Total Training Duration : {total_time / 60:.2f} minutes")
    print(f"Peak VRAM Consumption   : {peak_vram:.2f} MB")
    print(f"Peak CPU RAM Consumption: {peak_ram_mb:.2f} MB")

    info = {
        "fixed050": {
            "path": fixed_path, "epoch": best_fixed_epoch + 1,
            "selected_epoch": best_fixed_epoch + 1,
            "selection_threshold": 0.50,
            "best_threshold": float(best_thr_fixed),
            "calibrated_threshold": float(cal_fixed),
            "f1": float(best_fixed_f1), "roc": float(best_fixed_roc),
            "val_f1": float(best_fixed_f1), "val_roc": float(best_fixed_roc),
        },
        "bestthr": {
            "path": best_path, "epoch": best_thr_epoch + 1,
            "selected_epoch": best_thr_epoch + 1,
            "selection_threshold": float(best_thr_best),
            "best_threshold": float(best_thr_best),
            "calibrated_threshold": float(cal_best),
            "f1": float(best_thr_f1), "roc": float(best_thr_roc),
            "val_f1": float(best_thr_f1), "val_roc": float(best_thr_roc),
        },
    }
    if return_info:
        return info
    # I return the fixed050 validation F1 so the old 5-seed runner still works.
    return best_fixed_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="python", choices=["python", "java", "cpp"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Fallback fixed cut used by calibration")
    parser.add_argument("--adversarial", action="store_true",
                        help="Load the adversarial augmented training set instead of the clean baseline")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global RNG seed for init/shuffle (vary 42-46 for 5-seed average; data splits stay fixed)")
    parser.add_argument("--epoch-metrics-path", type=str, default=None,
                        help="Per-epoch test metrics CSV (default: epoch_metrics_{lang}_{variant}.csv)")
    args = parser.parse_args()

    if args.batch_size is None:
        args.batch_size = 64 if args.language == "python" else (32 if args.language == "java" else 16)

    variant = "adv" if args.adversarial else "clean"
    metrics_path = args.epoch_metrics_path or f"epoch_metrics_{args.language}_{variant}.csv"
    info = train_model(
        language=args.language, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, adversarial=args.adversarial,
        seed=args.seed, threshold=args.threshold,
        epoch_metrics_path=metrics_path, return_info=True)
    print(f"[i] Dual checkpoints: fixed050={info['fixed050']['path']} "
          f"(epoch {info['fixed050']['epoch']}), "
          f"bestthr={info['bestthr']['path']} "
          f"(epoch {info['bestthr']['epoch']}, thr {info['bestthr']['selection_threshold']:.4f})")
    print(f"[i] Per-epoch test metrics -> {metrics_path}")

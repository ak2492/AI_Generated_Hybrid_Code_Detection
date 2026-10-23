import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import joblib
import argparse
import random
import time
import json
import os
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from model import HybridCodeDetector
from threshold import compute_seed_threshold, test_prior

try:
    import psutil as _psutil
    _psutil_proc = _psutil.Process()
except ImportError:
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


def ensure_data(language, adversarial=False, rebuild=False):
    """Ensure .npy feature splits exist; extract via main.py if missing."""
    import subprocess
    import sys
    clean_splits = ("validation", "test") if adversarial else ("train", "validation", "test")
    clean_required = [f"{language}_{split}_{kind}.npy"
                      for split in clean_splits for kind in ("X", "y")]
    adv_required = [f"{language}_train_adv_X.npy", f"{language}_train_adv_y.npy"]
    required = clean_required + (adv_required if adversarial else [])

    missing_clean = [p for p in clean_required if not os.path.exists(p)]
    if missing_clean or rebuild:
        print(f"[data] Missing clean feature file(s); extracting via main.py --language {language}...")
        subprocess.run([sys.executable, "main.py", "--language", language], check=True)
    if adversarial:
        missing_adv = [p for p in adv_required if not os.path.exists(p)]
        if missing_adv or rebuild:
            print(f"[data] Missing adversarial feature file(s); extracting via main.py --language {language} --adversarial...")
            subprocess.run([sys.executable, "main.py", "--language", language, "--adversarial"], check=True)


def train_model(language="python", epochs=100, batch_size=64, learning_rate=1e-5, adversarial=False, seed=42):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing {language.upper()} training on {device}...")

    ensure_data(language, adversarial=adversarial)

    train_x_file = f"{language}_train_adv_X.npy" if adversarial else f"{language}_train_X.npy"
    train_y_file = f"{language}_train_adv_y.npy" if adversarial else f"{language}_train_y.npy"

    print(f"Loading training data from: {train_x_file}")
    X_train = np.load(train_x_file)
    y_train = np.load(train_y_file)
    X_val = np.load(f"{language}_validation_X.npy")
    y_val = np.load(f"{language}_validation_y.npy")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    scaler_file = f"{language}_adv_scaler.pkl" if adversarial else f"{language}_scaler.pkl"
    joblib.dump(scaler, scaler_file)

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val)), batch_size=batch_size, shuffle=False)

    model = HybridCodeDetector().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_f1 = 0.0
    best_val_roc = 0.0
    best_weights = None

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    peak_ram_mb = current_rss_mb()

    for epoch in range(epochs):
        ep_start = time.perf_counter()
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_probs, val_targets = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                val_probs.extend(model(batch_X).cpu().numpy().flatten())
                val_targets.extend(batch_y.numpy().flatten())

        val_probs = np.array(val_probs, dtype=np.float64)
        val_targets = np.array(val_targets, dtype=np.float64)
        prec_v, rec_v, thres_v = precision_recall_curve(val_targets, val_probs)
        f1_curve = 2 * (prec_v * rec_v) / np.maximum(1e-6, prec_v + rec_v)
        current_f1 = float(np.max(f1_curve[:-1])) if len(thres_v) > 0 else 0.0
        val_roc = float(roc_auc_score(val_targets, val_probs)) if len(np.unique(val_targets)) > 1 else 0.0

        peak_ram_mb = max(peak_ram_mb, current_rss_mb())
        ep_duration = time.perf_counter() - ep_start

        print(f"Epoch {epoch+1}/{epochs} | Val F1@best: {current_f1:.4f} | Val ROC: {val_roc:.4f} | Time: {ep_duration:.1f}s")

        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            best_val_roc = val_roc
            best_weights = copy.deepcopy(model.state_dict())

    if best_weights is not None:
        model.load_state_dict(best_weights)
    else:
        best_weights = copy.deepcopy(model.state_dict())

    model_file = f"{language}_adv_best_model.pt" if adversarial else f"{language}_best_model.pt"
    torch.save(model.state_dict(), model_file)

    eval_prior = test_prior(language)
    cal_threshold = compute_seed_threshold(language, adversarial=adversarial, fallback=0.50, prior=eval_prior)
    prior_txt = "n/a" if eval_prior is None else f"{eval_prior:.4f}"
    print(f"[i] Best-threshold: {cal_threshold:.4f} (eval prior {prior_txt}; val F1 {best_val_f1:.4f})")

    meta = {
        "optimal_threshold": float(cal_threshold),
        "val_f1": float(best_val_f1),
        "val_roc": float(best_val_roc),
    }
    with open(f"{model_file}.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    total_time = time.perf_counter() - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    peak_ram_mb = max(peak_ram_mb, current_rss_mb())

    print(f"Training complete. Best Val F1: {best_val_f1:.4f} | Threshold: {cal_threshold:.4f}")
    print(f"\n[DIAGNOSTICS - {'MODEL ADVERSARIAL' if adversarial else 'MODEL CLEAN'}]")
    print(f"Total Training Duration : {total_time / 60:.2f} minutes")
    print(f"Peak VRAM Consumption   : {peak_vram:.2f} MB")
    print(f"Peak CPU RAM Consumption: {peak_ram_mb:.2f} MB")
    return best_val_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="python", choices=["python", "java", "cpp"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--adversarial", action="store_true", help="Load the adversarial augmented training set instead of the clean baseline")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global RNG seed for init/shuffle (vary 42-46 for 5-seed average; data splits stay fixed)")
    args = parser.parse_args()

    if args.batch_size is None:
        args.batch_size = 64 if args.language == "python" else (32 if args.language == "java" else 16)

    train_model(language=args.language, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, adversarial=args.adversarial, seed=args.seed)

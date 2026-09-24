import torch
import numpy as np
import joblib
import argparse
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from model import HybridCodeDetector

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


def evaluate_model(language="python", batch_size=64, adversarial=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating {language.upper()} on {device}...")
    
    X_test = np.load(f"{language}_test_X.npy")
    y_test = np.load(f"{language}_test_y.npy")
    scaler_file = f"{language}_adv_scaler.pkl" if adversarial else f"{language}_scaler.pkl"
    scaler = joblib.load(scaler_file)
    X_test_scaled = scaler.transform(X_test)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test)), batch_size=batch_size, shuffle=False)
    
    model = HybridCodeDetector().to(device)
    # Fixed map_location for cross-hardware evaluation
    model_file = f"{language}_adv_best_model.pt" if adversarial else f"{language}_best_model.pt"
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    
    all_preds, all_targets, all_probs = [], [], []

    # I time the full inference pass like the CPG cost helper so latency and
    # throughput mean the same thing in both folders.
    rss_before = current_rss_mb()
    t_start = time.perf_counter()
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            probs = model(batch_X)
            preds = (probs >= 0.5).float()

            # Squeeze guarantees flat 1D lists even if batch sizes jitter
            all_probs.extend(probs.cpu().numpy().flatten())
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(batch_y.numpy().flatten())

    inf_duration = time.perf_counter() - t_start
    peak_ram_mb = max(rss_before, current_rss_mb())
    n_samples = len(all_targets)
    latency_ms = (inf_duration / n_samples) * 1000.0 if n_samples else 0.0
    throughput = n_samples / max(1e-6, inf_duration)

    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    roc = roc_auc_score(all_targets, all_probs) if len(np.unique(all_targets)) > 1 else 0.0

    # labels=[0,1] prevents crash if the test set only has 1 class
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds, labels=[0,1]).ravel()
    fpr = fp / max(1, fp + tn)

    print("\n--- FINAL TEST METRICS ---")
    print(f"Accuracy:  {acc:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {roc:.4f}")
    print(f"Precision: {prec:.4f} | Recall:   {rec:.4f} | FPR:     {fpr:.4f}")
    print(f"  Cost    -> Latency: {latency_ms:.2f} ms/sample | Throughput: {throughput:.2f} samples/sec | PeakRAM: {peak_ram_mb:.2f} MB")
    print(f"  Test time: {inf_duration:.2f}s over {n_samples} samples")

    return {
        'Acc': acc, 'Prec': prec, 'Rec': rec,
        'F1': f1, 'ROC': roc, 'FPR': fpr,
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp, 'N': n_samples,
        'Latency_ms': latency_ms, 'Throughput': throughput,
        'PeakRAM_MB': peak_ram_mb,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="python", choices=["python", "java", "cpp"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--adversarial", action="store_true")
    args = parser.parse_args()
    
    if args.batch_size is None:
        args.batch_size = 64 if args.language == "python" else (32 if args.language == "java" else 16)
            
    evaluate_model(language=args.language, batch_size=args.batch_size, adversarial=args.adversarial)

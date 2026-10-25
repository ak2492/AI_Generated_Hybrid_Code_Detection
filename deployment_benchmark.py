"""Raw-code end-to-end deployment benchmark for the hybrid detector.

Unlike evaluate.py, which scores precomputed 813-d features, this script
starts from raw test-set source files and times the whole deployment path:
tree-sitter/CodeT5+ semantic extraction, CodeBERT statistical extraction,
AST authorship extraction, scaling, and model inference.

Two views are reported:

  * per_file : files are processed one at a time (batch size 1), matching a
               request-scoped deployment. Reports mean/median/p95 latency and
               source tokens/sec.
  * batched  : the same work is done in batches, reporting end-to-end
               files/sec and tokens/sec.

The token denominator is the shared tree-sitter leaf-token definition in
token_metrics.py, so numbers are comparable with the CPG and legacy folders.

Example:
  python deployment_benchmark.py --language python --seed 42 --max-files 200
"""
import argparse
import csv
import gc
import json
import os
import random
import time

import joblib
import numpy as np
import torch

from attack_utils import MAX_CODE_SIZE_TRANSFORMER, get_auth_parser, safe_extract_authorship
from data_loader import load_code_data
from model import HybridCodeDetector
from semantic_extractor import SemanticExtractor
from statistical_extractor import StatisticalExtractor
from token_metrics import count_tokens_batch

FEATURE_DIM = 813


def _default_batch(language):
    return 64 if language == "python" else (32 if language == "java" else 16)


def _percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else None


def _summary(values):
    if not values:
        return {"mean_ms": None, "median_ms": None, "p95_ms": None, "total_s": 0.0}
    vals = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(vals.mean()),
        "median_ms": float(np.median(vals)),
        "p95_ms": float(np.percentile(vals, 95)),
        "total_s": float(vals.sum() / 1000.0),
    }


def _artifact_paths(language, seed, adversarial):
    tag = "_adv" if adversarial else ""
    model_file = f"{language}{tag}_best_model.pt"
    scaler_file = f"{language}{tag}_scaler.pkl"
    base, ext = os.path.splitext(model_file)
    seed_model = f"{base}_seed{seed}{ext}"
    sbase, sext = os.path.splitext(scaler_file)
    seed_scaler = f"{sbase}_seed{seed}{sext}"
    if os.path.exists(seed_model):
        model_file = seed_model
    if os.path.exists(seed_scaler):
        scaler_file = seed_scaler
    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"Missing {model_file}. Train the model or restore the archived per-seed files first.")
    if not os.path.exists(scaler_file):
        raise FileNotFoundError(f"Missing {scaler_file}. Train the model first.")
    return model_file, scaler_file


def _load_threshold(model_file, explicit):
    if explicit is not None:
        return float(explicit)
    base, ext = os.path.splitext(model_file)
    canonical = (base.split("_seed")[0] + ext) if "_seed" in base else model_file
    for path in (f"{model_file}.json", f"{canonical}.json"):
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    return float(json.load(fh).get("optimal_threshold", 0.50))
            except (OSError, ValueError, TypeError):
                pass
    return 0.50


def _extract_features(codes, sem, stat, auth_fn, batch_size):
    clipped = [str(c)[:MAX_CODE_SIZE_TRANSFORMER] for c in codes]
    sem_parts = []
    for i in range(0, len(clipped), batch_size):
        sem_parts.append(sem.extract_batch(clipped[i:i + batch_size]))
    stat_parts = []
    for i in range(0, len(clipped), batch_size):
        stat_parts.append(stat.extract_batch(clipped[i:i + batch_size]))
    sem_feat = np.vstack(sem_parts)
    stat_feat = np.vstack(stat_parts)
    auth_feat = np.asarray([safe_extract_authorship(c, auth_fn) for c in clipped])
    return np.hstack((sem_feat, stat_feat, auth_feat))


def _infer(model, X_scaled, device, batch_size=None):
    if batch_size is None:
        batch_size = len(X_scaled)
    probs = []
    with torch.no_grad():
        for i in range(0, len(X_scaled), max(1, batch_size)):
            chunk = torch.from_numpy(X_scaled[i:i + batch_size]).float().to(device)
            probs.extend(model(chunk).cpu().numpy().flatten().tolist())
    return probs


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _run_hot_once(code, language, sem, stat, auth_fn, scaler, model, device):
    """One full raw-code request; returns preprocessing/inference seconds."""
    _sync(device)
    t0 = time.perf_counter()
    feats = _extract_features([code], sem, stat, auth_fn, 1)
    scaled = scaler.transform(feats)
    x = torch.from_numpy(scaled).float().to(device)
    _sync(device)
    t1 = time.perf_counter()
    with torch.no_grad():
        _ = model(x)
    _sync(device)
    t2 = time.perf_counter()
    return t1 - t0, t2 - t1


def run_benchmark(language="python", seed=42, max_files=200, batch_size=None,
                  warmup=5, adversarial=False, threshold=None, output=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = batch_size or _default_batch(language)
    print(f"Hybrid raw-code deployment benchmark | {language.upper()} | {device}")

    codes, _labels = load_code_data(language=language, split="test")
    if max_files and max_files < len(codes):
        indices = sorted(random.Random(seed).sample(range(len(codes)), int(max_files)))
        codes = [codes[i] for i in indices]
    if not codes:
        raise RuntimeError(f"No test files loaded for {language}.")
    n_files = len(codes)
    token_counts = np.asarray(count_tokens_batch(codes, language), dtype=np.int64)
    total_tokens = int(token_counts.sum())
    print(f"Files: {n_files} | source tokens: {total_tokens} | "
          f"tokens/file: {total_tokens / max(1, n_files):.1f}")

    model_file, scaler_file = _artifact_paths(language, seed, adversarial)
    threshold = _load_threshold(model_file, threshold)
    scaler = joblib.load(scaler_file)
    model = HybridCodeDetector().to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    sem = SemanticExtractor(device)
    stat = StatisticalExtractor(device)
    auth_fn = get_auth_parser(language)
    print(f"Checkpoint: {model_file} | scaler: {scaler_file} | threshold: {threshold:.4f}")

    # Warm every lazy CUDA kernel / tokenizer cache before collecting timings.
    for code in codes[:max(0, min(int(warmup), n_files))]:
        _run_hot_once(code, language, sem, stat, auth_fn, scaler, model, device)
    _sync(device)

    per_file_rows = []
    preprocess_ms, inference_ms, total_ms = [], [], []
    for idx, code in enumerate(codes):
        pre_s, inf_s = _run_hot_once(code, language, sem, stat, auth_fn, scaler, model, device)
        pre_ms, inf_ms, tot_ms = pre_s * 1000.0, inf_s * 1000.0, (pre_s + inf_s) * 1000.0
        preprocess_ms.append(pre_ms)
        inference_ms.append(inf_ms)
        total_ms.append(tot_ms)
        per_file_rows.append({
            "index": idx,
            "tokens": int(token_counts[idx]),
            "preprocess_ms": round(pre_ms, 6),
            "inference_ms": round(inf_ms, 6),
            "total_ms": round(tot_ms, 6),
        })
    per_file_total_s = sum(total_ms) / 1000.0

    # Batched view: preprocessing includes all feature extractors; inference
    # uses the trained chunk size.
    _sync(device)
    t0 = time.perf_counter()
    features = _extract_features(codes, sem, stat, auth_fn, batch_size)
    scaled = scaler.transform(features)
    _sync(device)
    t_pre = time.perf_counter() - t0
    t1 = time.perf_counter()
    _infer(model, scaled, device, batch_size)
    _sync(device)
    t_inf = time.perf_counter() - t1
    batched_total_s = t_pre + t_inf

    payload = {
        "model": "hybrid",
        "language": language,
        "seed": int(seed),
        "adversarial": bool(adversarial),
        "device": str(device),
        "checkpoint": model_file,
        "scaler": scaler_file,
        "threshold": float(threshold),
        "feature_dim": FEATURE_DIM,
        "n_files": n_files,
        "total_tokens": total_tokens,
        "tokens_per_file": total_tokens / max(1, n_files),
        "warmup_files": int(min(max(0, warmup), n_files)),
        "per_file": {
            "batch_size": 1,
            "preprocess": _summary(preprocess_ms),
            "inference": _summary(inference_ms),
            "end_to_end": _summary(total_ms),
            "files_per_sec": n_files / max(1e-9, per_file_total_s),
            "tokens_per_sec": total_tokens / max(1e-9, per_file_total_s),
            "ms_per_token": 1000.0 * per_file_total_s / total_tokens if total_tokens else None,
        },
        "batched": {
            "batch_size": int(batch_size),
            "preprocess_s": float(t_pre),
            "inference_s": float(t_inf),
            "end_to_end_s": float(batched_total_s),
            "files_per_sec": n_files / max(1e-9, batched_total_s),
            "tokens_per_sec": total_tokens / max(1e-9, batched_total_s),
            "ms_per_file": 1000.0 * batched_total_s / n_files,
            "ms_per_token": 1000.0 * batched_total_s / total_tokens if total_tokens else None,
        },
    }
    out_json = output or f"deployment_hybrid_{language}_seed{seed}.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    csv_path = os.path.splitext(out_json)[0] + "_per_file.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["index", "tokens", "preprocess_ms",
                                                "inference_ms", "total_ms"])
        writer.writeheader()
        writer.writerows(per_file_rows)

    print("\nPer-file (batch=1):")
    print(f"  end-to-end mean/median/p95: {payload['per_file']['end_to_end']['mean_ms']:.2f} / "
          f"{payload['per_file']['end_to_end']['median_ms']:.2f} / "
          f"{payload['per_file']['end_to_end']['p95_ms']:.2f} ms")
    print(f"  files/sec: {payload['per_file']['files_per_sec']:.2f} | "
          f"tokens/sec: {payload['per_file']['tokens_per_sec']:.2f} | "
          f"ms/token: {payload['per_file']['ms_per_token']:.4f}")
    print("\nBatched end-to-end:")
    print(f"  files/sec: {payload['batched']['files_per_sec']:.2f} | "
          f"tokens/sec: {payload['batched']['tokens_per_sec']:.2f} | "
          f"ms/file: {payload['batched']['ms_per_file']:.2f} | "
          f"ms/token: {payload['batched']['ms_per_token']:.4f}")
    print(f"\nWrote {out_json} and {csv_path}")

    del sem, stat, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid raw-code deployment benchmark")
    parser.add_argument("--language", default="python", choices=["python", "java", "cpp"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-files", type=int, default=200,
                        help="Deterministic test-set sample size (0 = all files)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--adversarial", action="store_true")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run_benchmark(language=args.language, seed=args.seed, max_files=args.max_files,
                  batch_size=args.batch_size, warmup=args.warmup,
                  adversarial=args.adversarial, threshold=args.threshold,
                  output=args.output)

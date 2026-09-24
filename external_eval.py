"""External OOD evaluations for the Hybrid detector — dataset logic mirrors
CPG external_eval.py exactly (same sources, filters, wrappers, caps, seeds).

Per-language suites (identical to CPG):
  python: SemEval A/B (filter in ['python','py'], no wrapper) + HMCorp python
  java:   SemEval A/B (filter == 'java', DummyWrapper class) + HMCorp java +
          GPTSniffer (0_* = AI(1), 1_* = Human(0), DummyWrapper, 3000/class cap)
  cpp:    SemEval A/B (filter in ['cpp','c++','cxx','cc'], dummy_wrapper);
          HMCorp/GPTSniffer unavailable (same as CPG/notebooks).

Only the feature->model stage is Hybrid: CodeT5+ semantic (768-d) +
CodeBERT statistical (7-d) + AST authorship (38-d) = 813-d, scaled with the
run's {language}[_adv]_scaler.pkl and scored by HybridCodeDetector at
threshold 0.50, with identical Latency_ms/Throughput/PeakRAM_MB cost keys.

One model per run like every Hybrid CLI (default clean, --adversarial = adv):
  python external_eval.py --language python --suite all
  python external_eval.py --language java --suite gptsniffer --adversarial
"""
import argparse
import gc
import glob
import json
import os
import random
import subprocess
import time

import joblib
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from attack_utils import (MAX_CODE_SIZE_TRANSFORMER, current_rss_mb,
                          get_auth_parser, safe_extract_authorship, set_seed)
from model import HybridCodeDetector
from semantic_extractor import SemanticExtractor
from statistical_extractor import StatisticalExtractor

MAX_SEMEVAL_PER_CLASS = 3000  # I match CPG MAX_SEMEVAL_SAMPLES_PER_CLASS.
EXT_SEED = 42  # I use 42 here like CPG external_eval so both folders sample identical external rows.


def _default_batch(language):
    return 64 if language == "python" else (32 if language == "java" else 16)


def _evaluate_external(samples, language, sem_extractor, stat_extractor,
                       batch_size, threshold, adversarial, tag):
    """Extract Hybrid features for raw {code,label} rows and score one model."""
    codes = [s["code"] for s in samples]
    labels = np.array([s["label"] for s in samples])
    n_human = int((labels == 0).sum())
    n_machine = int((labels == 1).sum())
    print(f"Isolated {len(samples)} external samples ({n_human} Human, {n_machine} AI).")

    all_sem = []
    for i in tqdm(range(0, len(codes), 64), desc="CodeT5+ (external)", unit="batch", leave=True):
        batch = [c[:MAX_CODE_SIZE_TRANSFORMER] for c in codes[i:i + 64]]
        all_sem.append(sem_extractor.extract_batch(batch))
    all_stat = []
    for i in tqdm(range(0, len(codes), 32), desc="CodeBERT (external)", unit="batch", leave=True):
        batch = [c[:MAX_CODE_SIZE_TRANSFORMER] for c in codes[i:i + 32]]
        all_stat.append(stat_extractor.extract_batch(batch))
    auth_parser = get_auth_parser(language)
    all_auth = np.array([safe_extract_authorship(c, auth_parser)
                         for c in tqdm(codes, desc="AST (external)", leave=True)])
    X = np.hstack((np.vstack(all_sem), np.vstack(all_stat), all_auth))

    scaler_file = f"{language}_adv_scaler.pkl" if adversarial else f"{language}_scaler.pkl"
    if not os.path.exists(scaler_file):
        raise FileNotFoundError(
            f"Missing {scaler_file} in working directory. Run main.py/train.py first in the same folder.")
    scaler = joblib.load(scaler_file)
    X_scaled = scaler.transform(X)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HybridCodeDetector().to(device)
    model_file = f"{language}_adv_best_model.pt" if adversarial else f"{language}_best_model.pt"
    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"Missing {model_file} in working directory. Run main.py/train.py first in the same folder.")
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    loader = DataLoader(TensorDataset(torch.FloatTensor(X_scaled),
                                      torch.FloatTensor(labels)),
                        batch_size=batch_size, shuffle=False)

    all_preds, all_targets, all_probs = [], [], []
    rss_before = current_rss_mb()
    t_start = time.perf_counter()
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            probs = model(batch_X)
            preds = (probs >= threshold).float()
            all_probs.extend(probs.cpu().numpy().flatten())
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(batch_y.numpy().flatten())
    inf_duration = time.perf_counter() - t_start
    peak_ram_mb = max(rss_before, current_rss_mb())
    n = len(all_targets)
    latency_ms = (inf_duration / n) * 1000.0 if n else 0.0
    throughput = n / max(1e-6, inf_duration)

    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    roc = roc_auc_score(all_targets, all_probs) if len(np.unique(all_targets)) > 1 else 0.0
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds, labels=[0, 1]).ravel()
    fpr = fp / max(1, fp + tn)

    print(f"\n--- {tag} ---")
    print(f"Accuracy:  {acc:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {roc:.4f}")
    print(f"Precision: {prec:.4f} | Recall:   {rec:.4f} | FPR:     {fpr:.4f}")
    print(f"Confusion Matrix -> TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")
    print(f"  Cost    -> Latency: {latency_ms:.2f} ms/sample | Throughput: {throughput:.2f} samples/sec | PeakRAM: {peak_ram_mb:.2f} MB")

    del loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"Acc": acc, "Prec": prec, "Rec": rec, "F1": f1, "ROC": roc,
            "FPR": fpr, "TN": tn, "FP": fp, "FN": fn, "TP": tp, "N": n,
            "Latency_ms": latency_ms, "Throughput": throughput,
            "PeakRAM_MB": peak_ram_mb}


def _loaders(device):
    return SemanticExtractor(device), StatisticalExtractor(device)


def _model_tag(adversarial):
    return "Model 2 (Adversarial)" if adversarial else "Model 1 (Clean Baseline)"


def run_external_semeval_python(subtask_name, is_multiclass, sem_extractor,
                                stat_extractor, batch_size=None,
                                threshold=0.50, adversarial=False):
    from datasets import load_dataset
    batch_size = batch_size or _default_batch("python")
    print("\n" + "=" * 85)
    print(f"EXTERNAL EVALUATION: SemEval-2026 Task 13 Python (Subtask {subtask_name})")
    print("=" * 85)
    raw_ds = load_dataset("DaniilOr/SemEval-2026-Task13", subtask_name, split="validation")
    filtered_ds = raw_ds.filter(lambda x: str(x.get('language', '')).strip().lower() in ['python', 'py'])
    human_samples, ai_samples = [], []
    for row in filtered_ds:
        code_text = row.get('code') or row.get('text') or ''
        if not code_text.strip():
            continue
        raw_label = int(row.get('label', 0))
        binary_label = 1 if (raw_label > 0 if is_multiclass else raw_label == 1) else 0
        if binary_label == 0:
            human_samples.append({'code': code_text, 'label': binary_label})
        else:
            ai_samples.append({'code': code_text, 'label': binary_label})
    n_h = min(len(human_samples), MAX_SEMEVAL_PER_CLASS)
    n_a = min(len(ai_samples), MAX_SEMEVAL_PER_CLASS)
    if n_h == 0 or n_a == 0:
        print(f"[!] Insufficient Python samples: {n_h} Human, {n_a} AI. Skipping.")
        return None
    balanced = human_samples[:n_h] + ai_samples[:n_a]
    random.Random(EXT_SEED).shuffle(balanced)
    res = _evaluate_external(balanced, "python", sem_extractor, stat_extractor,
                             batch_size, threshold, adversarial,
                             f"SEMEVAL {subtask_name} PYTHON [{_model_tag(adversarial)}]")
    del raw_ds, filtered_ds, human_samples, ai_samples, balanced
    gc.collect()
    return res


def run_external_semeval_java(subtask_name, is_multiclass, sem_extractor,
                              stat_extractor, batch_size=None,
                              threshold=0.50, adversarial=False):
    from datasets import load_dataset
    batch_size = batch_size or _default_batch("java")
    print("\n" + "=" * 85)
    print(f"EXTERNAL EVALUATION: SemEval-2026 Task 13 (Subtask {subtask_name})")
    print("=" * 85)
    raw_ds = load_dataset("DaniilOr/SemEval-2026-Task13", subtask_name, split="validation")
    filtered_ds = raw_ds.filter(lambda x: str(x.get('language', '')).strip().lower() == 'java')
    human_samples, ai_samples = [], []
    for row in filtered_ds:
        code_text = row.get('code') or row.get('text') or ''
        trimmed = code_text.strip()
        if not trimmed:
            continue
        # I wrap bare functions like CPG external_eval so tree-sitter parses.
        if not ("class " in trimmed or "interface " in trimmed or "enum " in trimmed):
            code_text = f"public class DummyWrapper {{\n{code_text}\n}}"
        raw_label = int(row.get('label', 0))
        binary_label = 1 if (raw_label > 0 if is_multiclass else raw_label == 1) else 0
        if binary_label == 0:
            human_samples.append({'code': code_text, 'label': binary_label})
        else:
            ai_samples.append({'code': code_text, 'label': binary_label})
    n_h = min(len(human_samples), MAX_SEMEVAL_PER_CLASS)
    n_a = min(len(ai_samples), MAX_SEMEVAL_PER_CLASS)
    balanced = human_samples[:n_h] + ai_samples[:n_a]
    random.Random(EXT_SEED).shuffle(balanced)
    res = _evaluate_external(balanced, "java", sem_extractor, stat_extractor,
                             batch_size, threshold, adversarial,
                             f"SEMEVAL {subtask_name} JAVA [{_model_tag(adversarial)}]")
    del raw_ds, filtered_ds, human_samples, ai_samples, balanced
    gc.collect()
    return res


def run_external_semeval_cpp(subtask_name, is_multiclass, sem_extractor,
                             stat_extractor, batch_size=None,
                             threshold=0.50, adversarial=False):
    from datasets import load_dataset
    batch_size = batch_size or _default_batch("cpp")
    print("\n" + "=" * 85)
    print(f"EXTERNAL EVALUATION: SemEval-2026 Task 13 C++ (Subtask {subtask_name})")
    print("=" * 85)
    raw_ds = load_dataset("DaniilOr/SemEval-2026-Task13", subtask_name, split="validation")
    filtered_ds = raw_ds.filter(lambda x: str(x.get('language', '')).strip().lower() in ['cpp', 'c++', 'cxx', 'cc'])
    human_samples, ai_samples = [], []
    for row in filtered_ds:
        code_text = row.get('code') or row.get('text') or ''
        trimmed = code_text.strip()
        if not trimmed:
            continue
        # I wrap free statements like CPG external_eval so tree-sitter parses.
        has_func = any(k in trimmed for k in ['main(', 'void ', 'int ', 'double ', 'float ', 'bool ', 'char ', 'class ', 'struct ', 'template'])
        if not has_func:
            code_text = f"#include <iostream>\nusing namespace std;\nvoid dummy_wrapper() {{\n{code_text}\n}}"
        raw_label = int(row.get('label', 0))
        binary_label = 1 if (raw_label > 0 if is_multiclass else raw_label == 1) else 0
        if binary_label == 0:
            human_samples.append({'code': code_text, 'label': binary_label})
        else:
            ai_samples.append({'code': code_text, 'label': binary_label})
    n_h = min(len(human_samples), MAX_SEMEVAL_PER_CLASS)
    n_a = min(len(ai_samples), MAX_SEMEVAL_PER_CLASS)
    if n_h == 0 or n_a == 0:
        print(f"[!] Insufficient C++ samples extracted: {n_h} Human, {n_a} AI. Skipping.")
        return None
    balanced = human_samples[:n_h] + ai_samples[:n_a]
    random.Random(EXT_SEED).shuffle(balanced)
    res = _evaluate_external(balanced, "cpp", sem_extractor, stat_extractor,
                             batch_size, threshold, adversarial,
                             f"SEMEVAL {subtask_name} C++ [{_model_tag(adversarial)}]")
    del raw_ds, filtered_ds, human_samples, ai_samples, balanced
    gc.collect()
    return res


def _read_hmcorp(file_path, language, cap=2000):
    human_samples, ai_samples = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(human_samples) >= cap and len(ai_samples) >= cap:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for code_text, label in [(str(row.get('human_code', '')).strip(), 0),
                                    (str(row.get('chatgpt_code', '')).strip(), 1)]:
                if not code_text or len(code_text) < 10:
                    continue
                # I wrap Java like CPG external_eval so tree-sitter parses.
                if language == "java" and not ("class " in code_text or "interface " in code_text or "enum " in code_text):
                    code_text = f"public class DummyWrapper {{\n{code_text}\n}}"
                if label == 0 and len(human_samples) < cap:
                    human_samples.append({'code': code_text, 'label': label})
                elif label == 1 and len(ai_samples) < cap:
                    ai_samples.append({'code': code_text, 'label': label})
    balanced = human_samples + ai_samples
    random.Random(EXT_SEED).shuffle(balanced)
    return balanced


def evaluate_hmcorp_python(sem_extractor, stat_extractor, batch_size=None,
                           threshold=0.50, adversarial=False):
    from huggingface_hub import hf_hub_download
    batch_size = batch_size or _default_batch("python")
    print("\n" + "=" * 85)
    print("EXTERNAL OOD EVALUATION: HMCorp Dataset (Python)")
    print("=" * 85)
    try:
        file_path = hf_hub_download(repo_id="OSS-forge/HumanVsAICode",
                                    filename="python_dataset.jsonl", repo_type="dataset")
    except Exception as e:
        print(f"[!] Failed to download HMCorp Python dataset: {e}")
        return None
    balanced = _read_hmcorp(file_path, "python")
    return _evaluate_external(balanced, "python", sem_extractor, stat_extractor,
                              batch_size, threshold, adversarial,
                              f"HMCorp PYTHON [{_model_tag(adversarial)}]")


def evaluate_hmcorp_java(sem_extractor, stat_extractor, batch_size=None,
                         threshold=0.50, adversarial=False):
    from huggingface_hub import hf_hub_download
    batch_size = batch_size or _default_batch("java")
    print("\n" + "=" * 85)
    print("EXTERNAL OOD EVALUATION: HMCorp Dataset (Java)")
    print("=" * 85)
    try:
        file_path = hf_hub_download(repo_id="OSS-forge/HumanVsAICode",
                                    filename="java_dataset.jsonl", repo_type="dataset")
    except Exception as e:
        print(f"[!] Failed to download HMCorp Java dataset: {e}")
        return None
    balanced = _read_hmcorp(file_path, "java")
    return _evaluate_external(balanced, "java", sem_extractor, stat_extractor,
                              batch_size, threshold, adversarial,
                              f"HMCorp JAVA [{_model_tag(adversarial)}]")


def evaluate_gptsniffer(sem_extractor, stat_extractor, batch_size=None,
                        threshold=0.50, adversarial=False):
    batch_size = batch_size or _default_batch("java")
    print("\n" + "=" * 85)
    print("EXTERNAL 2023-ERA EVALUATION: GPTSniffer Dataset (Java)")
    print("=" * 85)
    if not os.path.exists("GPTSniffer"):
        print("Cloning GPTSniffer repository...")
        subprocess.run(["git", "clone", "https://huggingface.co/datasets/mahirlabibdihan/GPTSniffer"], check=True)
    human_samples, ai_samples = [], []
    files = glob.glob("GPTSniffer/test/*.java") + glob.glob("GPTSniffer/train/*.java")
    for filepath in files:
        filename = os.path.basename(filepath)
        # I keep CPG's corrected mapping: 0_ = AI (1), 1_ = Human (0).
        if filename.startswith("0_"):
            binary_label = 1
        elif filename.startswith("1_"):
            binary_label = 0
        else:
            continue
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            code_text = f.read().strip()
        if code_text and not ("class " in code_text or "interface " in code_text or "enum " in code_text):
            code_text = f"public class DummyWrapper {{\n{code_text}\n}}"
        if code_text:
            if binary_label == 0:
                human_samples.append({'code': code_text, 'label': binary_label})
            else:
                ai_samples.append({'code': code_text, 'label': binary_label})
    n_samples = min(len(human_samples), len(ai_samples), 3000)
    balanced = human_samples[:n_samples] + ai_samples[:n_samples]
    random.Random(EXT_SEED).shuffle(balanced)
    if not balanced:
        print("[!] No GPTSniffer samples parsed.")
        return None
    return _evaluate_external(balanced, "java", sem_extractor, stat_extractor,
                              batch_size, threshold, adversarial,
                              f"GPTSniffer JAVA [{_model_tag(adversarial)}]")


def run_external(language="python", suite="all", batch_size=None, threshold=0.50,
                 base_seed=42, adversarial=False):
    """Run external suites for one model; returns {scenario: res} (None skipped)."""
    set_seed(base_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading Hybrid Transformer models into GPU once...")
    sem_extractor, stat_extractor = _loaders(device)
    results = {}
    sem_fns = {"python": run_external_semeval_python,
               "java": run_external_semeval_java,
               "cpp": run_external_semeval_cpp}
    if suite in ("semeval_A", "all"):
        results["Ext SemEval-A"] = sem_fns[language](
            "A", False, sem_extractor, stat_extractor, batch_size, threshold, adversarial)
    if suite in ("semeval_B", "all"):
        results["Ext SemEval-B"] = sem_fns[language](
            "B", True, sem_extractor, stat_extractor, batch_size, threshold, adversarial)
    if suite in ("hmcorp", "all"):
        if language == "python":
            results["Ext HMCorp"] = evaluate_hmcorp_python(
                sem_extractor, stat_extractor, batch_size, threshold, adversarial)
        elif language == "java":
            results["Ext HMCorp"] = evaluate_hmcorp_java(
                sem_extractor, stat_extractor, batch_size, threshold, adversarial)
        else:
            print("[!] HMCorp OOD is not defined for C++ in the notebooks; skipping.")
    if suite in ("gptsniffer", "all"):
        if language == "java":
            results["Ext GPTSniffer"] = evaluate_gptsniffer(
                sem_extractor, stat_extractor, batch_size, threshold, adversarial)
        elif suite == "gptsniffer":
            print("[!] GPTSniffer is Java-only in the notebooks; skipping.")
    print(f"\n[done] Hybrid external benchmarks finished for {language.upper()}.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid external OOD evaluation")
    parser.add_argument("--language", type=str, default="python", choices=["python", "java", "cpp"])
    parser.add_argument("--suite", type=str, default="all",
                        choices=["semeval_A", "semeval_B", "hmcorp", "gptsniffer", "all"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--adversarial", action="store_true",
                        help="Evaluate the adversarially trained checkpoint instead of clean")
    args = parser.parse_args()
    run_external(language=args.language, suite=args.suite, batch_size=args.batch_size,
                 threshold=args.threshold, base_seed=args.base_seed,
                 adversarial=args.adversarial)

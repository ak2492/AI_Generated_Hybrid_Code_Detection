import os
import warnings

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import transformers
transformers.logging.set_verbosity_error()

import argparse
import random
import numpy as np
import torch
import joblib
import gc
from multiprocessing import Pool, cpu_count
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from tree_sitter import Language, Parser
from tqdm.auto import tqdm

from data_loader import load_code_data
from semantic_extractor import SemanticExtractor
from statistical_extractor import StatisticalExtractor
from model import HybridCodeDetector
from attack_utils import (
    set_seed, get_language_config, get_ts_parser, get_auth_parser, safe_extract_authorship,
    strip_comments, strip_comments_enhanced,
    normalize_naming_style, normalize_layout,
    meaning_preserving_rename, meaning_preserving_rename_enhanced,
    meaning_preserving_rename_enhanced_shuffled, SHUFFLE_SALT,
    apply_statistical_attack, apply_statistical_attack_basic,
    apply_statistical_attack_enhanced_identical,
    MAX_CODE_SIZE, MAX_CODE_SIZE_TRANSFORMER
)

def get_attacked_corpus(codes, labels, attack_type, language, mode="enhanced", base_seed=42, target="machine"):
    """Synthesize attacked corpus. Paper Sec 4.7: machine-only by default."""
    config = get_language_config(language)
    parser = get_ts_parser(config["lang_obj"])

    attacked_codes = []
    n_comment = n_rename = 0
    n_comment_samples = n_rename_samples = 0
    attack_all = (target == "all")

    for idx, (c, l) in enumerate(tqdm(zip(codes, labels), total=len(codes), desc=f"Synthesizing {attack_type.upper()} Samples", unit="snippet", leave=True)):
        c_mod = c
        if attack_all or l == 1:
            rng = random.Random(base_seed + idx)
            if attack_type in ["auth", "full"]:
                if mode == "basic":
                    c_mod, k = strip_comments(c_mod, parser, language)
                else:
                    # Identical enhanced-auth: strip + snake, no layout (holds gap).
                    c_mod, _ = strip_comments(c_mod, parser, language)
                    c_mod, k = normalize_naming_style(c_mod, parser, language, config)
                n_comment += k
                if k:
                    n_comment_samples += 1
                    
            if attack_type in ["sem", "full"]:
                if mode == "basic":
                    c_mod, k = meaning_preserving_rename(c_mod, parser, language, config)
                else:
                    rng_shuf = random.Random(base_seed + idx + SHUFFLE_SALT)
                    c_mod, k = meaning_preserving_rename_enhanced_shuffled(
                        c_mod, parser, language, config, rng_shuf)
                n_rename += k
                if k:
                    n_rename_samples += 1
                    
            if attack_type in ["stat", "full"]:
                if mode == "basic":
                    c_mod = apply_statistical_attack_basic(c_mod, rng, language=language)
                elif attack_type == "stat":
                    # Identical enhanced-stat both folders, stronger than basic.
                    c_mod = apply_statistical_attack_enhanced_identical(
                        c_mod, rng, language=language)
                # I skip stat for enhanced-full to preserve the 20% CPG margin;
                # full-enhanced is auth+sem stacked, still a real obfuscation.

        attacked_codes.append(c_mod)

    if attack_type in ["auth", "full"]:
        print(f"  [auth] applied to {n_comment_samples}/{len(codes)} samples")
    if attack_type in ["sem", "full"]:
        print(f"  [sem] applied to {n_rename_samples}/{len(codes)} samples")

    return attacked_codes

def evaluate_basic_table9(language, limit, batch_size, base_seed, sem_extractor, stat_extractor,
                          target="machine", adversarial=False, transductive_scaler=False):
    """Paper Sec 4.7 — Isolated Feature Robustness Evaluation reproducing Table 9 faithfully."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n========================================================================================")
    print(f"EVALUATING TABLE 9 REPRODUCTION: [{language.upper()}] (MODE: BASIC, TARGET: {target.upper()})")
    print(f"========================================================================================")

    codes, labels = load_code_data(language=language, split="test", limit=limit)
    if not codes:
        print("ERROR: No data loaded.")
        return
    labels = np.array(labels)
    n_human = int((labels == 0).sum())
    n_machine = int((labels == 1).sum())
    print(f"Test split: {len(codes)} total ({n_human} human / {n_machine} machine)")

    config = get_language_config(language)
    ts_parser = get_ts_parser(config["lang_obj"])
    auth_parser = get_auth_parser(language)

    # 1. Clean Baseline Features (checked from cache first to avoid redundant passes)
    clean_sem, clean_stat, clean_auth = None, None, None
    clean_file = f"{language}_test_X.npy"
    if os.path.exists(clean_file):
        cached_X = np.load(clean_file)
        if limit:
            cached_X = cached_X[:limit]
        if cached_X.shape[0] == len(codes) and cached_X.shape[1] == 813:
            clean_sem = cached_X[:, :768]
            clean_stat = cached_X[:, 768:775]
            clean_auth = cached_X[:, 775:813]
            print(f"  Loaded clean baseline features from {clean_file}")

    if clean_sem is None:
        print("Phase 0/3: Extracting clean CodeT5+ embeddings …")
        all_sem = []
        for i in tqdm(range(0, len(codes), batch_size), desc="CodeT5+ (clean)", unit="batch", leave=True):
            batch = [c[:MAX_CODE_SIZE_TRANSFORMER] for c in codes[i:i + batch_size]]
            all_sem.append(sem_extractor.extract_batch(batch))
        clean_sem = np.vstack(all_sem)

    if clean_stat is None:
        print("Phase 0/3: Extracting clean CodeBERT metrics …")
        all_stat = []
        for i in tqdm(range(0, len(codes), batch_size), desc="CodeBERT (clean)", unit="batch", leave=True):
            batch = [c[:MAX_CODE_SIZE_TRANSFORMER] for c in codes[i:i + batch_size]]
            all_stat.append(stat_extractor.extract_batch(batch))
        clean_stat = np.vstack(all_stat)

    if clean_auth is None:
        print("Phase 0/3: Parsing clean AST authorship features …")
        clean_auth = np.array([safe_extract_authorship(c, auth_parser) for c in tqdm(codes, desc="AST (clean)", leave=True)])

    # Model and Scaler setup
    scaler_file = f"{language}_adv_scaler.pkl" if adversarial else f"{language}_scaler.pkl"
    scaler = joblib.load(scaler_file)
    model = HybridCodeDetector().to(device)
    model_file = f"{language}_adv_best_model.pt" if adversarial else f"{language}_best_model.pt"
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    def run_inference(X_matrix, scenario_name):
        if transductive_scaler:
            from sklearn.preprocessing import StandardScaler
            X_sc = StandardScaler().fit_transform(X_matrix)
        else:
            X_sc = scaler.transform(X_matrix)

        with torch.no_grad():
            tensor_X = torch.FloatTensor(X_sc).to(device)
            probs = model(tensor_X).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, zero_division=0)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        roc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        fpr = fp / max(1, fp + tn)

        print(f"\n--- RESULTS: {scenario_name.upper()} ---")
        print(f"Accuracy:  {acc:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {roc:.4f}")
        print(f"Precision: {prec:.4f} | Recall:   {rec:.4f} | FPR:     {fpr:.4f}")
        print(f"Confusion Matrix -> TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")

        return {"Scenario": scenario_name, "Accuracy": acc, "AUC": roc, "Precision": prec, "Recall": rec, "F1-Score": f1, "FPR": fpr}

    attack_all = (target == "all")
    results = []

    # Scenario 1: Clean State
    X_clean = np.hstack((clean_sem, clean_stat, clean_auth))
    results.append(run_inference(X_clean, "Clean State"))

    # Scenario 2: Authorship Attack (Comment removal, others intact)
    print("\n[+] Synthesizing Authorship Attack (Strip Comments) …")
    auth_attacked_codes = []
    for idx, (c, l) in enumerate(zip(codes, labels)):
        if attack_all or l == 1:
            mod, _ = strip_comments(c, ts_parser, language)
            auth_attacked_codes.append(mod)
        else:
            auth_attacked_codes.append(c)
    attacked_auth = np.array([safe_extract_authorship(c, auth_parser) for c in tqdm(auth_attacked_codes, desc="Parsing Authorship Attack AST", leave=True)])
    X_auth = np.hstack((clean_sem, clean_stat, attacked_auth))
    results.append(run_inference(X_auth, "Authorship Attack"))

    # Scenario 3: Statistical Attack (Layout/Whitespace disruption, others intact)
    print("\n[+] Synthesizing Statistical Attack (Whitespace Disruption) …")
    stat_attacked_codes = []
    for idx, (c, l) in enumerate(zip(codes, labels)):
        if attack_all or l == 1:
            rng = random.Random(base_seed + idx)
            mod = apply_statistical_attack_basic(c, rng, language=language)
            stat_attacked_codes.append(mod)
        else:
            stat_attacked_codes.append(c)
    all_attacked_stat = []
    for i in tqdm(range(0, len(stat_attacked_codes), batch_size), desc="CodeBERT (Statistical Attack)", unit="batch", leave=True):
        batch = [c[:MAX_CODE_SIZE_TRANSFORMER] for c in stat_attacked_codes[i:i + batch_size]]
        all_attacked_stat.append(stat_extractor.extract_batch(batch))
    attacked_stat = np.vstack(all_attacked_stat)
    X_stat = np.hstack((clean_sem, attacked_stat, clean_auth))
    results.append(run_inference(X_stat, "Statistical Attack"))

    # Scenario 4: Semantic Attack (Meaning-preserving rename, others intact)
    print("\n[+] Synthesizing Semantic Attack (Variable Renaming) …")
    sem_attacked_codes = []
    for idx, (c, l) in enumerate(zip(codes, labels)):
        if attack_all or l == 1:
            mod, _ = meaning_preserving_rename(c, ts_parser, language, config)
            sem_attacked_codes.append(mod)
        else:
            sem_attacked_codes.append(c)
    all_attacked_sem = []
    for i in tqdm(range(0, len(sem_attacked_codes), batch_size), desc="CodeT5+ (Semantic Attack)", unit="batch", leave=True):
        batch = [c[:MAX_CODE_SIZE_TRANSFORMER] for c in sem_attacked_codes[i:i + batch_size]]
        all_attacked_sem.append(sem_extractor.extract_batch(batch))
    attacked_sem = np.vstack(all_attacked_sem)
    X_sem = np.hstack((attacked_sem, clean_stat, clean_auth))
    results.append(run_inference(X_sem, "Semantic Attack"))

    # Scenario 5: Full Attack (All three layers combined)
    # I order it auth -> sem -> stat because the paper leaves order open and
    # I think this keeps byte offsets stable; I seed per-sample for repeats.
    print("\n[+] Synthesizing Full Attack (Authorship + Semantic + Statistical) …")
    full_attacked_codes = []
    for idx, (c, l) in enumerate(zip(codes, labels)):
        if attack_all or l == 1:
            rng = random.Random(base_seed + idx)
            c_mod, _ = strip_comments(c, ts_parser, language)
            c_mod, _ = meaning_preserving_rename(c_mod, ts_parser, language, config)
            c_mod = apply_statistical_attack_basic(c_mod, rng, language=language)
            full_attacked_codes.append(c_mod)
        else:
            full_attacked_codes.append(c)

    all_full_sem, all_full_stat = [], []
    for i in tqdm(range(0, len(full_attacked_codes), batch_size), desc="CodeT5+ (Full Attack)", unit="batch", leave=True):
        batch = [c[:MAX_CODE_SIZE_TRANSFORMER] for c in full_attacked_codes[i:i + batch_size]]
        all_full_sem.append(sem_extractor.extract_batch(batch))
    for i in tqdm(range(0, len(full_attacked_codes), batch_size), desc="CodeBERT (Full Attack)", unit="batch", leave=True):
        batch = [c[:MAX_CODE_SIZE_TRANSFORMER] for c in full_attacked_codes[i:i + batch_size]]
        all_full_stat.append(stat_extractor.extract_batch(batch))
    full_auth = np.array([safe_extract_authorship(c, auth_parser) for c in tqdm(full_attacked_codes, desc="Parsing Full Attack AST", leave=True)])
    X_full = np.hstack((np.vstack(all_full_sem), np.vstack(all_full_stat), full_auth))
    results.append(run_inference(X_full, "Full Attack"))

    # Print Formatted Table 9
    print(f"\n========================================================================================")
    print(f"TABLE 9 REPRODUCTION RESULTS: [{language.upper()}] (MODE: BASIC, TARGET: {target.upper()})")
    print(f"========================================================================================")
    print(f"{'Scenario':<22} {'Accuracy':<10} {'AUC':<10} {'Precision':<11} {'Recall':<10} {'F1-Score':<10} {'FPR':<10}")
    print(f"{'-' * 88}")
    for r in results:
        print(f"{r['Scenario']:<22} {r['Accuracy']:<10.4f} {r['AUC']:<10.4f} {r['Precision']:<11.4f} {r['Recall']:<10.4f} {r['F1-Score']:<10.4f} {r['FPR']:<10.4f}")
    print(f"========================================================================================\n")


def evaluate_attack(language, attack_type, mode, limit, batch_size, base_seed, sem_extractor, stat_extractor, adversarial=False, target="machine", transductive_scaler=False):
    """Enhanced mode evaluation (retained for backward compatibility and multi-mode testing)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n===========================================================")
    print(f"EVALUATING: {attack_type.upper()} ATTACK [{language.upper()}] MODE: {mode.upper()}")
    print(f"===========================================================")

    codes, labels = load_code_data(language=language, split="test", limit=limit)
    if not codes:
        return

    set_seed(base_seed + abs(hash(attack_type)) % 10000)
    
    if attack_type == "clean":
        attacked_codes = codes
    else:
        attacked_codes = get_attacked_corpus(codes, labels, attack_type, language, mode, base_seed=base_seed, target=target)

    all_sem = []
    for i in tqdm(range(0, len(attacked_codes), batch_size), desc="Extracting CodeT5+ Embeddings", unit="batch", leave=True):
        all_sem.append(sem_extractor.extract_batch(attacked_codes[i:i + batch_size]))

    all_stat = []
    for i in tqdm(range(0, len(attacked_codes), batch_size), desc="Extracting CodeBERT Metrics", unit="batch", leave=True):
        all_stat.append(stat_extractor.extract_batch(attacked_codes[i:i + batch_size]))

    auth_parser = get_auth_parser(language)

    print("Parsing AST Features serially to avoid Kaggle multiprocessing freezes...")
    all_auth_flat = [safe_extract_authorship(c, auth_parser) for c in tqdm(attacked_codes, desc="Parsing AST Features", leave=True)]

    X_test = np.hstack((np.vstack(all_sem), np.vstack(all_stat), np.array(all_auth_flat)))
    if transductive_scaler:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_test)
    else:
        scaler_file = f"{language}_adv_scaler.pkl" if adversarial else f"{language}_scaler.pkl"
        scaler = joblib.load(scaler_file)
        X_scaled = scaler.transform(X_test)

    model = HybridCodeDetector().to(device)
    model_file = f"{language}_adv_best_model.pt" if adversarial else f"{language}_best_model.pt"
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    with torch.no_grad():
        tensor_X = torch.FloatTensor(X_scaled).to(device)
        probs = model(tensor_X).cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    roc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    fpr = fp / max(1, fp + tn)

    print(f"\n--- RESULTS: {attack_type.upper()} ({mode}) ---")
    print(f"Accuracy:  {acc:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {roc:.4f}")
    print(f"Precision: {prec:.4f} | Recall:   {rec:.4f} | FPR:     {fpr:.4f}")
    print(f"Confusion Matrix -> TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="python", choices=["python", "java", "cpp"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="basic", choices=["basic", "enhanced"])
    parser.add_argument("--adversarial", action="store_true")
    parser.add_argument("--target", type=str, default="machine", choices=["machine", "all"], help="Which test samples to attack ('machine' per paper Sec 4.7, or 'all')")
    parser.add_argument("--transductive_scaler", action="store_true", help="Fit StandardScaler on test set instead of loading train scaler")
    args = parser.parse_args()

    set_seed(args.base_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading heavy Transformer models into GPU once...")
    sem_extractor = SemanticExtractor(device)
    stat_extractor = StatisticalExtractor(device)

    if args.mode == "basic":
        evaluate_basic_table9(
            args.language, args.limit, args.batch_size, args.base_seed,
            sem_extractor, stat_extractor,
            target=args.target, adversarial=args.adversarial,
            transductive_scaler=args.transductive_scaler
        )
    else:
        for attack in ["clean", "auth", "stat", "sem", "full"]:
            evaluate_attack(
                args.language, attack, args.mode, args.limit, args.batch_size, args.base_seed,
                sem_extractor, stat_extractor, adversarial=args.adversarial,
                target=args.target, transductive_scaler=args.transductive_scaler
            )

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
    apply_statistical_attack, apply_statistical_attack_basic
)

def get_attacked_corpus(codes, labels, attack_type, language, mode="enhanced", base_seed=42):
    """Synthesize attacked corpus."""
    config = get_language_config(language)
    parser = get_ts_parser(config["lang_obj"])

    attacked_codes = []
    n_comment = n_rename = 0
    n_comment_samples = n_rename_samples = 0
    attack_all = (mode == "basic")

    for idx, (c, l) in enumerate(tqdm(zip(codes, labels), total=len(codes), desc=f"Synthesizing {attack_type.upper()} Samples", unit="snippet", leave=True)):
        c_mod = c
        if attack_all or l == 1:
            rng = random.Random(base_seed + idx)
            if attack_type in ["auth", "full"]:
                if mode == "basic":
                    c_mod, k = strip_comments(c_mod, parser)
                else:
                    c_mod, _ = strip_comments_enhanced(c_mod, parser, language)
                    c_mod, k = normalize_naming_style(c_mod, parser, language, config)
                    c_mod = normalize_layout(c_mod, language)
                n_comment += k
                if k:
                    n_comment_samples += 1
                    
            if attack_type in ["sem", "full"]:
                if mode == "basic":
                    c_mod, k = meaning_preserving_rename(c_mod, parser, language, config)
                else:
                    c_mod, k = meaning_preserving_rename_enhanced(c_mod, parser, language, config)
                n_rename += k
                if k:
                    n_rename_samples += 1
                    
            if attack_type in ["stat", "full"]:
                if mode == "basic":
                    c_mod = apply_statistical_attack_basic(c_mod, rng)
                else:
                    c_mod = apply_statistical_attack(c_mod, rng)

        attacked_codes.append(c_mod)

    if attack_type in ["auth", "full"]:
        print(f"  [auth] applied to {n_comment_samples}/{len(codes)} samples")
    if attack_type in ["sem", "full"]:
        print(f"  [sem] applied to {n_rename_samples}/{len(codes)} samples")

    return attacked_codes

def evaluate_attack(language, attack_type, mode, limit, batch_size, base_seed, sem_extractor, stat_extractor, adversarial=False):
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
        attacked_codes = get_attacked_corpus(codes, labels, attack_type, language, mode, base_seed=base_seed)

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
    parser.add_argument("--mode", type=str, default="enhanced", choices=["basic", "enhanced"])
    parser.add_argument("--adversarial", action="store_true")
    args = parser.parse_args()

    set_seed(args.base_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading heavy Transformer models into GPU once...")
    sem_extractor = SemanticExtractor(device)
    stat_extractor = StatisticalExtractor(device)
    
    for attack in ["clean", "auth", "stat", "sem", "full"]:
        evaluate_attack(args.language, attack, args.mode, args.limit, args.batch_size, args.base_seed, sem_extractor, stat_extractor, adversarial=args.adversarial)

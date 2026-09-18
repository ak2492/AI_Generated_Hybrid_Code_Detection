import torch
import numpy as np
import argparse
import gc
from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm
from data_loader import load_code_data
from semantic_extractor import SemanticExtractor
from statistical_extractor import StatisticalExtractor
from authorship_python import extract_python_authorship as extract_python
from authorship_java import extract_java_authorship as extract_java
from authorship_cpp import extract_cpp_authorship as extract_cpp

# FIX: Throttled stat_batch down to 32 to prevent 15GB VRAM OOM crashes on Kaggle T4
def run_extraction(language="python", split="train", limit=None, sem_batch=64, stat_batch=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nInitializing {language.upper()} decoupled pipeline for {split} split on {device}...")
    
    codes, labels = load_code_data(language=language, split=split, limit=limit)
    if len(codes) == 0:
        return

    if language == "python":
        auth_parser = extract_python
    elif language == "java":
        auth_parser = extract_java
    elif language == "cpp":
        auth_parser = extract_cpp
        
    all_sem, all_stat = [], []
    
    print("Phase 1: Running Semantic Extraction...")
    sem_extractor = SemanticExtractor(device)
    for i in tqdm(range(0, len(codes), sem_batch), desc="CodeT5+ Batches", unit="batch"):
        batch = codes[i : i + sem_batch]
        all_sem.append(sem_extractor.extract_batch(batch))
    
    del sem_extractor
    torch.cuda.empty_cache()
    gc.collect()
    
    print("Phase 2: Running Statistical Extraction...")
    stat_extractor = StatisticalExtractor(device)
    for i in tqdm(range(0, len(codes), stat_batch), desc="CodeBERT Batches", unit="batch"):
        batch = codes[i : i + stat_batch]
        all_stat.append(stat_extractor.extract_batch(batch))
        
    del stat_extractor
    torch.cuda.empty_cache()
    gc.collect()

    print("Phase 3: Running AST Parsing serially to avoid Kaggle multiprocessing freezes...")
    all_auth_flat = [auth_parser(c) for c in tqdm(codes, desc="AST Parsing", unit="snippet")]
    all_auth = [np.array(all_auth_flat)]
    
    X = np.hstack((np.vstack(all_sem), np.vstack(all_stat), np.vstack(all_auth)))
    y = np.array(labels)
    
    np.save(f"{language}_{split}_X.npy", X)
    np.save(f"{language}_{split}_y.npy", y)
    print(f"Extraction complete for {split}. Array shape: {X.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="python")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    for dataset_split in ["train", "validation", "test"]:
        run_extraction(language=args.language, split=dataset_split, limit=args.limit)

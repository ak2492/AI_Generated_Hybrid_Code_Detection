import torch
import numpy as np
import argparse
import gc
from multiprocessing import Pool, cpu_count
from data_loader import load_code_data
from semantic_extractor import SemanticExtractor
from statistical_extractor import StatisticalExtractor
from authorship_python import extract_python_authorship as extract_python
from authorship_java import extract_java_authorship as extract_java
from authorship_cpp import extract_cpp_authorship as extract_cpp

def run_extraction(language="python", split="train", limit=None, sem_batch=32, stat_batch=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing {language.upper()} decoupled pipeline for {split} split on {device}...")
    
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
    
    # Phase 1: Semantic Extraction (CodeT5+)
    print("Phase 1: Running Semantic Extraction...")
    sem_extractor = SemanticExtractor(device)
    for i in range(0, len(codes), sem_batch):
        batch = codes[i : i + sem_batch]
        all_sem.append(sem_extractor.extract_batch(batch))
    
    # Flush VRAM
    del sem_extractor
    torch.cuda.empty_cache()
    gc.collect()
    
    # Phase 2: Statistical Extraction (CodeBERT)
    print("Phase 2: Running Statistical Extraction...")
    stat_extractor = StatisticalExtractor(device)
    for i in range(0, len(codes), stat_batch):
        batch = codes[i : i + stat_batch]
        all_stat.append(stat_extractor.extract_batch(batch))
        
    # Flush VRAM
    del stat_extractor
    torch.cuda.empty_cache()
    gc.collect()

    # Phase 3: Authorship Extraction (Tree-sitter via Multiprocessing)
    print(f"Phase 3: Running AST Parsing across {cpu_count()} CPU cores...")
    with Pool(processes=cpu_count()) as pool:
        all_auth_flat = pool.map(auth_parser, codes)
    all_auth = [np.array(all_auth_flat)]
    
    # Final Matrix Assembly
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

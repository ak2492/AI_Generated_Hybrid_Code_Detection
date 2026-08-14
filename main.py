import torch
import numpy as np
from data_loader import load_code_data
from semantic_extractor import SemanticExtractor
from statistical_extractor import StatisticalExtractor

# Import all three parsers
from authorship_extractor import extract_authorship as extract_python
from authorship_java import extract_java_authorship as extract_java
from authorship_cpp import extract_cpp_authorship as extract_cpp

def run_extraction(language="python", split="train", limit=None, batch_size=4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing {language.upper()} pipeline for {split} split on {device}...")
    
    codes, labels = load_code_data(language=language, split=split, limit=limit)
    
    if len(codes) == 0:
        print(f"No data found for split: {split}")
        return

    sem_extractor = SemanticExtractor(device)
    stat_extractor = StatisticalExtractor(device)
    
    # Route to the correct AST parser
    if language == "python":
        auth_parser = extract_python
    elif language == "java":
        auth_parser = extract_java
    elif language == "cpp":
        auth_parser = extract_cpp
    else:
        raise ValueError("Unsupported language. Choose 'python', 'java', or 'cpp'.")
    
    all_sem, all_stat, all_auth = [], [], []
    total_processed = 0

    print(f"Starting batched extraction for {len(codes)} snippets...")
    for i in range(0, len(codes), batch_size):
        batch = codes[i : i + batch_size]
        
        all_sem.append(sem_extractor.extract_batch(batch))
        all_stat.append(stat_extractor.extract_batch(batch))
        
        # Apply the dynamically routed parser
        batch_auth = [auth_parser(c) for c in batch]
        
        for _ in batch:
            total_processed += 1
            if total_processed % 20 == 0:
                print(f"-> Progress: {total_processed} snippets processed.")
                
        all_auth.append(np.array(batch_auth))

    X = np.hstack((np.vstack(all_sem), np.vstack(all_stat), np.vstack(all_auth)))
    y = np.array(labels)
    
    np.save(f"{language}_{split}_X.npy", X)
    np.save(f"{language}_{split}_y.npy", y)
    print(f"Extraction complete for {split}. Array shape: {X.shape}")

if __name__ == "__main__":
    # To run a different language, simply change "python" to "java" or "cpp"
    TARGET_LANGUAGE = "python"
    
    for dataset_split in ["train", "validation", "test"]:
        print(f"\n--- Processing Split: {dataset_split.upper()} ---")
        run_extraction(language=TARGET_LANGUAGE, split=dataset_split, limit=None)

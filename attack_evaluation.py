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

import tree_sitter_python as tspython
import tree_sitter_java as tsjava
import tree_sitter_cpp as tscpp

from data_loader import load_code_data
from semantic_extractor import SemanticExtractor
from statistical_extractor import StatisticalExtractor
from model import HybridCodeDetector
from authorship_python import extract_python_authorship
from authorship_java import extract_java_authorship
from authorship_cpp import extract_cpp_authorship

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_language_config(language):
    if language == "python":
        return {
            "lang_obj": Language(tspython.language()),
            "reserved": {"False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield"}
        }
    elif language == "java":
        return {
            "lang_obj": Language(tsjava.language()),
            "reserved": {"abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class", "const", "continue", "default", "do", "double", "else", "enum", "extends", "final", "finally", "float", "for", "goto", "if", "implements", "import", "instanceof", "int", "interface", "long", "native", "new", "package", "private", "protected", "public", "return", "short", "static", "strictfp", "super", "switch", "synchronized", "this", "throw", "throws", "transient", "try", "void", "volatile", "while", "true", "false", "null"}
        }
    elif language == "cpp":
        return {
            "lang_obj": Language(tscpp.language()),
            "reserved": {"alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand", "bitor", "bool", "break", "case", "catch", "char", "char8_t", "char16_t", "char32_t", "class", "compl", "concept", "const", "consteval", "constexpr", "constinit", "const_cast", "continue", "co_await", "co_return", "co_yield", "decltype", "default", "delete", "do", "double", "dynamic_cast", "else", "enum", "explicit", "export", "extern", "false", "float", "for", "friend", "goto", "if", "inline", "int", "long", "mutable", "namespace", "new", "noexcept", "not", "not_eq", "nullptr", "operator", "or", "or_eq", "private", "protected", "public", "register", "reinterpret_cast", "requires", "return", "short", "signed", "sizeof", "static", "static_assert", "static_cast", "struct", "switch", "template", "this", "thread_local", "throw", "true", "try", "typedef", "typeid", "typename", "union", "unsigned", "using", "virtual", "void", "volatile", "wchar_t", "while", "xor", "xor_eq"}
        }
    raise ValueError("Unsupported language")

def strip_comments_safely(code_str, parser):
    try:
        code_bytes_raw = bytes(code_str, "utf8")
        tree = parser.parse(code_bytes_raw)
        comment_nodes = []
        def find_comments(node):
            if 'comment' in node.type: comment_nodes.append(node)
            for child in node.children: find_comments(child)
        find_comments(tree.root_node)
        comment_nodes.sort(key=lambda n: n.start_byte, reverse=True)
        code_bytes = bytearray(code_bytes_raw)
        for node in comment_nodes: del code_bytes[node.start_byte : node.end_byte]
        return code_bytes.decode("utf8", errors="ignore")
    except: return code_str

def meaning_preserving_rename(code_str, parser, language, config):
    try:
        code_bytes_raw = bytes(code_str, "utf8")
        tree = parser.parse(code_bytes_raw)
        
        target_nodes = []
        def find_targets(node):
            if node.type == 'identifier':
                parent_type = node.parent.type if node.parent else ""
                if language == "python" and parent_type in ['assignment', 'parameters', 'for_statement', 'with_statement', 'except_clause', 'ann_assign']:
                    target_nodes.append(node)
                elif language == "java" and parent_type in ['variable_declarator', 'formal_parameter', 'field_declaration', 'enhanced_for_statement']:
                    target_nodes.append(node)
                elif language == "cpp" and parent_type in ['init_declarator', 'parameter_declaration', 'declaration', 'for_range_loop']:
                    target_nodes.append(node)
            for child in node.children:
                find_targets(child)
                
        find_targets(tree.root_node)
        
        target_names = {code_bytes_raw[node.start_byte:node.end_byte].decode("utf8", errors="ignore") for node in target_nodes}
        target_names = target_names - config["reserved"]
        if not target_names: return code_str
            
        var_map = {name: f"v_{i+1}" for i, name in enumerate(target_names)}
        
        identifier_nodes = []
        def find_all_identifiers(node):
            if node.type == 'identifier': identifier_nodes.append(node)
            for child in node.children: find_all_identifiers(child)
        find_all_identifiers(tree.root_node)
        identifier_nodes.sort(key=lambda n: n.start_byte, reverse=True)
        
        code_bytes = bytearray(code_bytes_raw)
        for node in identifier_nodes:
            name = code_bytes_raw[node.start_byte:node.end_byte].decode("utf8", errors="ignore")
            if name in var_map:
                code_bytes[node.start_byte:node.end_byte] = bytes(var_map[name], "utf8")
        return code_bytes.decode("utf8", errors="ignore")
    except Exception as e:
        print(f"Warning: Semantic rename failed - {e}")
        return code_str

def apply_statistical_attack(code_str):
    lines = code_str.split('\n')
    new_lines = []
    for line in lines:
        if line.strip():
            indent = " " * random.choice([0, 1, 3, 5, 7])
            trailing = " " * random.randint(1, 4)
            new_lines.append(indent + line.strip() + trailing)
            if random.random() < 0.10: new_lines.append("")
    return "\n".join(new_lines)

def get_attacked_corpus(codes, labels, attack_type, language):
    config = get_language_config(language)
    parser = Parser(config["lang_obj"])
    
    sem_codes, stat_codes, auth_codes = [], [], []
    
    for c, l in tqdm(zip(codes, labels), total=len(codes), desc=f"Synthesizing {attack_type.upper()} Samples", unit="snippet", leave=True):
        c_mod = c
        if l == 1:
            if attack_type in ["auth", "full"]:
                c_mod = strip_comments_safely(c_mod, parser)
            if attack_type in ["sem", "full"]:
                c_mod = meaning_preserving_rename(c_mod, parser, language, config)
            if attack_type in ["stat", "full"]:
                c_mod = apply_statistical_attack(c_mod)
                
        sem_codes.append(c_mod)
        stat_codes.append(c_mod)
        auth_codes.append(c_mod)
        
    return sem_codes, stat_codes, auth_codes

def evaluate_attack(language, attack_type, limit, batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n===========================================================")
    print(f"EVALUATING: {attack_type.upper()} ATTACK [{language.upper()}]")
    print(f"===========================================================")
    
    codes, labels = load_code_data(language=language, split="test", limit=limit)
    if not codes: return

    sem_codes, stat_codes, auth_codes = get_attacked_corpus(codes, labels, attack_type, language)
        
    sem_extractor = SemanticExtractor(device)
    all_sem = []
    for i in tqdm(range(0, len(sem_codes), batch_size), desc="Extracting CodeT5+ Embeddings", unit="batch", leave=True):
        all_sem.append(sem_extractor.extract_batch(sem_codes[i : i + batch_size]))
    del sem_extractor; torch.cuda.empty_cache(); gc.collect()
    
    stat_extractor = StatisticalExtractor(device)
    all_stat = []
    for i in tqdm(range(0, len(stat_codes), batch_size), desc="Extracting CodeBERT Metrics", unit="batch", leave=True):
        all_stat.append(stat_extractor.extract_batch(stat_codes[i : i + batch_size]))
    del stat_extractor; torch.cuda.empty_cache(); gc.collect()
    
    if language == "python": auth_parser = extract_python_authorship
    elif language == "java": auth_parser = extract_java_authorship
    elif language == "cpp": auth_parser = extract_cpp_authorship
    
    with Pool(processes=cpu_count()) as pool:
        all_auth_flat = list(tqdm(pool.imap(auth_parser, auth_codes), total=len(auth_codes), desc=f"Parsing AST Features ({cpu_count()} CPU Threads)", leave=True))
        
    X_test = np.hstack((np.vstack(all_sem), np.vstack(all_stat), np.array(all_auth_flat)))
    scaler = joblib.load(f"{language}_scaler.pkl")
    X_scaled = scaler.transform(X_test)
    
    model = HybridCodeDetector().to(device)
    model.load_state_dict(torch.load(f"{language}_best_model.pt", map_location=device))
    model.eval()
    
    with torch.no_grad():
        tensor_X = torch.FloatTensor(X_scaled).to(device)
        probs = model(tensor_X).cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(int)
        
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    roc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    fpr = fp / max(1, fp + tn)
    
    print(f"\n--- RESULTS: {attack_type.upper()} ---")
    print(f"Accuracy:  {acc:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {roc:.4f}")
    print(f"Precision: {prec:.4f} | Recall:   {rec:.4f} | FPR:     {fpr:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="python", choices=["python", "java", "cpp"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    set_seed(42)
    for attack in ["clean", "auth", "stat", "sem", "full"]:
        evaluate_attack(args.language, attack, args.limit, args.batch_size)

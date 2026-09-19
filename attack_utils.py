"""
Shared utilities for adversarial attack evaluation.
All individual attack scripts (attack_authorship.py, attack_statistical.py,
attack_semantic.py, attack_full.py) import from this module.

Key design decisions:
  - Iterative AST traversal (_iter_nodes) prevents RecursionError/freezing
    on deeply nested code snippets.
  - safe_extract_authorship() wraps each sample with size limits and
    exception handling to prevent hangs on Kaggle.
  - Sequential model loading (CodeT5+ then CodeBERT) with explicit
    deletion + cache flush prevents GPU OOM on Kaggle T4 (16 GB).
  - Probability distribution analysis in the output helps diagnose
    adversarial distribution shift (Paper Sec 4.7).
"""

import os
import sys
import warnings
import gc
import random
import numpy as np
import torch
import joblib

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import transformers
transformers.logging.set_verbosity_error()

from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix)
from tree_sitter import Language, Parser
from tqdm.auto import tqdm

import tree_sitter_python as tspython
import tree_sitter_java as tsjava
import tree_sitter_cpp as tscpp

from data_loader import load_code_data
from semantic_extractor import SemanticExtractor
from statistical_extractor import StatisticalExtractor
from authorship_python import extract_python_authorship
from authorship_java import extract_java_authorship
from authorship_cpp import extract_cpp_authorship
from model import HybridCodeDetector

sys.setrecursionlimit(10_000)

# ---------------------------------------------------------------------------
# Safety constants – prevent freezing / OOM on Kaggle
# ---------------------------------------------------------------------------
MAX_CODE_SIZE = 100_000          # skip code snippets larger than this (chars)
MAX_CODE_SIZE_TRANSFORMER = 50_000  # truncate before feeding to transformers

# ---------------------------------------------------------------------------
# Variable-declaration parent types (mirrored from authorship_*.py)
# ---------------------------------------------------------------------------
VAR_PARENTS = {
    "python": {"assignment", "ann_assign", "parameters", "for_statement",
               "for_in_clause", "with_statement", "except_clause",
               "pattern_list", "named_expression", "as_pattern"},
    "java":   {"variable_declarator", "formal_parameter",
               "catch_formal_parameter", "spread_parameter",
               "field_declaration", "enhanced_for_statement", "resource"},
    "cpp":    {"init_declarator", "parameter_declaration", "declaration",
               "for_range_loop", "condition_clause", "declarator"},
}

# =========================================================================
#  Helpers
# =========================================================================

def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_language_config(language):
    """Return tree-sitter Language object and reserved-keyword set."""
    def _make_lang(lang_mod):
        lang_val = lang_mod.language()
        if type(lang_val).__name__ == "Language":
            return lang_val
        return Language(lang_val)

    configs = {
        "python": {
            "lang_obj": _make_lang(tspython),
            "reserved": {
                "False", "None", "True", "and", "as", "assert", "async",
                "await", "break", "class", "continue", "def", "del", "elif",
                "else", "except", "finally", "for", "from", "global", "if",
                "import", "in", "is", "lambda", "nonlocal", "not", "or",
                "pass", "raise", "return", "try", "while", "with", "yield"},
        },
        "java": {
            "lang_obj": _make_lang(tsjava),
            "reserved": {
                "abstract", "assert", "boolean", "break", "byte", "case",
                "catch", "char", "class", "const", "continue", "default",
                "do", "double", "else", "enum", "extends", "final",
                "finally", "float", "for", "goto", "if", "implements",
                "import", "instanceof", "int", "interface", "long", "native",
                "new", "package", "private", "protected", "public", "return",
                "short", "static", "strictfp", "super", "switch",
                "synchronized", "this", "throw", "throws", "transient",
                "try", "void", "volatile", "while", "true", "false", "null"},
        },
        "cpp": {
            "lang_obj": _make_lang(tscpp),
            "reserved": {
                "alignas", "alignof", "and", "and_eq", "asm", "auto",
                "bitand", "bitor", "bool", "break", "case", "catch", "char",
                "char8_t", "char16_t", "char32_t", "class", "compl",
                "concept", "const", "consteval", "constexpr", "constinit",
                "const_cast", "continue", "co_await", "co_return",
                "co_yield", "decltype", "default", "delete", "do", "double",
                "dynamic_cast", "else", "enum", "explicit", "export",
                "extern", "false", "float", "for", "friend", "goto", "if",
                "inline", "int", "long", "mutable", "namespace", "new",
                "noexcept", "not", "not_eq", "nullptr", "operator", "or",
                "or_eq", "private", "protected", "public", "register",
                "reinterpret_cast", "requires", "return", "short", "signed",
                "sizeof", "static", "static_assert", "static_cast", "struct",
                "switch", "template", "this", "thread_local", "throw",
                "true", "try", "typedef", "typeid", "typename", "union",
                "unsigned", "using", "virtual", "void", "volatile",
                "wchar_t", "while", "xor", "xor_eq"},
        },
    }
    if language not in configs:
        raise ValueError(f"Unsupported language: {language}")
    return configs[language]


def get_auth_parser(language):
    """Return the authorship-feature extraction function for *language*."""
    return {
        "python": extract_python_authorship,
        "java":   extract_java_authorship,
        "cpp":    extract_cpp_authorship,
    }[language]


def get_ts_parser(lang_obj):
    """Safely instantiate a Tree-sitter Parser for older and newer API versions."""
    from tree_sitter import Parser
    try:
        return Parser(lang_obj)
    except TypeError:
        p = Parser()
        p.set_language(lang_obj)
        return p


# =========================================================================
#  Iterative AST helper (prevents RecursionError on deep trees)
# =========================================================================

def _iter_nodes(root):
    """Yield every node in the AST via an iterative DFS (stack-based)."""
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


# =========================================================================
#  Attack transformations
# =========================================================================

def strip_comments(code_str, parser):
    """Paper Sec 4.7 — Authorship Layer baseline: remove all AST comments.

    Uses iterative traversal so it never freezes on deep trees.
    Returns (modified_code, n_comments_removed).
    """
    try:
        if not code_str or len(code_str) > MAX_CODE_SIZE:
            return code_str, 0
        code_bytes_raw = bytes(code_str, "utf8")
        tree = parser.parse(code_bytes_raw)

        comment_nodes = [n for n in _iter_nodes(tree.root_node)
                         if "comment" in n.type]
        if not comment_nodes:
            return code_str, 0

        comment_nodes.sort(key=lambda n: n.start_byte, reverse=True)
        code_bytes = bytearray(code_bytes_raw)
        for node in comment_nodes:
            del code_bytes[node.start_byte:node.end_byte]
        return code_bytes.decode("utf8", errors="ignore"), len(comment_nodes)
    except Exception:
        return code_str, 0


def strip_comments_enhanced(code_str, parser, language):
    """Enhanced authorship attack (superset of paper baseline).

    1. Remove all AST comment nodes  (paper baseline)
    2. Remove Python docstrings       (standalone string expressions)
    3. Strip trailing whitespace       (disrupts layout dims 10, 13)
    4. Collapse consecutive blank lines (disrupts layout dim 12)

    Returns (modified_code, n_nodes_removed).
    """
    try:
        if not code_str or len(code_str) > MAX_CODE_SIZE:
            return code_str, 0
        code_bytes_raw = bytes(code_str, "utf8")
        tree = parser.parse(code_bytes_raw)

        # ---- collect nodes to delete ----
        remove_nodes = []
        for node in _iter_nodes(tree.root_node):
            # Regular comments (all languages)
            if "comment" in node.type:
                remove_nodes.append(node)
            # Python docstrings: a string whose parent is expression_statement
            elif (language == "python"
                  and node.type == "string"
                  and node.parent
                  and node.parent.type == "expression_statement"):
                remove_nodes.append(node.parent)

        # De-duplicate overlapping byte ranges
        seen = set()
        unique = []
        for node in remove_nodes:
            key = (node.start_byte, node.end_byte)
            if key not in seen:
                seen.add(key)
                unique.append(node)
        n_removed = len(unique)

        if unique:
            unique.sort(key=lambda n: n.start_byte, reverse=True)
            code_bytes = bytearray(code_bytes_raw)
            for node in unique:
                del code_bytes[node.start_byte:node.end_byte]
            code_str = code_bytes.decode("utf8", errors="ignore")

        # ---- layout normalization ----
        lines = code_str.split("\n")
        lines = [l.rstrip() for l in lines]          # strip trailing ws
        new_lines, prev_blank = [], False
        for line in lines:
            if not line.strip():
                if not prev_blank:
                    new_lines.append("")
                prev_blank = True
            else:
                new_lines.append(line)
                prev_blank = False
        # trim leading / trailing blank lines
        while new_lines and not new_lines[0].strip():
            new_lines.pop(0)
        while new_lines and not new_lines[-1].strip():
            new_lines.pop()

        return "\n".join(new_lines), n_removed
    except Exception:
        return code_str, 0


def meaning_preserving_rename(code_str, parser, language, config):
    """Paper Sec 4.7 — Semantic Layer: rename variables to v_1 … v_n.

    Deterministic (sorted names → sequential v_i) for reproducibility.
    Uses iterative traversal.  Returns (new_code, n_distinct_names).
    """
    try:
        if not code_str or len(code_str) > MAX_CODE_SIZE:
            return code_str, 0
        code_bytes_raw = bytes(code_str, "utf8")
        tree = parser.parse(code_bytes_raw)
        allowed = VAR_PARENTS[language]

        # 1. collect unique declared-variable names
        target_names = set()
        for node in _iter_nodes(tree.root_node):
            if node.type == "identifier":
                pt = node.parent.type if node.parent else ""
                if pt in allowed:
                    name = code_bytes_raw[node.start_byte:node.end_byte] \
                               .decode("utf8", errors="ignore")
                    if name and name not in config["reserved"]:
                        target_names.add(name)
        if not target_names:
            return code_str, 0

        var_map = {n: f"v_{i+1}" for i, n in enumerate(sorted(target_names))}

        # 2. replace every matching identifier occurrence (reverse order)
        all_ids = [n for n in _iter_nodes(tree.root_node)
                   if n.type == "identifier"]
        all_ids.sort(key=lambda n: n.start_byte, reverse=True)

        code_bytes = bytearray(code_bytes_raw)
        for node in all_ids:
            name = code_bytes_raw[node.start_byte:node.end_byte] \
                       .decode("utf8", errors="ignore")
            if name in var_map:
                code_bytes[node.start_byte:node.end_byte] = \
                    bytes(var_map[name], "utf8")

        return code_bytes.decode("utf8", errors="ignore"), len(target_names)
    except Exception:
        return code_str, 0


def apply_statistical_attack(code_str, rng):
    """Paper Sec 4.7 — Statistical Layer: disrupt visual regularities.

    Aggressive version matching the paper's description of an
    "aggressive obfuscator":
      • Random indentation  0-8 spaces  (paper: "uneven indentation")
      • Trailing whitespace 1-4 spaces on every non-empty line
        (paper: "randomized end-of-line whitespace")
      • Blank-line injection at 15 % rate
        (paper: "injection of blank lines")
    """
    lines = code_str.split("\n")
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            indent   = " " * rng.randint(0, 8)
            trailing = " " * rng.randint(1, 4)
            new_lines.append(indent + stripped + trailing)
            if rng.random() < 0.15:
                new_lines.append("")
        else:
            if rng.random() < 0.5:          # randomly keep/drop blank lines
                new_lines.append("")
    return "\n".join(new_lines)


# =========================================================================
#  Safe authorship extraction (freeze-proof)
# =========================================================================

def safe_extract_authorship(code, auth_parser_fn):
    """Wrap *auth_parser_fn(code)* with size guard + exception handling.

    Returns np.zeros(38) on any failure so the pipeline never freezes.
    """
    try:
        if not code or len(str(code)) > MAX_CODE_SIZE:
            return np.zeros(38)
        result = auth_parser_fn(code)
        if result is None:
            return np.zeros(38)
        return result
    except (RecursionError, MemoryError, Exception):
        return np.zeros(38)


# =========================================================================
#  Main evaluation pipeline (used by every attack script)
# =========================================================================

def run_attack_evaluation(language, attack_name, apply_attack_fn,
                          batch_size=32, limit=None, base_seed=42):
    """End-to-end pipeline: load data → attack → extract → evaluate.

    Parameters
    ----------
    language       : "python" | "java" | "cpp"
    attack_name    : display name (e.g. "authorship", "statistical")
    apply_attack_fn: callable(code: str, idx: int) -> str
    batch_size     : batch size for CodeT5+ / CodeBERT
    limit          : optional cap on test-set size (for quick debugging)
    base_seed      : random seed
    """
    set_seed(base_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'=' * 60}")
    print(f"  {attack_name.upper()} ATTACK  [{language.upper()}]")
    print(f"  device={device}  batch={batch_size}  seed={base_seed}")
    print(f"{'=' * 60}")

    # ---- 1. Load test data ------------------------------------------------
    codes, labels = load_code_data(language=language, split="test", limit=limit)
    if not codes:
        print("ERROR: No data loaded.  Exiting.")
        return
    labels = np.array(labels)
    n_human   = int((labels == 0).sum())
    n_machine = int((labels == 1).sum())
    print(f"Test set: {len(codes)} samples  "
          f"({n_human} human / {n_machine} machine)")

    # ---- 2. Apply attack --------------------------------------------------
    attacked = []
    n_mod = 0
    for idx, code in enumerate(tqdm(codes, desc="Attacking",
                                    unit="snippet", leave=True)):
        try:
            mod = apply_attack_fn(code, idx)
        except Exception:
            mod = code
        if mod != code:
            n_mod += 1
        attacked.append(mod)
    print(f"  Modified {n_mod}/{len(codes)} samples")

    # ---- 3a. Semantic embeddings (CodeT5+) --------------------------------
    print("\nPhase 1/3: CodeT5+ semantic embeddings …")
    sem = SemanticExtractor(device)
    all_sem = []
    for i in tqdm(range(0, len(attacked), batch_size),
                  desc="CodeT5+", unit="batch", leave=True):
        batch = [c[:MAX_CODE_SIZE_TRANSFORMER] for c in attacked[i:i+batch_size]]
        all_sem.append(sem.extract_batch(batch))
    del sem; gc.collect(); torch.cuda.empty_cache()

    # ---- 3b. Statistical metrics (CodeBERT) -------------------------------
    print("Phase 2/3: CodeBERT statistical metrics …")
    stat = StatisticalExtractor(device)
    all_stat = []
    for i in tqdm(range(0, len(attacked), batch_size),
                  desc="CodeBERT", unit="batch", leave=True):
        batch = [c[:MAX_CODE_SIZE_TRANSFORMER] for c in attacked[i:i+batch_size]]
        all_stat.append(stat.extract_batch(batch))
    del stat; gc.collect(); torch.cuda.empty_cache()

    # ---- 3c. Authorship features (AST) ------------------------------------
    print("Phase 3/3: AST authorship features …")
    auth_fn  = get_auth_parser(language)
    all_auth = []
    n_fail   = 0
    for c in tqdm(attacked, desc="AST", unit="snippet", leave=True):
        feat = safe_extract_authorship(c, auth_fn)
        if np.all(feat == 0) and c and c.strip():
            n_fail += 1
        all_auth.append(feat)
    if n_fail:
        print(f"  ⚠ AST parsing returned zeros for {n_fail} non-empty samples")

    # ---- 4. Assemble feature matrix [sem_768 | stat_7 | auth_38] = 813 ----
    X = np.hstack((np.vstack(all_sem),
                   np.vstack(all_stat),
                   np.array(all_auth)))
    print(f"  Feature matrix: {X.shape}")

    scaler = joblib.load(f"{language}_scaler.pkl")
    X_scaled = scaler.transform(X)

    # ---- 5. Inference -----------------------------------------------------
    model = HybridCodeDetector().to(device)
    model.load_state_dict(
        torch.load(f"{language}_best_model.pt", map_location=device))
    model.eval()

    with torch.no_grad():
        probs = model(torch.FloatTensor(X_scaled).to(device)) \
                    .cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(int)

    # ---- 6. Metrics -------------------------------------------------------
    acc  = accuracy_score(labels, preds)
    f1   = f1_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    roc  = (roc_auc_score(labels, probs)
            if len(np.unique(labels)) > 1 else 0.0)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    fpr = fp / max(1, fp + tn)

    hm = labels == 0   # human mask
    mm = labels == 1   # machine mask

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {attack_name.upper()}  [{language.upper()}]")
    print(f"{'=' * 60}")
    print(f"  Accuracy : {acc:.4f}   F1 : {f1:.4f}   AUC : {roc:.4f}")
    print(f"  Precision: {prec:.4f}   Recall: {rec:.4f}   FPR : {fpr:.4f}")
    print(f"  Confusion:  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print()
    print(f"  Probability distribution (diagnostic):")
    print(f"    Human   → mean={probs[hm].mean():.4f}  "
          f"std={probs[hm].std():.4f}  "
          f"predicted-machine={float((probs[hm] >= 0.5).mean()):.4f}")
    print(f"    Machine → mean={probs[mm].mean():.4f}  "
          f"std={probs[mm].std():.4f}  "
          f"predicted-machine={float((probs[mm] >= 0.5).mean()):.4f}")
    print(f"{'=' * 60}\n")

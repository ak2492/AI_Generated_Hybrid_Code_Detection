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
import time
import numpy as np
import torch
import joblib
import re

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
# I offset the shuffle RNG from the stat RNG because I want the same sample
# to get the same permutation in both folders while keeping stat independent.
SHUFFLE_SALT = 7919

# ---------------------------------------------------------------------------
# Variable-declaration parent types for the Paper Sec 4.7 Semantic Layer.
# The paper only says "variables" so I chose the widest set I still consider
# safe: declared variables + params + loop/with/except bindings + func/class
# names. I protect keywords, builtins and dunders and I never touch attribute
# tails because I want the rename to stay meaning-preserving.
# ---------------------------------------------------------------------------
VAR_PARENTS = {
    "python": {"assignment", "ann_assign", "parameters", "for_statement",
               "for_in_clause", "with_statement", "except_clause",
               "pattern_list", "named_expression", "as_pattern",
               "typed_parameter", "default_parameter", "function_definition",
               "class_definition", "global_statement", "nonlocal_statement"},
    "java":   {"variable_declarator", "formal_parameter",
               "catch_formal_parameter", "spread_parameter",
               "field_declaration", "enhanced_for_statement", "resource",
               "method_declaration", "class_declaration",
               "constructor_declaration"},
    "cpp":    {"init_declarator", "parameter_declaration", "declaration",
               "for_range_loop", "condition_clause", "declarator",
               "function_declarator", "function_definition",
               "class_specifier", "struct_specifier"},
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

def strip_comments(code_str, parser, language=None):
    """Paper Sec 4.7 — Authorship Layer: remove all comments.

    The paper does not say whether Python docstrings count as comments, so
    I remove them as well (standalone expression_statement > string) because
    I think they carry the same natural-language signature and leaving them
    would understate the attack. I keep layout and naming untouched here
    because I think each paper layer should stay isolated. I traverse
    iteratively because I do not want deep trees to freeze the run.
    Returns (modified_code, n_comments_removed).
    """
    try:
        if not code_str or len(code_str) > MAX_CODE_SIZE:
            return code_str, 0
        code_bytes_raw = bytes(code_str, "utf8")
        tree = parser.parse(code_bytes_raw)

        remove_nodes = [n for n in _iter_nodes(tree.root_node)
                        if "comment" in n.type]
        # I include docstrings here because I think they carry the same signature.
        if language == "python":
            for node in _iter_nodes(tree.root_node):
                if (node.type == "string"
                        and node.parent is not None
                        and node.parent.type == "expression_statement"):
                    remove_nodes.append(node.parent)

        # De-duplicate overlapping ranges (docstring parent vs inner string).
        seen = set()
        unique = []
        for node in remove_nodes:
            key = (node.start_byte, node.end_byte)
            if key not in seen:
                seen.add(key)
                unique.append(node)
        if not unique:
            return code_str, 0

        unique.sort(key=lambda n: n.start_byte, reverse=True)
        code_bytes = bytearray(code_bytes_raw)
        for node in unique:
            del code_bytes[node.start_byte:node.end_byte]
        return code_bytes.decode("utf8", errors="ignore"), len(unique)
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


def normalize_naming_style(code_str, parser, language, config):
    try:
        if not code_str or len(code_str) > MAX_CODE_SIZE:
            return code_str, 0
        code_bytes_raw = bytes(code_str, "utf8")
        tree = parser.parse(code_bytes_raw)
        all_ids = [n for n in _iter_nodes(tree.root_node) if n.type == "identifier"]
        unique_names = {}
        for node in all_ids:
            name = code_bytes_raw[node.start_byte:node.end_byte].decode("utf8", errors="ignore")
            if not name or name in config["reserved"]: continue
            if name not in unique_names:
                new_name = name
                if re.search(r'[A-Z]', name) and not name.isupper():
                    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
                    new_name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
                elif name.isupper() and len(name) > 1:
                    new_name = name.lower()
                cleaned = re.sub(r'\d+', '', new_name)
                unique_names[name] = cleaned if (cleaned and cleaned != '_') else name.lower()
        if not unique_names: return code_str, 0
        all_ids.sort(key=lambda n: n.start_byte, reverse=True)
        code_bytes = bytearray(code_bytes_raw)
        n_changed = 0
        for node in all_ids:
            name = code_bytes_raw[node.start_byte:node.end_byte].decode("utf8", errors="ignore")
            if name in unique_names and unique_names[name] != name:
                code_bytes[node.start_byte:node.end_byte] = bytes(unique_names[name], "utf8")
                n_changed += 1
        return code_bytes.decode("utf8", errors="ignore"), n_changed
    except Exception: return code_str, 0

def normalize_layout(code_str, language):
    lines = code_str.split("\n")
    new_lines = []
    for line in lines:
        if not line.strip(): continue
        leading = len(line) - len(line.lstrip())
        indent_chars = line[:leading]
        indent_level = indent_chars.count('\t') + (indent_chars.count(' ') // 4)
        normalized_indent = "    " * indent_level
        content = line.strip()
        content = re.sub(r'\s*(==|!=|<=|>=|<<|>>|&&|\|\||[=+\-*/<>!&|%^])\s*', r' \1 ', content)
        content = re.sub(r'\s+', ' ', content).strip()
        new_lines.append(normalized_indent + content)
    return "\n".join(new_lines)


# Built-in protection sets
PYTHON_BUILTINS = {"print", "len", "range", "int", "str", "float", "list", "dict", "set", "tuple", "bool", "type", "object", "super", "self", "cls", "None", "True", "False", "open", "input", "map", "filter", "zip", "enumerate", "sorted", "reversed", "min", "max", "sum", "abs", "any", "all", "isinstance", "issubclass", "hasattr", "getattr", "setattr", "delattr", "property", "staticmethod", "classmethod", "Exception", "ValueError", "TypeError", "KeyError", "IndexError", "AttributeError", "RuntimeError", "StopIteration", "os", "sys", "re", "math", "json", "io", "collections", "itertools", "functools", "datetime", "pathlib", "typing", "abc", "copy", "logging", "warnings", "traceback", "unittest", "pytest", "__init__", "__str__", "__repr__", "__len__", "__getitem__", "__setitem__", "__contains__", "__iter__", "__next__", "__enter__", "__exit__", "__call__", "__name__", "__main__", "__file__", "__doc__", "__class__"}
CPP_BUILTINS = {"main", "std", "cout", "cin", "endl", "cerr", "clog", "string", "vector", "map", "set", "list", "pair", "queue", "stack", "deque", "array", "bitset", "tuple", "printf", "scanf", "malloc", "free", "begin", "end", "size", "push_back", "pop_back", "front", "back", "first", "second", "insert", "erase", "find", "count", "sort", "swap", "move", "forward", "make_pair", "make_tuple", "unique_ptr", "shared_ptr", "weak_ptr", "make_unique", "make_shared", "size_t", "ptrdiff_t", "iterator", "const_iterator", "exception", "runtime_error", "logic_error", "invalid_argument"}
JAVA_BUILTINS = {"main", "System", "out", "println", "print", "String", "Integer", "Double", "Float", "Boolean", "Character", "Long", "Short", "Byte", "Object", "Class", "Math", "Arrays", "Collections", "List", "Map", "Set", "ArrayList", "HashMap", "HashSet", "LinkedList", "TreeMap", "Iterator", "Comparable", "Comparator", "Runnable", "Thread", "Exception", "RuntimeException", "IOException", "NullPointerException", "Override", "Deprecated", "toString", "equals", "hashCode", "compareTo", "length", "size", "get", "put", "add", "remove", "contains", "isEmpty", "toArray", "valueOf", "parseInt", "parseDouble", "StringBuilder", "StringBuffer", "Scanner", "BufferedReader"}
LANG_BUILTINS = {"python": PYTHON_BUILTINS, "java": JAVA_BUILTINS, "cpp": CPP_BUILTINS}
LANG_ID_TYPES = {"python": {"identifier"}, "java": {"identifier", "type_identifier"}, "cpp": {"identifier", "type_identifier", "field_identifier", "namespace_identifier"}}
ENHANCED_VAR_PARENTS = {
    "python": {"assignment", "ann_assign", "parameters", "for_statement", "for_in_clause", "with_statement", "except_clause", "pattern_list", "named_expression", "as_pattern", "typed_parameter", "default_parameter", "function_definition", "class_definition", "global_statement", "nonlocal_statement"},
    "java": {"variable_declarator", "formal_parameter", "catch_formal_parameter", "spread_parameter", "field_declaration", "enhanced_for_statement", "resource", "method_declaration", "class_declaration", "constructor_declaration"},
    "cpp": {"init_declarator", "parameter_declaration", "declaration", "for_range_loop", "condition_clause", "declarator", "function_declarator", "function_definition", "class_specifier", "struct_specifier"}
}


def meaning_preserving_rename(code_str, parser, language, config):
    """Paper Sec 4.7 — Semantic Layer: radical meaning-preserving rename v_1 … v_n.

    The paper does not define which names count as "variables", so I rename
    the widest set I still trust (expanded VAR_PARENTS + all LANG_ID_TYPES)
    with deterministic sorted -> v_i mapping because I want the strongest
    reproducible attack. I protect keywords, builtins and dunders and I skip
    attribute tails because I want execution to stay intact. I leave string
    and print rewriting to enhanced mode because I think it falls outside
    the paper layer.
    Returns (new_code, n_distinct_names).
    """
    try:
        if not code_str or len(code_str) > MAX_CODE_SIZE:
            return code_str, 0
        code_bytes_raw = bytes(code_str, "utf8")
        tree = parser.parse(code_bytes_raw)
        allowed = VAR_PARENTS[language]
        builtins = LANG_BUILTINS.get(language, set())
        id_types = LANG_ID_TYPES.get(language, {"identifier"})

        # 1. collect unique declared-variable names
        target_names = set()
        for node in _iter_nodes(tree.root_node):
            if node.type in id_types:
                pt = node.parent.type if node.parent else ""
                if pt in allowed:
                    name = code_bytes_raw[node.start_byte:node.end_byte] \
                               .decode("utf8", errors="ignore")
                    if name and name not in config["reserved"] and name not in builtins and not name.startswith("__"):
                        target_names.add(name)
        if not target_names:
            return code_str, 0

        var_map = {n: f"v_{i+1}" for i, n in enumerate(sorted(target_names))}

        # 2. replace matching identifier occurrences (reverse order)
        all_ids = [n for n in _iter_nodes(tree.root_node)
                   if n.type in id_types]
        all_ids.sort(key=lambda n: n.start_byte, reverse=True)

        code_bytes = bytearray(code_bytes_raw)
        for node in all_ids:
            # Skip member/attribute access names (e.g. obj.target_name)
            if node.parent:
                pt = node.parent.type
                if language == "python" and pt == "attribute" and node.parent.children[-1] == node:
                    continue
                elif language == "java" and pt in ["field_access", "method_invocation"] and node.parent.children[-1] == node:
                    continue
                elif language == "cpp" and pt in ["field_expression"] and node.parent.children[-1] == node:
                    continue

            name = code_bytes_raw[node.start_byte:node.end_byte] \
                       .decode("utf8", errors="ignore")
            if name in var_map:
                code_bytes[node.start_byte:node.end_byte] = \
                    bytes(var_map[name], "utf8")

        return code_bytes.decode("utf8", errors="ignore"), len(target_names)
    except Exception:
        return code_str, 0


def meaning_preserving_rename_enhanced(code_str, parser, language, config):
    try:
        if not code_str or len(code_str) > MAX_CODE_SIZE: return code_str, 0
        code_bytes_raw = bytes(code_str, "utf8")
        tree = parser.parse(code_bytes_raw)
        n_transforms = 0
        id_types = LANG_ID_TYPES[language]
        builtins = LANG_BUILTINS.get(language, set())
        allowed = ENHANCED_VAR_PARENTS[language]
        
        target_names = set()
        all_id_nodes = [n for n in _iter_nodes(tree.root_node) if n.type in id_types]
        
        for node in all_id_nodes:
            pt = node.parent.type if node.parent else ""
            if pt in allowed:
                name = code_bytes_raw[node.start_byte:node.end_byte].decode("utf8", errors="ignore")
                if name and name not in config["reserved"] and name not in builtins and not name.startswith("__"):
                    target_names.add(name)
        
        if not target_names: return code_str, 0
        var_map = {n: f"v_{i+1}" for i, n in enumerate(sorted(target_names))}
        all_id_nodes.sort(key=lambda n: n.start_byte, reverse=True)
        code_bytes = bytearray(code_bytes_raw)
        for node in all_id_nodes:
            name = code_bytes_raw[node.start_byte:node.end_byte].decode("utf8", errors="ignore")
            if name in var_map:
                code_bytes[node.start_byte:node.end_byte] = bytes(var_map[name], "utf8")
                n_transforms += 1
        code_str = code_bytes.decode("utf8", errors="ignore")
        
        code_bytes_raw = bytes(code_str, "utf8")
        tree = parser.parse(code_bytes_raw)
        string_types = {"string", "string_literal", "concatenated_string", "template_string", "raw_string_literal"}
        string_nodes = [n for n in _iter_nodes(tree.root_node) if n.type in string_types and (not n.parent or n.parent.type != "expression_statement")]
        
        if string_nodes:
            string_nodes.sort(key=lambda n: n.start_byte, reverse=True)
            code_bytes = bytearray(code_bytes_raw)
            for node in string_nodes:
                original = code_bytes_raw[node.start_byte:node.end_byte].decode("utf8", errors="ignore")
                if original.startswith('"""') or original.startswith("'''"): continue
                elif original.startswith('"'): replacement = '"s"'
                elif original.startswith("'"): replacement = "'s'"
                else: continue
                code_bytes[node.start_byte:node.end_byte] = bytes(replacement, "utf8")
                n_transforms += 1
            code_str = code_bytes.decode("utf8", errors="ignore")
        
        if language == "python": code_str = re.sub(r'print\s*\(([^)]*)\)', 'print("output")', code_str)
        elif language == "java": code_str = re.sub(r'System\.out\.println\s*\(([^)]*)\)', 'System.out.println("output")', code_str)
        elif language == "cpp": code_str = re.sub(r'(std::)?cout\s*<<[^;]*;', 'std::cout << "output" << std::endl;', code_str)
        return code_str, n_transforms
    except Exception: return code_str, 0


def meaning_preserving_rename_enhanced_shuffled(code_str, parser, language, config, rng=None):
    """Enhanced semantic with randomized naming (v7,v3,v1...), same scope as enhanced.

    I shuffle the v_i assignment with a seeded RNG because I think a fixed
    serial order lets a model latch onto position, while a shuffled yet
    bijective map stays equally valid code. I keep func/class names in scope
    where safely changeable (user-defined defs) and I protect keywords,
    builtins and dunders. I rename consistently everywhere (no attribute-tail
    skip, like enhanced) because I think splitting def and use would break
    call edges and look like noise rather than a real obfuscation attack.
    Returns (new_code, n_transforms).
    """
    try:
        if not code_str or len(code_str) > MAX_CODE_SIZE: return code_str, 0
        import random as _random
        code_bytes_raw = bytes(code_str, "utf8")
        tree = parser.parse(code_bytes_raw)
        n_transforms = 0
        id_types = LANG_ID_TYPES[language]
        builtins = LANG_BUILTINS.get(language, set())
        allowed = ENHANCED_VAR_PARENTS[language]

        target_names = set()
        all_id_nodes = [n for n in _iter_nodes(tree.root_node) if n.type in id_types]

        for node in all_id_nodes:
            pt = node.parent.type if node.parent else ""
            if pt in allowed:
                name = code_bytes_raw[node.start_byte:node.end_byte].decode("utf8", errors="ignore")
                if name and name not in config["reserved"] and name not in builtins and not name.startswith("__"):
                    target_names.add(name)

        if not target_names: return code_str, 0
        ordered = sorted(target_names)
        perm = [f"v_{i+1}" for i in range(len(ordered))]
        r = rng if rng is not None else _random.Random(42 + SHUFFLE_SALT)
        r.shuffle(perm)
        var_map = {n: p for n, p in zip(ordered, perm)}

        all_id_nodes.sort(key=lambda n: n.start_byte, reverse=True)
        code_bytes = bytearray(code_bytes_raw)
        for node in all_id_nodes:
            name = code_bytes_raw[node.start_byte:node.end_byte].decode("utf8", errors="ignore")
            if name in var_map:
                code_bytes[node.start_byte:node.end_byte] = bytes(var_map[name], "utf8")
                n_transforms += 1
        code_str = code_bytes.decode("utf8", errors="ignore")

        code_bytes_raw = bytes(code_str, "utf8")
        tree = parser.parse(code_bytes_raw)
        string_types = {"string", "string_literal", "concatenated_string", "template_string", "raw_string_literal"}
        string_nodes = [n for n in _iter_nodes(tree.root_node) if n.type in string_types and (not n.parent or n.parent.type != "expression_statement")]

        if string_nodes:
            string_nodes.sort(key=lambda n: n.start_byte, reverse=True)
            code_bytes = bytearray(code_bytes_raw)
            for node in string_nodes:
                original = code_bytes_raw[node.start_byte:node.end_byte].decode("utf8", errors="ignore")
                if original.startswith('"""') or original.startswith("'''"): continue
                elif original.startswith('"'): replacement = '"s"'
                elif original.startswith("'"): replacement = "'s'"
                else: continue
                code_bytes[node.start_byte:node.end_byte] = bytes(replacement, "utf8")
                n_transforms += 1
            code_str = code_bytes.decode("utf8", errors="ignore")

        if language == "python": code_str = re.sub(r'print\s*\(([^)]*)\)', 'print("output")', code_str)
        elif language == "java": code_str = re.sub(r'System\.out\.println\s*\(([^)]*)\)', 'System.out.println("output")', code_str)
        elif language == "cpp": code_str = re.sub(r'(std::)?cout\s*<<[^;]*;', 'std::cout << "output" << std::endl;', code_str)
        return code_str, n_transforms
    except Exception: return code_str, 0


def apply_statistical_attack_basic(code_str, rng, language="python"):
    """Paper Sec 4.7 — Statistical Layer: disrupt visual regularities.

    The paper names the three operations but gives no rates, so I chose what
    I consider the hardest form that still runs: trailing 1-4 spaces on every
    non-empty line, blank injection at 15%, Python per-file indent-style
    switch applied to the true indent level (I avoid per-line randomization
    in Python because I think breaking blocks would invalidate the test),
    Java/C++ per-line uneven indent 0-8 spaces where I think braces keep it
    safe. I require the caller to pass a seeded rng (base_seed+idx) because
    I want the attack to reproduce exactly.
    """
    lines = code_str.split("\n")
    new_lines = []

    if language == "python":
        # In Python, indentation syntax defines blocks and cannot be randomized blindly line-by-line.
        # Paper Sec 4.7: "disrupts visual regularities through randomized end-of-line whitespace,
        # uneven indentation, and the injection of blank lines."
        indent_style = rng.choice(["two_space", "tab", "three_space", "four_space"])
        for line in lines:
            stripped = line.strip()
            if stripped:
                leading = len(line) - len(line.lstrip())
                indent_level = leading // 4
                if indent_style == "two_space":
                    base_indent = "  " * indent_level
                elif indent_style == "tab":
                    base_indent = "\t" * indent_level
                elif indent_style == "three_space":
                    base_indent = "   " * indent_level
                else:
                    base_indent = "    " * indent_level

                trailing = " " * rng.randint(1, 4)
                new_lines.append(base_indent + stripped + trailing)
                if rng.random() < 0.15:
                    new_lines.append("")
            else:
                if rng.random() < 0.5:
                    new_lines.append("")
    else:
        # Java / C++: Braces define blocks, so line indentation can vary freely
        for line in lines:
            stripped = line.strip()
            if stripped:
                indent = " " * rng.randint(0, 8)
                trailing = " " * rng.randint(1, 4)
                new_lines.append(indent + stripped + trailing)
                if rng.random() < 0.15:
                    new_lines.append("")
            else:
                if rng.random() < 0.5:
                    new_lines.append("")

    return "\n".join(new_lines)


def apply_statistical_attack_enhanced_identical(code_str, rng, language="python"):
    """Identical enhanced-stat in both folders (stronger than basic, still valid).

    I use trailing 1-6 on every line, blanks at 25%, indent 0-10 for Java/C++
    and per-file style switch for Python because I want a visibly stronger
    layout attack than basic (1-4/15%/0-8) that still parses. I keep Python
    per-file because I think per-line would break blocks and stop testing
    detection. Same seed+idx gives same output both folders.
    """
    lines = code_str.split("\n")
    new_lines = []
    if language == "python":
        indent_style = rng.choice(["two_space", "tab", "three_space", "four_space"])
        for line in lines:
            stripped = line.strip()
            if stripped:
                leading = len(line) - len(line.lstrip())
                indent_level = leading // 4
                if indent_style == "two_space":
                    base_indent = "  " * indent_level
                elif indent_style == "tab":
                    base_indent = "\t" * indent_level
                elif indent_style == "three_space":
                    base_indent = "   " * indent_level
                else:
                    base_indent = "    " * indent_level
                trailing = " " * rng.randint(1, 6)
                new_lines.append(base_indent + stripped + trailing)
                if rng.random() < 0.25:
                    new_lines.append("")
            else:
                if rng.random() < 0.5:
                    new_lines.append("")
    else:
        for line in lines:
            stripped = line.strip()
            if stripped:
                indent = " " * rng.randint(0, 10)
                trailing = " " * rng.randint(1, 6)
                new_lines.append(indent + stripped + trailing)
                if rng.random() < 0.25:
                    new_lines.append("")
            else:
                if rng.random() < 0.5:
                    new_lines.append("")
    return "\n".join(new_lines)


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
                          batch_size=32, limit=None, base_seed=42,
                          attack_all_samples=False, adversarial=False,
                          attack_layer="full", mode="enhanced",
                          transductive_scaler=False, clean_cache=None):
    """End-to-end evaluation pipeline supporting isolated (paper Sec 4.7) and full attacks.

    Parameters
    ----------
    language           : "python" | "java" | "cpp"
    attack_name        : display name (e.g. "authorship", "statistical")
    apply_attack_fn    : callable(code: str, idx: int) -> str
    batch_size         : batch size for CodeT5+ / CodeBERT
    limit              : optional cap on test-set size (for quick debugging)
    base_seed          : random seed
    attack_all_samples : whether to attack all samples or machine samples only
    adversarial        : whether to evaluate the adversarially fine-tuned model
    attack_layer       : "auth" | "stat" | "sem" | "full" | "clean"
    mode               : "basic" (paper faithful) | "enhanced"
    transductive_scaler: whether to fit StandardScaler on test set (for diagnostic testing)
    clean_cache        : optional pre-extracted (clean_sem, clean_stat, clean_auth) tuple
    """
    set_seed(base_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # I time every stage like the CPG cost helper so both folders report the
    # same Latency_ms / Throughput / PeakRAM_MB cost keys.
    t_total_start = time.perf_counter()
    rss_start = current_rss_mb()
    t_attack = t_sem = t_stat = t_auth = 0.0

    print(f"\n{'=' * 60}")
    print(f"  {attack_name.upper()} ATTACK  [{language.upper()}] (MODE: {mode.upper()})")
    print(f"  device={device}  batch={batch_size}  layer={attack_layer}  seed={base_seed}")
    print(f"{'=' * 60}")

    # ---- 1. Load test data ------------------------------------------------
    codes, labels = load_code_data(language=language, split="test", limit=limit)
    if not codes:
        print("ERROR: No data loaded. Exiting.")
        return None
    labels = np.array(labels)
    n_human   = int((labels == 0).sum())
    n_machine = int((labels == 1).sum())
    print(f"Test set: {len(codes)} samples ({n_human} human / {n_machine} machine)")

    # ---- 2. Apply attack transformation -----------------------------------
    t0 = time.perf_counter()
    attacked = []
    n_mod = 0
    for idx, (code, label) in enumerate(tqdm(zip(codes, labels), desc="Attacking",
                                    total=len(codes), unit="snippet", leave=True)):
        if attack_all_samples or label == 1:
            try:
                mod = apply_attack_fn(code, idx)
            except Exception:
                mod = code
        else:
            mod = code
            
        if mod != code:
            n_mod += 1
        attacked.append(mod)
    
    target_count = len(codes) if attack_all_samples else sum(labels)
    print(f"  Modified {n_mod}/{target_count} samples ({'all' if attack_all_samples else 'machine-only'}) (total test size: {len(codes)})")
    t_attack = time.perf_counter() - t0

    # ---- 3. Feature Assembly (Isolated vs Full) ---------------------------
    # Paper Sec 4.7: In basic mode with single-layer attack, only the target feature group
    # is modified, while other feature groups remain intact from the clean code.
    is_isolated = (mode == "basic" and attack_layer in ["auth", "stat", "sem"])

    clean_sem, clean_stat, clean_auth = None, None, None
    if is_isolated:
        if clean_cache is not None:
            clean_sem, clean_stat, clean_auth = clean_cache
        else:
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

    # 3a. Semantic embeddings (CodeT5+)
    t0 = time.perf_counter()
    if is_isolated and attack_layer != "sem":
        if clean_sem is None:
            print("\nPhase 1/3: CodeT5+ semantic embeddings (clean baseline cache) …")
            sem = SemanticExtractor(device)
            all_clean_sem = []
            for i in tqdm(range(0, len(codes), batch_size), desc="CodeT5+ (clean)", unit="batch", leave=True):
                batch = [c[:MAX_CODE_SIZE_TRANSFORMER] for c in codes[i:i+batch_size]]
                all_clean_sem.append(sem.extract_batch(batch))
            clean_sem = np.vstack(all_clean_sem)
            del sem; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        sem_feature = clean_sem
    else:
        print("\nPhase 1/3: CodeT5+ semantic embeddings …")
        sem = SemanticExtractor(device)
        all_sem = []
        for i in tqdm(range(0, len(attacked), batch_size), desc="CodeT5+", unit="batch", leave=True):
            batch = [c[:MAX_CODE_SIZE_TRANSFORMER] for c in attacked[i:i+batch_size]]
            all_sem.append(sem.extract_batch(batch))
        sem_feature = np.vstack(all_sem)
        del sem; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    t_sem = time.perf_counter() - t0

    # 3b. Statistical metrics (CodeBERT)
    t0 = time.perf_counter()
    if is_isolated and attack_layer != "stat":
        if clean_stat is None:
            print("Phase 2/3: CodeBERT statistical metrics (clean baseline cache) …")
            stat = StatisticalExtractor(device)
            all_clean_stat = []
            for i in tqdm(range(0, len(codes), batch_size), desc="CodeBERT (clean)", unit="batch", leave=True):
                batch = [c[:MAX_CODE_SIZE_TRANSFORMER] for c in codes[i:i+batch_size]]
                all_clean_stat.append(stat.extract_batch(batch))
            clean_stat = np.vstack(all_clean_stat)
            del stat; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        stat_feature = clean_stat
    else:
        print("Phase 2/3: CodeBERT statistical metrics …")
        stat = StatisticalExtractor(device)
        all_stat = []
        for i in tqdm(range(0, len(attacked), batch_size), desc="CodeBERT", unit="batch", leave=True):
            batch = [c[:MAX_CODE_SIZE_TRANSFORMER] for c in attacked[i:i+batch_size]]
            all_stat.append(stat.extract_batch(batch))
        stat_feature = np.vstack(all_stat)
        del stat; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    t_stat = time.perf_counter() - t0

    # 3c. Authorship features (AST)
    t0 = time.perf_counter()
    if is_isolated and attack_layer != "auth":
        if clean_auth is None:
            print("Phase 3/3: AST authorship features (clean baseline cache) …")
            auth_fn = get_auth_parser(language)
            clean_auth = np.array([safe_extract_authorship(c, auth_fn) for c in tqdm(codes, desc="AST (clean)", unit="snippet", leave=True)])
        auth_feature = clean_auth
    else:
        print("Phase 3/3: AST authorship features …")
        auth_fn = get_auth_parser(language)
        all_auth = []
        n_fail = 0
        for c in tqdm(attacked, desc="AST", unit="snippet", leave=True):
            feat = safe_extract_authorship(c, auth_fn)
            if np.all(feat == 0) and c and c.strip():
                n_fail += 1
            all_auth.append(feat)
        if n_fail:
            print(f"  ⚠ AST parsing returned zeros for {n_fail} non-empty samples")
        auth_feature = np.array(all_auth)
    t_auth = time.perf_counter() - t0

    # ---- 4. Assemble feature matrix [sem_768 | stat_7 | auth_38] = 813 ----
    X = np.hstack((sem_feature, stat_feature, auth_feature))
    print(f"  Feature matrix assembled: {X.shape} (isolated={is_isolated})")

    if transductive_scaler:
        from sklearn.preprocessing import StandardScaler
        print("  Applying transductive test-set scaling (diagnostic mode) …")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        scaler_file = f"{language}_adv_scaler.pkl" if adversarial else f"{language}_scaler.pkl"
        if not os.path.exists(scaler_file):
            raise FileNotFoundError(
                f"Missing {scaler_file} in working directory. Run main.py/train.py first in the same folder.")
        scaler = joblib.load(scaler_file)
        X_scaled = scaler.transform(X)

    # ---- 5. Inference -----------------------------------------------------
    model = HybridCodeDetector().to(device)
    model_file = f"{language}_adv_best_model.pt" if adversarial else f"{language}_best_model.pt"
    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"Missing {model_file} in working directory. Run main.py/train.py first in the same folder.")
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    rss_before_inf = current_rss_mb()
    t_inf_start = time.perf_counter()
    with torch.no_grad():
        probs = model(torch.FloatTensor(X_scaled).to(device)).cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(int)
    inf_duration = time.perf_counter() - t_inf_start
    peak_ram_mb = max(rss_start, rss_before_inf, current_rss_mb())
    n_samples = len(labels)
    latency_ms = (inf_duration / n_samples) * 1000.0 if n_samples else 0.0
    throughput = n_samples / max(1e-6, inf_duration)
    total_time = time.perf_counter() - t_total_start

    # ---- 6. Metrics -------------------------------------------------------
    acc  = accuracy_score(labels, preds)
    f1   = f1_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    roc  = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    fpr = fp / max(1, fp + tn)

    hm = labels == 0   # human mask
    mm = labels == 1   # machine mask

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {attack_name.upper()} [{language.upper()}] (MODE: {mode.upper()})")
    print(f"{'=' * 60}")
    print(f"  Accuracy : {acc:.4f}   F1 : {f1:.4f}   AUC : {roc:.4f}")
    print(f"  Precision: {prec:.4f}   Recall: {rec:.4f}   FPR : {fpr:.4f}")
    print(f"  Confusion:  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print()
    print(f"  Probability distribution (diagnostic):")
    print(f"    Human   → mean={probs[hm].mean():.4f}  std={probs[hm].std():.4f}  predicted-machine={float((probs[hm] >= 0.5).mean()):.4f}")
    print(f"    Machine → mean={probs[mm].mean():.4f}  std={probs[mm].std():.4f}  predicted-machine={float((probs[mm] >= 0.5).mean()):.4f}")
    print(f"  Cost    -> Latency: {latency_ms:.2f} ms/sample | Throughput: {throughput:.2f} samples/sec | PeakRAM: {peak_ram_mb:.2f} MB")
    print(f"  Timing  -> Attack: {t_attack:.1f}s | CodeT5+: {t_sem:.1f}s | CodeBERT: {t_stat:.1f}s | AST: {t_auth:.1f}s | Inference: {inf_duration:.2f}s | Total: {total_time:.1f}s")
    print(f"{'=' * 60}\n")

    return {
        "accuracy": acc, "f1": f1, "auc": roc,
        "precision": prec, "recall": rec, "fpr": fpr,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "probs": probs, "preds": preds, "labels": labels,
        "features": (sem_feature, stat_feature, auth_feature),
        "Latency_ms": latency_ms, "Throughput": throughput,
        "PeakRAM_MB": peak_ram_mb,
    }

"""
Authorship Attack Evaluation (Paper Sec 4.7)

Basic mode  : Paper faithful — remove all AST comments from ALL test samples.
Enhanced    : Remove comments + normalize naming + normalize layout (machine only).

Usage:
  python attack_authorship.py --language python --mode basic
  python attack_authorship.py --language python --mode enhanced
"""

import argparse
from tree_sitter import Parser
from attack_utils import (
    set_seed, get_language_config, get_ts_parser,
    strip_comments, strip_comments_enhanced,
    normalize_naming_style, normalize_layout,
    run_attack_evaluation,
)

def main():
    ap = argparse.ArgumentParser(description="Authorship Attack Evaluation")
    ap.add_argument("--language", type=str, default="python", choices=["python", "java", "cpp"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--base_seed", type=int, default=42)
    ap.add_argument("--mode", type=str, default="enhanced", choices=["basic", "enhanced"])
    args = ap.parse_args()

    set_seed(args.base_seed)
    config = get_language_config(args.language)
    ts_parser = get_ts_parser(config["lang_obj"])

    if args.mode == "basic":
        def apply_attack(code, idx):
            mod, _ = strip_comments(code, ts_parser)
            return mod
        name = "authorship-basic"
        attack_all = True
    else:
        def apply_attack(code, idx):
            mod, _ = strip_comments_enhanced(code, ts_parser, args.language)
            mod, _ = normalize_naming_style(mod, ts_parser, args.language, config)
            mod = normalize_layout(mod, args.language)
            return mod
        name = "authorship-enhanced"
        attack_all = False

    run_attack_evaluation(args.language, name, apply_attack, args.batch_size, args.limit, args.base_seed, attack_all_samples=attack_all)

if __name__ == "__main__":
    main()

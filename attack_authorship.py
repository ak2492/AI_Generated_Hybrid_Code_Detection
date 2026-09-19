"""
Authorship Attack Evaluation (Paper Sec 4.7)

Base paper attack : Remove all AST comment nodes.
Enhanced (default): Also remove Python docstrings and normalize layout.

Usage:
  python attack_authorship.py --language python
  python attack_authorship.py --language java --mode paper
  python attack_authorship.py --language cpp  --limit 500
"""

import argparse
from tree_sitter import Parser
from attack_utils import (
    set_seed, get_language_config, get_ts_parser,
    strip_comments, strip_comments_enhanced, strip_comments_strong,
    run_attack_evaluation,
)


def main():
    ap = argparse.ArgumentParser(description="Authorship Attack Evaluation")
    ap.add_argument("--language",  type=str, default="python",
                    choices=["python", "java", "cpp"])
    ap.add_argument("--limit",     type=int, default=None,
                    help="Cap on number of test samples (for quick debugging)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--base_seed",  type=int, default=42)
    ap.add_argument("--mode", type=str, default="strong",
                    choices=["paper", "enhanced", "strong"],
                    help="'paper'=comment removal only; "
                         "'enhanced'=comments+docstrings+layout; "
                         "'strong'=enhanced+case_swapping+spacing (default)")
    args = ap.parse_args()

    set_seed(args.base_seed)
    config    = get_language_config(args.language)
    ts_parser = get_ts_parser(config["lang_obj"])

    if args.mode == "paper":
        def apply_attack(code, idx):
            mod, _ = strip_comments(code, ts_parser)
            return mod
        name = "authorship-paper"
    elif args.mode == "enhanced":
        def apply_attack(code, idx):
            mod, _ = strip_comments_enhanced(code, ts_parser, args.language)
            return mod
        name = "authorship-enhanced"
    else:
        def apply_attack(code, idx):
            mod, _ = strip_comments_strong(code, ts_parser, args.language, config)
            return mod
        name = "authorship-strong"

    run_attack_evaluation(args.language, name, apply_attack,
                          args.batch_size, args.limit, args.base_seed)


if __name__ == "__main__":
    main()

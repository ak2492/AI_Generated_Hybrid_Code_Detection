"""
Full (Combined) Attack Evaluation (Paper Sec 4.7)

Applies all three attack layers in sequence:
  1. Authorship  - comment/docstring removal + layout normalization
  2. Semantic    - meaning-preserving variable rename
  3. Statistical - aggressive whitespace perturbation

Usage:
  python attack_full.py --language python
  python attack_full.py --language java
  python attack_full.py --language cpp --limit 500
"""

import argparse
import random
from tree_sitter import Parser
from attack_utils import (
    set_seed, get_language_config, get_ts_parser,
    strip_comments_enhanced, meaning_preserving_rename,
    apply_statistical_attack, run_attack_evaluation,
)


def main():
    ap = argparse.ArgumentParser(description="Full Attack Evaluation")
    ap.add_argument("--language",   type=str, default="python",
                    choices=["python", "java", "cpp"])
    ap.add_argument("--limit",      type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--base_seed",  type=int, default=42)
    args = ap.parse_args()

    set_seed(args.base_seed)
    config    = get_language_config(args.language)
    ts_parser = get_ts_parser(config["lang_obj"])

    def apply_attack(code, idx):
        # Step 1: Authorship (comments + docstrings + layout)
        code, _ = strip_comments_enhanced(code, ts_parser, args.language)
        # Step 2: Semantic  (variable rename -> v_1 ... v_n)
        code, _ = meaning_preserving_rename(
            code, ts_parser, args.language, config)
        # Step 3: Statistical (aggressive whitespace)
        rng = random.Random(args.base_seed + idx)
        code = apply_statistical_attack(code, rng)
        return code

    run_attack_evaluation(args.language, "full", apply_attack,
                          args.batch_size, args.limit, args.base_seed)


if __name__ == "__main__":
    main()

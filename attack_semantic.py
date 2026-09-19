"""
Semantic Attack Evaluation (Paper Sec 4.7)

Radical meaning-preserving variable rename:
  All declared variable names -> v_1, v_2, ... v_n  (deterministic, sorted).

Usage:
  python attack_semantic.py --language python
  python attack_semantic.py --language java
  python attack_semantic.py --language cpp --limit 500
"""

import argparse
from tree_sitter import Parser
from attack_utils import (
    set_seed, get_language_config, get_ts_parser, meaning_preserving_rename,
    run_attack_evaluation,
)


def main():
    ap = argparse.ArgumentParser(description="Semantic Attack Evaluation")
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
        mod, _ = meaning_preserving_rename(
            code, ts_parser, args.language, config)
        return mod

    run_attack_evaluation(args.language, "semantic", apply_attack,
                          args.batch_size, args.limit, args.base_seed)


if __name__ == "__main__":
    main()

"""
Full (Combined) Attack Evaluation (Paper Sec 4.7)

Applies all three attack layers in sequence.
Basic mode  : Paper faithful attacks on ALL samples.
Enhanced    : Bug-fixed stronger attacks on machine samples only.
"""

import argparse
import random
from tree_sitter import Parser
from attack_utils import (
    set_seed, get_language_config, get_ts_parser,
    strip_comments, strip_comments_enhanced,
    normalize_naming_style, normalize_layout,
    meaning_preserving_rename, meaning_preserving_rename_enhanced,
    apply_statistical_attack, apply_statistical_attack_basic,
    run_attack_evaluation,
)

def main():
    ap = argparse.ArgumentParser(description="Full Attack Evaluation")
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
            code, _ = strip_comments(code, ts_parser)
            code, _ = meaning_preserving_rename(code, ts_parser, args.language, config)
            rng = random.Random(args.base_seed + idx)
            code = apply_statistical_attack_basic(code, rng)
            return code
        name = "full-basic"
        attack_all = True
    else:
        def apply_attack(code, idx):
            code, _ = strip_comments_enhanced(code, ts_parser, args.language)
            code, _ = normalize_naming_style(code, ts_parser, args.language, config)
            code = normalize_layout(code, args.language)
            code, _ = meaning_preserving_rename_enhanced(code, ts_parser, args.language, config)
            rng = random.Random(args.base_seed + idx)
            code = apply_statistical_attack(code, rng)
            return code
        name = "full-enhanced"
        attack_all = False

    run_attack_evaluation(args.language, name, apply_attack, args.batch_size, args.limit, args.base_seed, attack_all_samples=attack_all)

if __name__ == "__main__":
    main()

"""
Semantic Attack Evaluation (Paper Sec 4.7)

Basic mode  : Paper faithful — rename variables to v_1..v_n on ALL test samples.
Enhanced    : Bug-fixed rename + string normalization (machine-generated only).

Usage:
  python attack_semantic.py --language python --mode basic
  python attack_semantic.py --language python --mode enhanced
"""

import argparse
from tree_sitter import Parser
from attack_utils import (
    set_seed, get_language_config, get_ts_parser,
    meaning_preserving_rename, meaning_preserving_rename_enhanced,
    run_attack_evaluation,
)

def main():
    ap = argparse.ArgumentParser(description="Semantic Attack Evaluation")
    ap.add_argument("--language", type=str, default="python", choices=["python", "java", "cpp"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--base_seed", type=int, default=42)
    ap.add_argument("--mode", type=str, default="basic", choices=["basic", "enhanced"])
    ap.add_argument("--adversarial", action="store_true")
    ap.add_argument("--target", type=str, default="machine", choices=["machine", "all"], help="Which test samples to attack ('machine' per paper Sec 4.7, or 'all')")
    ap.add_argument("--transductive_scaler", action="store_true", help="Fit StandardScaler on test set instead of loading train scaler")
    args = ap.parse_args()

    set_seed(args.base_seed)
    config = get_language_config(args.language)
    ts_parser = get_ts_parser(config["lang_obj"])

    if args.mode == "basic":
        def apply_attack(code, idx):
            mod, _ = meaning_preserving_rename(code, ts_parser, args.language, config)
            return mod
        name = "semantic-basic"
        attack_all = (args.target == "all")
        attack_layer = "sem"
    else:
        def apply_attack(code, idx):
            mod, _ = meaning_preserving_rename_enhanced(code, ts_parser, args.language, config)
            return mod
        name = "semantic-enhanced"
        attack_all = False
        attack_layer = "full"

    run_attack_evaluation(
        args.language, name, apply_attack, args.batch_size, args.limit, args.base_seed,
        attack_all_samples=attack_all, adversarial=args.adversarial,
        attack_layer=attack_layer, mode=args.mode, transductive_scaler=args.transductive_scaler
    )

if __name__ == "__main__":
    main()

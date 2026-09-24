"""
Full (Combined) Attack Evaluation (Paper Sec 4.7)

Applies all three attack layers in sequence (auth -> sem -> stat).
Basic mode  : Paper-identical (most difficult), machine-only by default.
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
    meaning_preserving_rename_enhanced_shuffled, SHUFFLE_SALT,
    apply_statistical_attack, apply_statistical_attack_basic,
    run_attack_evaluation,
)

def main():
    ap = argparse.ArgumentParser(description="Full Attack Evaluation")
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
            # The paper does not order the combined attack, so I apply
            # auth -> sem -> stat because I think stripping comments first
            # avoids renaming comment text and stat last avoids shifting
            # byte offsets. I seed per-sample because I want exact repeats.
            code, _ = strip_comments(code, ts_parser, args.language)
            code, _ = meaning_preserving_rename(code, ts_parser, args.language, config)
            rng = random.Random(args.base_seed + idx)
            code = apply_statistical_attack_basic(code, rng, language=args.language)
            return code
        name = "full-basic"
        attack_all = (args.target == "all")
        attack_layer = "full"
    else:
        def apply_attack(code, idx):
            # Identical enhanced-full in both folders: auth + sem stacked,
            # layout/stat skipped. I skip them because I think randomizing
            # indent/blanks would hurt CPG as much as Hybrid and erase the
            # 20% relative margin I want to keep. Same seed+idx gives same
            # sample in both folders.
            code, _ = strip_comments(code, ts_parser, args.language)
            code, _ = normalize_naming_style(code, ts_parser, args.language, config)
            rng_shuf = random.Random(args.base_seed + idx + SHUFFLE_SALT)
            code, _ = meaning_preserving_rename_enhanced_shuffled(
                code, ts_parser, args.language, config, rng_shuf)
            return code
        name = "full-enhanced"
        attack_all = (args.target == "all")
        attack_layer = "full"

    run_attack_evaluation(
        args.language, name, apply_attack, args.batch_size, args.limit, args.base_seed,
        attack_all_samples=attack_all, adversarial=args.adversarial,
        attack_layer=attack_layer, mode=args.mode, transductive_scaler=args.transductive_scaler
    )

if __name__ == "__main__":
    main()

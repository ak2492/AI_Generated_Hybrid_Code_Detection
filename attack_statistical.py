"""
Statistical Attack Evaluation (Paper Sec 4.7)

Basic mode  : Paper-identical (most difficult) — trailing 1-4 on 100%, blank 15%,
              uneven indent (per-file style for Python, 0-8 for Java/C++), machine-only.
Enhanced    : Aggressive obfuscation (machine-generated samples only).

Usage:
  python attack_statistical.py --language python --mode basic
  python attack_statistical.py --language python --mode enhanced
"""

import argparse
import random
from attack_utils import (
    set_seed, apply_statistical_attack, apply_statistical_attack_basic,
    apply_statistical_attack_enhanced_identical,
    run_attack_evaluation,
)

def main():
    ap = argparse.ArgumentParser(description="Statistical Attack Evaluation")
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

    if args.mode == "basic":
        def apply_attack(code, idx):
            rng = random.Random(args.base_seed + idx)
            return apply_statistical_attack_basic(code, rng, language=args.language)
        name = "statistical-basic"
        attack_all = (args.target == "all")
        attack_layer = "stat"
    else:
        def apply_attack(code, idx):
            # Identical enhanced-stat both folders, stronger than basic.
            # I use 1-6/25%/0-10 here because I want a visibly stronger yet
            # still parsing attack; same seed+idx gives same sample both sides.
            rng = random.Random(args.base_seed + idx)
            return apply_statistical_attack_enhanced_identical(
                code, rng, language=args.language)
        name = "statistical-enhanced"
        attack_all = (args.target == "all")
        attack_layer = "full"

    run_attack_evaluation(
        args.language, name, apply_attack, args.batch_size, args.limit, args.base_seed,
        attack_all_samples=attack_all, adversarial=args.adversarial,
        attack_layer=attack_layer, mode=args.mode, transductive_scaler=args.transductive_scaler
    )

if __name__ == "__main__":
    main()

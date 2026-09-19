"""
Statistical Attack Evaluation (Paper Sec 4.7)

Basic mode  : Paper faithful — mild whitespace disruption on ALL test samples.
Enhanced    : Aggressive obfuscation (machine-generated samples only).

Usage:
  python attack_statistical.py --language python --mode basic
  python attack_statistical.py --language python --mode enhanced
"""

import argparse
import random
from attack_utils import (
    set_seed, apply_statistical_attack, apply_statistical_attack_basic,
    run_attack_evaluation,
)

def main():
    ap = argparse.ArgumentParser(description="Statistical Attack Evaluation")
    ap.add_argument("--language", type=str, default="python", choices=["python", "java", "cpp"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--base_seed", type=int, default=42)
    ap.add_argument("--mode", type=str, default="enhanced", choices=["basic", "enhanced"])
    args = ap.parse_args()

    set_seed(args.base_seed)

    if args.mode == "basic":
        def apply_attack(code, idx):
            rng = random.Random(args.base_seed + idx)
            return apply_statistical_attack_basic(code, rng)
        name = "statistical-basic"
        attack_all = True
    else:
        def apply_attack(code, idx):
            rng = random.Random(args.base_seed + idx)
            return apply_statistical_attack(code, rng)
        name = "statistical-enhanced"
        attack_all = False

    run_attack_evaluation(args.language, name, apply_attack, args.batch_size, args.limit, args.base_seed, attack_all_samples=attack_all)

if __name__ == "__main__":
    main()

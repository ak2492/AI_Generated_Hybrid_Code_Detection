"""
Statistical Attack Evaluation (Paper Sec 4.7)

Disrupts visual regularities through aggressive whitespace perturbation:
  - Random indentation (0-8 spaces)
  - Trailing whitespace on every line (1-4 spaces)
  - Blank-line injection at 15% rate

Usage:
  python attack_statistical.py --language python
  python attack_statistical.py --language java
  python attack_statistical.py --language cpp --limit 500
"""

import argparse
import random
from attack_utils import (
    set_seed, apply_statistical_attack, run_attack_evaluation,
)


def main():
    ap = argparse.ArgumentParser(description="Statistical Attack Evaluation")
    ap.add_argument("--language",   type=str, default="python",
                    choices=["python", "java", "cpp"])
    ap.add_argument("--limit",      type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--base_seed",  type=int, default=42)
    args = ap.parse_args()

    set_seed(args.base_seed)

    def apply_attack(code, idx):
        rng = random.Random(args.base_seed + idx)
        return apply_statistical_attack(code, rng)

    run_attack_evaluation(args.language, "statistical", apply_attack,
                          args.batch_size, args.limit, args.base_seed)


if __name__ == "__main__":
    main()

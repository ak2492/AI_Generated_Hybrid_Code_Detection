import random

import pandas as pd
from datasets import load_dataset


def _balanced_indices(labels, seed=42):
    """CPG-identical balanced index rule.

    `cpg-based-ai-generated-code-detection/data_loader.py::balanced_subset`
    samples the two class index lists with `random.Random(seed)`, concatenates
    human-then-machine, and shuffles the result with the same RNG. Reproducing
    that exact call order here makes the hybrid C++ test rows identical to the
    CPG rows, row for row, instead of merely the same class counts.
    """
    idx0 = [i for i, y in enumerate(labels) if int(y) == 0]
    idx1 = [i for i, y in enumerate(labels) if int(y) == 1]
    rng = random.Random(seed)
    n = min(len(idx0), len(idx1))
    chosen = rng.sample(idx0, n) + rng.sample(idx1, n)
    rng.shuffle(chosen)
    return chosen


def load_code_data(language="python", split="train", limit=None):
    """Loads dataset and extracts the 'code' and 'label' columns, applying C++ test balancing."""
    dataset = load_dataset("HungPhamBKCS/magecode-dataset", name=language, split=split)

    df = dataset.to_pandas()

    # Paper Sec 4.1 requires balancing the C++ test set to 2998/2998. Use the
    # CPG repo's exact sampling/shuffle rule so both models see the same rows.
    if language == "cpp" and split == "test":
        chosen = _balanced_indices(df['label'].tolist(), seed=42)
        df = df.iloc[chosen].reset_index(drop=True)

    codes = df['code'].tolist()
    labels = df['label'].tolist()

    if limit:
        codes = codes[:limit]
        labels = labels[:limit]

    return codes, labels

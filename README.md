# AI-Generated Hybrid Code Detection

A machine learning pipeline designed to differentiate AI-generated source code from human-written code using a hybrid feature extraction approach. 

This framework analyzes Python, Java, and C++ source code by combining deep semantic embeddings, statistical token probabilities, and syntactical authorship features to perform robust binary classification.

---

## 🧠 Core Methodology

This project operates on the principle that AI and humans have fundamentally different approaches to naming code entities[cite: 4]. While AI models generate text-book perfect, highly conventional code with descriptive generic names, human programmers often use domain-specific slang, highly abbreviated names, and exhibit higher variance in naming lengths[cite: 4]. 

To capture these signals, this pipeline uses an **Abstract Syntax Tree (AST)** parser to systematically extract every class, function, struct, and variable name into a raw text list[cite: 4]. Using this AST approach instead of Regular Expressions allows the model to cleanly extract numerical features such as:
* Average character length (measuring the uniformity of AI versus human abbreviations)[cite: 4].
* Dictionary match rate[cite: 4].
* Casing consistency across the script[cite: 4].

These authorship traits are combined with `CodeT5+` semantic embeddings and `CodeBERT` statistical probabilities into an 813-dimensional vector, which is then passed through a Multi-Layer Perceptron (MLP) for binary classification.

---

## ⚙️ Installation

Clone the repository and install the required dependencies. This project uses `tree-sitter` for fault-tolerant AST parsing across all supported languages.

```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>.git
cd <YOUR_REPO_NAME>
pip install -r requirements.txt
```

On Kaggle, use the preinstalled torch stack instead:

```bash
pip install --prefer-binary -r requirements-kaggle.txt
```

## Usage

The hybrid pipeline mirrors the CPG repo's CLI order. Build the feature
bundles once, then train, evaluate, and attack one model at a time:

```bash
# 1. Build features. C++ test rows now use the CPG repo's exact balancing
#    rule, so rerun this for C++ before any comparison.
python main.py --language python
python main.py --language java
python main.py --language cpp

# 2. Train one model (clean by default; --adversarial uses the adv bundle).
python train.py --language python
python train.py --language python --adversarial

# 3. Clean-test evaluation. Threshold policy is evaluation-time only.
python evaluate.py --language python --threshold-mode fixed
python evaluate.py --language python --threshold-mode calibrated

# 4. Full 5-seed benchmark. Attacks default to whole-model recomputation
#    (--attack-eval full), matching the CPG attack procedure.
python run_five_seeds.py --language python --threshold-mode fixed
python run_five_seeds.py --language python --threshold-mode calibrated --resume
python run_five_seeds.py --language all --seeds 42-46 --attack-eval full

# Paper Table 9 reproduction: feature-block isolation instead of full
# recomputation. This writes the unsuffixed CSV; full mode writes *_fullmodel.
python run_five_seeds.py --language python --attack-eval isolated
```

Both threshold modes reuse the same per-seed checkpoint. Fixed mode evaluates
at 0.50; calibrated mode applies the CPG clean-validation rule to the clean
validation split. Run fixed first, then add `--resume` to the calibrated run
so the archived seed checkpoints/scalers are reused instead of retraining.
Checkpoint selection in `train.py` is fixed at clean validation F1@0.50, so
the threshold comparison does not select a different model. The full attack table is written to
`five_seed_{language}_{variant}_fullmodel.csv`; the isolated table keeps the
historical `five_seed_{language}_{variant}.csv` name.

C++ test features built before the balancing-rule change are stale. Regenerate
them with `python main.py --language cpp`; Python and Java do not need a
rebuild. Also pass `--rebuild-cache` once on the next C++ attack run so old
attack-feature caches built from the previous rows are not reused.

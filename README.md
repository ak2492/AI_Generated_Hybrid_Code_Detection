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

# 2. Train one model. One pass saves fixed050 + bestthr checkpoints and the
#    per-epoch test-metrics CSV.
python train.py --language python
python train.py --language python --adversarial

# 3. Clean-test evaluation. Threshold policy is evaluation-time only.
python evaluate.py --language python --threshold-mode fixed
python evaluate.py --language python --threshold-mode calibrated

# 4. Full 5-seed benchmark. Default = 2x2 matrix:
#    {fixed050, bestthr} x {0.50, each checkpoint's validation-best cut}.
#    Attacks default to whole-model recomputation, matching CPG.
python run_five_seeds.py --language python
python run_five_seeds.py --language python --threshold-mode all   # + calibrated
python run_five_seeds.py --language all --seeds 42-46 --attack-eval full

# Paper Table 9 reproduction: feature-block isolation instead of full
# recomputation. This writes the unsuffixed CSV; full mode writes *_fullmodel.
python run_five_seeds.py --language python --attack-eval isolated
```

One training pass produces `fixed050` (best clean-validation F1 at 0.50) and
`bestthr` (best clean-validation F1 over every threshold), plus JSON sidecars
with the selected epoch and cuts. The default result table is the 2x2 matrix
in `five_seed_{language}_{variant}_dual_fullmodel.csv`; fixed-only compatibility
runs keep `..._fullmodel.csv`, and the isolated Table 9 table keeps
`five_seed_{language}_{variant}.csv`. `--resume` reuses the dual checkpoints,
the shared scaler, and any completed CSV cells instead of retraining. Resume
cells carry an `EvalConfig` identity (attack seed, target, threshold policy,
and external-suite setting); a changed fixed cut or skip-external flag replaces
that cell rather than appending a mixed row set. A partial dual-artifact set
(for example, checkpoints without the shared scaler) retrains that seed so all
artifacts come from one trajectory. The `TrainValF1`/`TrainValROC` columns are
the checkpoint's selection metrics, not a re-evaluation at the cell's cut.

The per-seed `.pt.json` sidecars are part of the artifact set and are uploaded
alongside the `.pt`/`.pkl` files. If you copy checkpoints between machines,
copy the sidecars too; missing sidecars are reconstructed from the checkpoint
when possible, but the stored selection metadata is more precise.

C++ test features built before the balancing-rule change are stale. Regenerate
them with `python main.py --language cpp`; Python and Java do not need a
rebuild. Attack-feature caches are now content-keyed, so caches written by this
version remain valid when moved. A legacy cache written by older code is
reused only when its exact old filename still exists; because the legacy name
includes the old attack-code hash, editing `attack_utils.py` can invalidate
those older caches and force a one-time re-extraction.

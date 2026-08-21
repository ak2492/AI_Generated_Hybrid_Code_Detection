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
git clone [https://github.com/](https://github.com/)<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>.git
cd <YOUR_REPO_NAME>
pip install -r requirements.txt

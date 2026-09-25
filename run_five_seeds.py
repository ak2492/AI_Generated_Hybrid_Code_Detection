"""Five-seed protocol runner — train + clean eval + 8 attack suites + external OOD.

Same protocol as the CPG folder's run_five_seeds.py: every training seed
runs over IDENTICAL .npy bundles (main.py output) and identical attacked
sets (attack --base_seed fixed at 42), then reports every seed's values
plus mean +/- std. No manual math.

Sequential usage (bundles first, once per language):
  python main.py --language python
  python run_five_seeds.py --language python
  python run_five_seeds.py --language all --seeds 42-46 --adversarial

Per-seed checkpoint+scaler pairs are copied to *_seed{s}.* BEFORE the next
seed overwrites the canonical files, and copied back before every eval
step, so existing single-seed CLIs work unmodified.

Note: each attack suite reloads CodeT5+/CodeBERT internally (existing
run_attack_evaluation behavior); a full 5-seed x 3-language run is an
overnight Kaggle job. Basic isolated suites reuse {language}_test_X.npy.
"""
import argparse
import gc
import os
import random
import shutil
import subprocess
import sys

import pandas as pd
import torch

from attack_utils import (
    SHUFFLE_SALT,
    apply_statistical_attack_basic,
    apply_statistical_attack_enhanced_identical,
    get_language_config,
    get_ts_parser,
    meaning_preserving_rename,
    meaning_preserving_rename_enhanced_shuffled,
    normalize_naming_style,
    run_attack_evaluation,
    set_seed,
    strip_comments,
)
from evaluate import evaluate_model
from external_eval import run_external
from train import train_model

LANGUAGES = ["python", "java", "cpp"]
METRIC_COLS = ["Acc", "Prec", "Rec", "F1", "ROC", "FPR",
               "Latency_ms", "Throughput", "PeakRAM_MB", "N"]
COLUMNS = (["Language", "Variant", "Seed", "Scenario", "TrainValF1"]
           + METRIC_COLS)


def parse_seeds(spec):
    """Accept '42,43,44,45,46', '42-46', or mixes like '42-44,46'."""
    seeds = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    # I dedupe here because I want --resume reruns to stay idempotent.
    return sorted(set(seeds))


def _artifact_names(language, adversarial):
    tag = "_adv" if adversarial else ""
    ckpt = f"{language}{tag}_best_model.pt"
    scaler = f"{language}{tag}_scaler.pkl"
    base, ext = os.path.splitext(ckpt)
    sbase, sext = os.path.splitext(scaler)
    return ((ckpt, f"{base}_seed{{s}}{ext}"),
            (scaler, f"{sbase}_seed{{s}}{sext}"))


def _default_batch(language):
    return 64 if language == "python" else (32 if language == "java" else 16)


def _attack_suites(language, base_seed, target):
    """(scenario, name, attack_layer, mode, apply_attack_fn) — I mirror the
    four attack_*.py wrappers exactly, including layer/mode labels."""
    config = get_language_config(language)
    ts_parser = get_ts_parser(config["lang_obj"])
    attack_all = (target == "all")

    def auth_basic(code, idx):
        mod, _ = strip_comments(code, ts_parser, language)
        return mod

    def auth_enhanced(code, idx):
        mod, _ = strip_comments(code, ts_parser, language)
        mod, _ = normalize_naming_style(mod, ts_parser, language, config)
        return mod

    def stat_basic(code, idx, _b=base_seed, _l=language):
        return apply_statistical_attack_basic(code, random.Random(_b + idx), language=_l)

    def stat_enhanced(code, idx, _b=base_seed, _l=language):
        return apply_statistical_attack_enhanced_identical(
            code, random.Random(_b + idx), language=_l)

    def sem_basic(code, idx, _l=language):
        mod, _ = meaning_preserving_rename(code, ts_parser, _l, config)
        return mod

    def sem_enhanced(code, idx, _b=base_seed, _l=language):
        rng_shuf = random.Random(_b + idx + SHUFFLE_SALT)
        mod, _ = meaning_preserving_rename_enhanced_shuffled(
            code, ts_parser, _l, config, rng_shuf)
        return mod

    def full_basic(code, idx, _b=base_seed, _l=language):
        code, _ = strip_comments(code, ts_parser, _l)
        code, _ = meaning_preserving_rename(code, ts_parser, _l, config)
        rng = random.Random(_b + idx)
        return apply_statistical_attack_basic(code, rng, language=_l)

    def full_enhanced(code, idx, _b=base_seed, _l=language):
        code, _ = strip_comments(code, ts_parser, _l)
        code, _ = normalize_naming_style(code, ts_parser, _l, config)
        rng_shuf = random.Random(_b + idx + SHUFFLE_SALT)
        code, _ = meaning_preserving_rename_enhanced_shuffled(
            code, ts_parser, _l, config, rng_shuf)
        return code

    return [
        ("Auth (Basic)", "authorship-basic", "auth", "basic", auth_basic, attack_all),
        ("Auth (Enhanced)", "authorship-enhanced", "full", "enhanced", auth_enhanced, attack_all),
        ("Stat (Basic)", "statistical-basic", "stat", "basic", stat_basic, attack_all),
        ("Stat (Enhanced)", "statistical-enhanced", "full", "enhanced", stat_enhanced, attack_all),
        ("Sem (Basic)", "semantic-basic", "sem", "basic", sem_basic, attack_all),
        ("Sem (Enhanced)", "semantic-enhanced", "full", "enhanced", sem_enhanced, attack_all),
        ("Full (Basic)", "full-basic", "full", "basic", full_basic, attack_all),
        ("Full (Enhanced)", "full-enhanced", "full", "enhanced", full_enhanced, attack_all),
    ]


def run_language(language, seeds, adversarial=False, epochs=100, batch_size=None,
                 learning_rate=1e-5, threshold=0.50, base_seed=42, target="machine",
                 skip_external=False, out_csv=None, resume=False,
                 feature_cache_dir=".", rebuild_cache=False):
    variant = "adv" if adversarial else "clean"
    tag = "ADV" if adversarial else "CLEAN"
    if batch_size is None:
        batch_size = _default_batch(language)
    (ckpt_canon, ckpt_tpl), (scaler_canon, scaler_tpl) = _artifact_names(language, adversarial)
    rows = []
    prev_all = None
    if out_csv and resume and os.path.exists(out_csv):
        prev_all = pd.read_csv(out_csv)
        prev = prev_all[(prev_all["Language"] != language) | (prev_all["Variant"] != variant)]
        rows.extend(prev.to_dict("records"))
        print(f"[resume] kept {len(prev)} existing rows for other configs.")

    for seed in seeds:
        # I start each seed from a clean memory state because 5 seeds of
        # CodeT5+/CodeBERT alloc/free cycles fragment VRAM; the reset also
        # keeps peak stats honest.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        print("\n" + "=" * 85)
        print(f"SEED {seed} [{language.upper()} {tag}]")
        print("=" * 85)
        seed_ckpt, seed_scaler = ckpt_tpl.format(s=seed), scaler_tpl.format(s=seed)

        if resume and os.path.exists(seed_ckpt) and os.path.exists(seed_scaler):
            print(f"[resume] reusing {seed_ckpt} + {seed_scaler}, skipping training.")
            best_f1 = float("nan")
            if prev_all is not None:
                hit = prev_all[(prev_all["Language"] == language)
                               & (prev_all["Variant"] == variant)
                               & (prev_all["Seed"] == seed)]
                if len(hit):
                    best_f1 = float(hit["TrainValF1"].iloc[0])
        else:
            best_f1 = train_model(language=language, epochs=epochs,
                                  batch_size=batch_size,
                                  learning_rate=learning_rate,
                                  adversarial=adversarial, seed=seed)
            shutil.copyfile(ckpt_canon, seed_ckpt)
            shutil.copyfile(scaler_canon, seed_scaler)
            print(f"Artifacts archived -> {seed_ckpt}, {seed_scaler}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        def _restore():
            # I restore the per-seed pair before every eval step because all
            # eval CLIs load the canonical filenames.
            shutil.copyfile(seed_ckpt, ckpt_canon)
            shutil.copyfile(seed_scaler, scaler_canon)

        def _row(scenario, res):
            if res is None:
                return
            rows.append({
                "Language": language, "Variant": variant, "Seed": seed,
                "Scenario": scenario, "TrainValF1": best_f1,
                "Acc": res["Acc"], "Prec": res["Prec"], "Rec": res["Rec"],
                "F1": res["F1"], "ROC": res["ROC"], "FPR": res["FPR"],
                "Latency_ms": res["Latency_ms"], "Throughput": res["Throughput"],
                "PeakRAM_MB": res.get("PeakRAM_MB", 0.0), "N": res["N"],
            })

        _restore()
        _row("Clean Test", evaluate_model(language=language, batch_size=batch_size,
                                          adversarial=adversarial))

        for scenario, name, layer, mode, fn, attack_all in _attack_suites(
                language, base_seed, target):
            _restore()
            res = run_attack_evaluation(
                language, name, fn, batch_size, None, base_seed,
                attack_all_samples=attack_all, adversarial=adversarial,
                attack_layer=layer, mode=mode,
                feature_cache_dir=feature_cache_dir,
                rebuild_cache=rebuild_cache)
            if res is None:
                continue
            _row(scenario, {"Acc": res["accuracy"], "Prec": res["precision"],
                            "Rec": res["recall"], "F1": res["f1"], "ROC": res["auc"],
                            "FPR": res["fpr"], "Latency_ms": res["Latency_ms"],
                            "Throughput": res["Throughput"],
                            "PeakRAM_MB": res.get("PeakRAM_MB", 0.0),
                            "N": len(res["labels"])})
            # I free the suite result here because it carries full feature
            # arrays (test set x 813-d); only scalars were copied into rows.
            del res
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # I drop the last attack closure here because it retains the
        # tree-sitter parser via closure; next seed builds its own.
        del fn

        if not skip_external:
            _restore()
            ext_results = run_external(language=language, suite="all",
                                       batch_size=batch_size, threshold=threshold,
                                       base_seed=base_seed, adversarial=adversarial,
                                       feature_cache_dir=feature_cache_dir,
                                       rebuild_cache=rebuild_cache)
            for scenario, res in ext_results.items():
                _row(scenario, res)
            del ext_results
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if out_csv:
            pd.DataFrame(rows, columns=COLUMNS).to_csv(out_csv, index=False)

    df_all = pd.DataFrame(rows, columns=COLUMNS)
    if out_csv:
        df_all.to_csv(out_csv, index=False)
    return df_all


def summarize(df_all):
    cur = df_all
    print("\n" + "=" * 100)
    print("PER-SEED RESULTS")
    print("=" * 100)
    print(cur.to_string(index=False))
    summary = cur.groupby(["Language", "Variant", "Scenario"], as_index=False).agg(
        {c: ["mean", "std"] for c in METRIC_COLS + ["TrainValF1"]})
    summary.columns = ["_".join(c).strip("_") for c in summary.columns.values]
    print("\n" + "=" * 100)
    print("SUMMARY: MEAN +/- STD (sample std, ddof=1)")
    print("=" * 100)
    show = summary.copy()
    for c in METRIC_COLS:
        show[c] = (summary[f"{c}_mean"].map("{:.4f}".format) + " +/- "
                   + summary[f"{c}_std"].map("{:.4f}".format))
    print(show[["Language", "Variant", "Scenario"] + METRIC_COLS].to_string(index=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid 5-seed protocol runner")
    parser.add_argument("--language", type=str, default="python",
                        choices=["python", "java", "cpp", "all"])
    parser.add_argument("--seeds", type=str, default="42-46",
                        help="Seed list, e.g. '42-46' or '7,123,999' (any ints allowed)")
    parser.add_argument("--adversarial", action="store_true",
                        help="Run the adversarial variant instead of clean (default: clean)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--base_seed", type=int, default=42,
                        help="Fixed attack-sampling seed (keep 42 so attacked sets match across training seeds)")
    parser.add_argument("--target", type=str, default="machine", choices=["machine", "all"])
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--no-feature-cache", action="store_true",
                        help="Disable the extract-once feature cache (extract every seed)")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Force re-extraction even when cache files exist")
    parser.add_argument("--cache-dir", type=str, default=".",
                        help="Directory for extract-once cache files")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse existing *_seed{s} artifacts and keep other configs' CSV rows")
    parser.add_argument("--upload", action="store_true",
                        help="Upload per-seed .pt/.pkl files to Hugging Face at the end")
    parser.add_argument("--upload-repo", type=str, default=None,
                        help="Required with --upload: full repo id 'owner/name' (no default; "
                             "missing repo/token/files skips upload gracefully)")
    args = parser.parse_args()

    set_seed(args.base_seed)
    seeds = parse_seeds(args.seeds)
    print(f"Seeds: {seeds}")
    langs = LANGUAGES if args.language == "all" else [args.language]
    variant = "adv" if args.adversarial else "clean"

    # I pass None (not ".") when caching is off so library code takes its
    # original uncached path byte-for-byte.
    feature_cache_dir = None if args.no_feature_cache else args.cache_dir
    for lang in langs:
        out_csv = f"five_seed_{lang}_{variant}.csv"
        df_all = run_language(lang, seeds, adversarial=args.adversarial,
                              epochs=args.epochs, batch_size=args.batch_size,
                              learning_rate=args.learning_rate,
                              threshold=args.threshold, base_seed=args.base_seed,
                              target=args.target,
                              skip_external=args.skip_external,
                              out_csv=out_csv, resume=args.resume,
                              feature_cache_dir=feature_cache_dir,
                              rebuild_cache=args.rebuild_cache)
        summary = summarize(df_all[df_all["Language"] == lang])
        summary.to_csv(f"five_seed_{lang}_{variant}_summary.csv", index=False)

        if args.upload:
            # I require an explicit repo and never default to a personal
            # account; anything missing is a warned skip, never an error.
            if not args.upload_repo:
                print("[!] --upload needs --upload-repo <owner/name>; skipping upload.")
                continue
            script = os.path.join("..", "cpg-based-ai-generated-code-detection",
                                  "tools", "hf_upload", "upload_seed_models.py")
            subprocess.run([sys.executable, script, "--model-dir", ".",
                            "--repo", args.upload_repo,
                            "--pattern", "*_seed*.pt",
                            "--pattern", "*_seed*.pkl"], check=True)

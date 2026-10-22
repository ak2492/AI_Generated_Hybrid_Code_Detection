"""Five-seed protocol runner — train + clean eval + 8 attack suites + external OOD.

One training pass per seed produces TWO checkpoints from the same trajectory:

  fixed050  best validation F1 at the fixed 0.50 cut
  bestthr   best validation F1 over every threshold

Both are evaluated at BOTH threshold policies (unless you narrow the matrix):

  fixed       0.50 (or --threshold)
  best        that checkpoint's validation-F1-argmax cut
  calibrated  the conservative, prior-aware clean-validation cut stored in the
              checkpoint sidecar (opt-in compatibility mode)

The default table is therefore the 2x2 matrix

  {fixed050, bestthr} x {0.50, best-threshold}

with identical test rows and the same attack/external procedure for every cell.
No checkpoint averaging, no ensembling, and no retraining per threshold.

Data preflight is reuse-first: if the six clean ``.npy`` splits are present they
are used as-is. Adversarial training additionally needs
``{language}_train_adv_X.npy`` / ``_y.npy``. Pass ``--rebuild-data`` to force
re-extraction. Attack-feature caches are content-keyed, so moving the
preprocessed features into a fresh clone does not invalidate them.

Per-seed artifacts are archived before the next seed overwrites the canonical
files. Every evaluation cell copies its chosen checkpoint and the shared scaler
back to the canonical names, because the existing single-seed CLIs load those
paths.

Usage:
  python run_five_seeds.py --language python
  python run_five_seeds.py --language all --seeds 42-46 --adversarial
  python run_five_seeds.py --language python --resume
  python run_five_seeds.py --language python --threshold-mode fixed
"""
import argparse
import gc
import glob
import json
import os
import random
import shutil
import subprocess
import sys

import numpy as np
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
    normalize_layout_enhanced_identical,
    normalize_naming_style,
    run_attack_evaluation,
    set_seed,
    strip_comments,
)
from evaluate import evaluate_model
from external_eval import run_external
from threshold import compute_seed_threshold, test_prior
from train import _best_threshold_and_f1, train_model

LANGUAGES = ["python", "java", "cpp"]
METRIC_COLS = ["Acc", "Prec", "Rec", "F1", "ROC", "FPR",
               "Latency_ms", "Throughput", "PeakRAM_MB", "N"]
COLUMNS = (["Language", "Variant", "Seed", "Checkpoint", "Scenario",
            "Threshold", "ThresholdMode", "EvalConfig", "SelectionThreshold",
            "BestThreshold", "CalibratedThreshold", "SelectedEpoch",
            "TrainValF1", "TrainValROC"]
           + METRIC_COLS)
CHECKPOINTS = ("fixed050", "bestthr")


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
    best = f"{language}{tag}_best_model_bestthr.pt"
    scaler = f"{language}{tag}_scaler.pkl"
    cbase, cext = os.path.splitext(ckpt)
    bbase, bext = os.path.splitext(best)
    sbase, sext = os.path.splitext(scaler)
    return {
        "fixed050": {
            "canonical": ckpt,
            "archive_tpl": f"{cbase}_seed{{s}}{cext}",
            "meta_canonical": ckpt + ".json",
            "meta_tpl": f"{cbase}_seed{{s}}{cext}.json",
        },
        "bestthr": {
            "canonical": best,
            "archive_tpl": f"{bbase}_seed{{s}}{bext}",
            "meta_canonical": best + ".json",
            "meta_tpl": f"{bbase}_seed{{s}}{bext}.json",
        },
        "scaler": {
            "canonical": scaler,
            "archive_tpl": f"{sbase}_seed{{s}}{sext}",
        },
    }


def _default_batch(language):
    return 64 if language == "python" else (32 if language == "java" else 16)


def _meta_path(path):
    return path + ".json"


def _metadata_for(path):
    meta_path = _meta_path(path)
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"[!] Could not read metadata {meta_path}: {exc}")
        return {}


def _write_json(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)


def _threshold_for_mode(mode, record, language, adversarial, fixed_cut,
                        eval_prior):
    meta = record["meta"]
    if fixed_cut is None:
        fixed_cut = 0.50
    if mode == "fixed":
        return float(fixed_cut)
    if mode == "best":
        value = meta.get("best_threshold", meta.get("selection_threshold"))
        if value is not None:
            return float(value)
        # Legacy sidecar-free archive: derive the exact val-best cut.
        from threshold import clean_val_probs
        probs, labels = clean_val_probs(
            language, adversarial=adversarial, checkpoint=record["archive"],
            scaler_file=record["scaler_archive"])
        value, _ = _best_threshold_and_f1(probs, labels)
        print(f"[i] Derived best threshold {value:.4f} for {record['archive']}")
        return float(value)
    if mode == "calibrated":
        value = meta.get("calibrated_threshold")
        if value is not None:
            return float(value)
        return float(compute_seed_threshold(
            language, adversarial=adversarial, checkpoint=record["archive"],
            scaler_file=record["scaler_archive"], fallback=fixed_cut,
            prior=eval_prior))
    raise ValueError(f"Unknown threshold mode {mode!r}")


def _modes_for(threshold_mode):
    if threshold_mode == "both":
        return ["fixed", "best"]
    if threshold_mode == "all":
        return ["fixed", "best", "calibrated"]
    return [threshold_mode]


def _eval_config(mode, cut, base_seed, target, skip_external):
    """Identity of an evaluation cell, excluding the seed/checkpoint.

    Fixed-mode cut is part of the identity so a resumed run with a different
    --threshold replaces that cell instead of silently mixing cuts. For the
    best/calibrated policies the cut is checkpoint metadata, so retraining
    already invalidates the row set.
    """
    text = (f"base={int(base_seed)}|target={target}"
            f"|ext={0 if skip_external else 1}")
    if mode == "fixed":
        text += f"|thr={float(cut):.8f}"
    return text


def _attack_suites(language, base_seed, target, force_full=False):
    """(scenario, name, attack_layer, mode, apply_attack_fn) — I mirror the
    four attack_*.py wrappers exactly, including layer/mode labels.

    `force_full=True` recomputes every feature block for every transform,
    which is what the CPG model necessarily does (it has no 813-d feature
    groups to isolate). It is the runner default because the CPG/hybrid
    comparison must use the same whole-model attacked inputs; pass
    --attack-eval isolated for the paper Table 9 reproduction.
    """
    config = get_language_config(language)
    ts_parser = get_ts_parser(config["lang_obj"])
    attack_all = (target == "all")

    def auth_basic(code, idx):
        mod, _ = strip_comments(code, ts_parser, language)
        return mod

    def auth_enhanced(code, idx):
        # Must stay byte-identical to attack_authorship.py --enhanced and to
        # the CPG's enhanced-auth path: comments -> snake_case -> layout
        # normalization. The old version skipped the last step.
        mod, _ = strip_comments(code, ts_parser, language)
        mod, _ = normalize_naming_style(mod, ts_parser, language, config)
        return normalize_layout_enhanced_identical(mod)

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
        # Must stay byte-identical to attack_full.py --enhanced and to the
        # CPG's enhanced-full path: auth (comments -> snake_case -> layout
        # normalization) -> semantic (shuffled rename) -> statistical.
        code, _ = strip_comments(code, ts_parser, _l)
        code, _ = normalize_naming_style(code, ts_parser, _l, config)
        code = normalize_layout_enhanced_identical(code)
        rng_shuf = random.Random(_b + idx + SHUFFLE_SALT)
        code, _ = meaning_preserving_rename_enhanced_shuffled(
            code, ts_parser, _l, config, rng_shuf)
        rng_stat = random.Random(_b + idx)
        return apply_statistical_attack_enhanced_identical(
            code, rng_stat, language=_l)

    suites = [
        ("Auth (Basic)", "authorship-basic", "auth", "basic", auth_basic, attack_all),
        ("Auth (Enhanced)", "authorship-enhanced", "full", "enhanced", auth_enhanced, attack_all),
        ("Stat (Basic)", "statistical-basic", "stat", "basic", stat_basic, attack_all),
        ("Stat (Enhanced)", "statistical-enhanced", "full", "enhanced", stat_enhanced, attack_all),
        ("Sem (Basic)", "semantic-basic", "sem", "basic", sem_basic, attack_all),
        ("Sem (Enhanced)", "semantic-enhanced", "full", "enhanced", sem_enhanced, attack_all),
        ("Full (Basic)", "full-basic", "full", "basic", full_basic, attack_all),
        ("Full (Enhanced)", "full-enhanced", "full", "enhanced", full_enhanced, attack_all),
    ]
    if force_full:
        suites = [(scenario, name, "full", mode, fn, attack_all)
                  for scenario, name, _layer, mode, fn, attack_all in suites]
    return suites


def _required_data(language, adversarial):
    splits = ("validation", "test") if adversarial else ("train", "validation", "test")
    req = [f"{language}_{split}_{kind}.npy"
           for split in splits
           for kind in ("X", "y")]
    if adversarial:
        req += [f"{language}_train_adv_X.npy", f"{language}_train_adv_y.npy"]
    return req


def _run_main(language, adversarial=False):
    cmd = [sys.executable, "main.py", "--language", language]
    if adversarial:
        cmd.append("--adversarial")
    print(f"[data] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def ensure_data(language, adversarial=False, rebuild=False):
    """Reuse existing .npy features; extract only when something is missing.

    Hybrid``main.py --adversarial`` writes ONLY ``train_adv_*`` plus the clean
    validation/test splits; it deliberately does not write ``train_X.npy``.
    So a missing clean train bundle needs the clean command first, then the
    adversarial command. Running only ``--adversarial`` here used to leave a
    fresh clone permanently missing the clean train arrays.
    """
    # Adversarial training reads train_adv plus clean validation/test. It does
    # not read clean train, so a fresh adv feature bundle must not trigger a
    # full clean-train extraction just because train_X.npy is absent.
    clean_splits = (("validation", "test") if adversarial
                    else ("train", "validation", "test"))
    clean_required = [f"{language}_{split}_{kind}.npy"
                      for split in clean_splits for kind in ("X", "y")]
    adv_required = [f"{language}_train_adv_X.npy", f"{language}_train_adv_y.npy"]
    required = clean_required + (adv_required if adversarial else [])

    if rebuild:
        print(f"[data] --rebuild-data: re-extracting {language} features")
        _run_main(language, adversarial=False)
        if adversarial:
            _run_main(language, adversarial=True)
    else:
        reused = True
        missing_clean = [p for p in clean_required if not os.path.exists(p)]
        if missing_clean:
            reused = False
            print(f"[data] missing {len(missing_clean)} clean feature file(s); "
                  f"extracting once: {', '.join(missing_clean[:4])}"
                  f"{'...' if len(missing_clean) > 4 else ''}")
            _run_main(language, adversarial=False)
        if adversarial:
            missing_adv = [p for p in adv_required if not os.path.exists(p)]
            if missing_adv:
                reused = False
                print(f"[data] missing {len(missing_adv)} adversarial feature "
                      f"file(s); building the adv train split once.")
                _run_main(language, adversarial=True)
        if reused:
            print(f"[data] reusing existing {language} feature bundle "
                  f"({len(required)} .npy files, no re-extraction).")

    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"Feature extraction did not produce: {missing}")
    shapes = {p: np.load(p, mmap_mode="r").shape for p in required}
    print(f"[data] {language}: " + ", ".join(
        f"{os.path.basename(p)}={shapes[p]}" for p in required[:4]) + " ...")
    return shapes


def _report_caches(language, cache_dir, rebuild=False):
    if not cache_dir:
        print("[cache] disabled (--no-feature-cache)")
        return
    if rebuild:
        print(f"[cache] --rebuild-cache set; existing caches in {cache_dir} will be ignored")
        return
    attack = glob.glob(os.path.join(cache_dir, f"{language}_atkfeat_*.npz"))
    external = glob.glob(os.path.join(cache_dir, f"{language}_extfeat_*.npz"))
    print(f"[cache] {cache_dir}: {len(attack)} attack cache(s), "
          f"{len(external)} external cache(s) reused when content matches")


def _existing_keys(df, language, variant):
    if df is None or not len(df) or "Checkpoint" not in df.columns:
        return set()
    cur = df[(df["Language"] == language) & (df["Variant"] == variant)]
    keys = set()
    for _, row in cur.iterrows():
        try:
            keys.add((int(row["Seed"]), str(row["Checkpoint"]),
                      str(row["ThresholdMode"]), str(row["EvalConfig"])))
        except (TypeError, ValueError):
            continue
    return keys


def run_language(language, seeds, adversarial=False, epochs=100, batch_size=None,
                 learning_rate=1e-5, threshold=0.50, base_seed=42,
                 target="machine", skip_external=False, out_csv=None,
                 resume=False, feature_cache_dir=".", rebuild_cache=False,
                 threshold_mode="both", attack_eval="full", rebuild_data=False):
    variant = "adv" if adversarial else "clean"
    tag = "ADV" if adversarial else "CLEAN"
    if not seeds:
        raise ValueError("No seeds requested")
    if threshold_mode not in ("both", "fixed", "best", "calibrated", "all"):
        raise ValueError(f"Unknown threshold_mode {threshold_mode!r}")
    if attack_eval not in ("isolated", "full"):
        raise ValueError(f"Unknown attack_eval {attack_eval!r}")
    if batch_size is None:
        batch_size = _default_batch(language)

    artifacts = _artifact_names(language, adversarial)
    ckpt_canon = artifacts["fixed050"]["canonical"]
    scaler_canon = artifacts["scaler"]["canonical"]
    rows = []
    existing = set()
    thresholds = {}
    thr_path = (os.path.splitext(out_csv)[0] + "_thresholds.json") if out_csv else None
    if resume and thr_path and os.path.exists(thr_path):
        try:
            with open(thr_path, "r", encoding="utf-8") as fh:
                previous = json.load(fh).get("cells", {})
            # Keep cells for seeds outside this invocation; current seeds are
            # rebuilt below so stale modes do not survive a resumed subset run.
            for key, value in previous.items():
                try:
                    if int(key) not in seeds:
                        thresholds[key] = value
                except (TypeError, ValueError):
                    thresholds[key] = value
        except (OSError, ValueError):
            thresholds = {}

    prev_all = None
    if out_csv and resume and os.path.exists(out_csv):
        prev_all = pd.read_csv(out_csv)
        other = prev_all[(prev_all["Language"] != language)
                         | (prev_all["Variant"] != variant)]
        existing = _existing_keys(prev_all, language, variant)
        if "Checkpoint" in prev_all.columns:
            cur = prev_all[(prev_all["Language"] == language)
                           & (prev_all["Variant"] == variant)]
            keep = pd.concat([other, cur], ignore_index=True)
        else:
            # Pre-dual CSVs have no Checkpoint column; regenerate them.
            keep = other
            print("[resume] old CSV has no Checkpoint column; "
                  "regenerating this language/variant.")
        rows.extend(keep.reindex(columns=COLUMNS).to_dict("records"))
        print(f"[resume] kept {len(keep)} rows "
              f"({len(existing)} eval cells already complete).")

    ensure_data(language, adversarial=adversarial, rebuild=rebuild_data)
    _report_caches(language, feature_cache_dir, rebuild_cache)
    eval_prior = test_prior(language)
    modes = _modes_for(threshold_mode)
    print(f"[i] Evaluation matrix: checkpoints={list(CHECKPOINTS)} "
          f"threshold_modes={modes}")

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

        archives = {}
        for name in CHECKPOINTS:
            spec = artifacts[name]
            archive = spec["archive_tpl"].format(s=seed)
            meta_archive = spec["meta_tpl"].format(s=seed)
            archives[name] = {
                "archive": archive,
                "meta_archive": meta_archive,
                "canonical": spec["canonical"],
                "meta_canonical": spec["meta_canonical"],
            }
        scaler_archive = artifacts["scaler"]["archive_tpl"].format(s=seed)
        for rec in archives.values():
            rec["scaler_archive"] = scaler_archive

        have_fixed = os.path.exists(archives["fixed050"]["archive"])
        have_best = os.path.exists(archives["bestthr"]["archive"])
        have_scaler = os.path.exists(scaler_archive)

        if resume and have_fixed and have_best and have_scaler:
            print(f"[resume] reusing {archives['fixed050']['archive']}, "
                  f"{archives['bestthr']['archive']}, {scaler_archive}; "
                  f"skipping training.")
            for rec in archives.values():
                rec["meta"] = _metadata_for(rec["archive"])
        else:
            if resume and (have_fixed or have_best or have_scaler):
                print("[resume] partial dual artifact set found for seed "
                      f"{seed}; retraining that seed so both checkpoints and "
                      "the shared scaler come from one trajectory.")
            metrics_path = f"epoch_metrics_{language}_{variant}_seed{seed}.csv"
            info = train_model(
                language=language, epochs=epochs, batch_size=batch_size,
                learning_rate=learning_rate, adversarial=adversarial,
                seed=seed, threshold=threshold,
                epoch_metrics_path=metrics_path, return_info=True)
            for name, rec in archives.items():
                shutil.copyfile(artifacts[name]["canonical"], rec["archive"])
                if os.path.exists(artifacts[name]["meta_canonical"]):
                    shutil.copyfile(artifacts[name]["meta_canonical"],
                                    rec["meta_archive"])
                # The sidecar is the canonical metadata record (selected_epoch,
                # val_f1/val_roc, calibrated cut). `info` is the in-memory
                # fallback, and the sidecar copies are what resume will read.
                rec["meta"] = _metadata_for(rec["archive"]) or info.get(name, {})
            shutil.copyfile(scaler_canon, scaler_archive)
            print(f"Artifacts archived -> {archives['fixed050']['archive']}, "
                  f"{archives['bestthr']['archive']}, {scaler_archive}")
            # A retrain supersedes any stale CSV rows for this seed.
            rows[:] = [r for r in rows
                       if not (r.get("Language") == language
                               and r.get("Variant") == variant
                               and r.get("Seed") == seed)]
            existing = {k for k in existing if k[0] != seed}
            del info
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        def _restore(record):
            # Every eval CLI loads the canonical fixed filename plus the
            # canonical scaler. Copy the chosen archive into those paths.
            shutil.copyfile(record["archive"], ckpt_canon)
            shutil.copyfile(scaler_archive, scaler_canon)

        def _record_row(record, checkpoint, mode, scenario, res, cut):
            if res is None:
                return
            meta = record.get("meta", {})
            rows.append({
                "Language": language, "Variant": variant, "Seed": seed,
                "Checkpoint": checkpoint, "Scenario": scenario,
                "Threshold": float(cut), "ThresholdMode": mode,
                "EvalConfig": _eval_config(mode, cut, base_seed, target,
                                           skip_external),
                "SelectionThreshold": meta.get("selection_threshold", ""),
                "BestThreshold": meta.get("best_threshold", ""),
                "CalibratedThreshold": meta.get("calibrated_threshold", ""),
                "SelectedEpoch": meta.get("selected_epoch", ""),
                "TrainValF1": meta.get("val_f1", float("nan")),
                "TrainValROC": meta.get("val_roc", float("nan")),
                "Acc": res["Acc"], "Prec": res["Prec"], "Rec": res["Rec"],
                "F1": res["F1"], "ROC": res["ROC"], "FPR": res["FPR"],
                "Latency_ms": res.get("Latency_ms", float("nan")),
                "Throughput": res.get("Throughput", float("nan")),
                "PeakRAM_MB": res.get("PeakRAM_MB", 0.0),
                "N": res["N"],
            })

        suites = _attack_suites(language, base_seed, target,
                               force_full=(attack_eval == "full"))
        for checkpoint in CHECKPOINTS:
            record = archives[checkpoint]
            if not os.path.exists(record["archive"]):
                raise SystemExit(f"Missing checkpoint archive {record['archive']}")
            for mode in modes:
                cut = _threshold_for_mode(mode, record, language, adversarial,
                                          threshold, eval_prior)
                config = _eval_config(mode, cut, base_seed, target,
                                      skip_external)
                cell = (seed, checkpoint, mode, config)
                thresholds.setdefault(str(seed), {}).setdefault(checkpoint, {})[mode] = {
                    "threshold": float(cut),
                    "selection_threshold": record["meta"].get("selection_threshold"),
                    "best_threshold": record["meta"].get("best_threshold"),
                    "calibrated_threshold": record["meta"].get("calibrated_threshold"),
                }
                if resume and cell in existing:
                    print(f"[resume] skip existing cell {cell}")
                    continue
                # A cell evaluated under a different fixed cut / target /
                # attack seed / external-suite policy is stale: replace its
                # rows instead of appending a silently mixed configuration.
                rows[:] = [r for r in rows
                           if not (r.get("Language") == language
                                   and r.get("Variant") == variant
                                   and r.get("Seed") == seed
                                   and r.get("Checkpoint") == checkpoint
                                   and r.get("ThresholdMode") == mode)]
                print(f"\n--- SEED {seed} | {checkpoint} @ {mode} "
                      f"({cut:.4f}) ---")
                _restore(record)
                res = evaluate_model(language=language, batch_size=batch_size,
                                     adversarial=adversarial, threshold=cut)
                _record_row(record, checkpoint, mode, "Clean Test", res, cut)

                for scenario, name, layer, atk_mode, fn, attack_all in suites:
                    _restore(record)
                    res = run_attack_evaluation(
                        language, name, fn, batch_size, None, base_seed,
                        attack_all_samples=attack_all, adversarial=adversarial,
                        attack_layer=layer, mode=atk_mode,
                        feature_cache_dir=feature_cache_dir,
                        rebuild_cache=rebuild_cache, threshold=cut)
                    if res is None:
                        continue
                    _record_row(record, checkpoint, mode, scenario,
                                {"Acc": res["accuracy"], "Prec": res["precision"],
                                 "Rec": res["recall"], "F1": res["f1"],
                                 "ROC": res["auc"], "FPR": res["fpr"],
                                 "Latency_ms": res["Latency_ms"],
                                 "Throughput": res["Throughput"],
                                 "PeakRAM_MB": res.get("PeakRAM_MB", 0.0),
                                 "N": len(res["labels"])}, cut)
                    del res
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if not skip_external:
                    _restore(record)
                    ext_results = run_external(
                        language=language, suite="all", batch_size=batch_size,
                        threshold=cut, base_seed=base_seed,
                        adversarial=adversarial,
                        feature_cache_dir=feature_cache_dir,
                        rebuild_cache=rebuild_cache)
                    for scenario, res in ext_results.items():
                        _record_row(record, checkpoint, mode, scenario, res, cut)
                    del ext_results
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if out_csv:
                    # Flush after every cell so a preemption loses only the
                    # current cell, never the whole five-seed run.
                    pd.DataFrame(rows, columns=COLUMNS).to_csv(out_csv, index=False)

        # I drop the attack closures here because they retain the tree-sitter
        # parser; the next seed builds its own.
        del suites
        # Restore fixed050 as the canonical default for later single-seed CLIs.
        shutil.copyfile(archives["fixed050"]["archive"], ckpt_canon)
        shutil.copyfile(scaler_archive, scaler_canon)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df_all = pd.DataFrame(rows, columns=COLUMNS)
    if out_csv:
        df_all.to_csv(out_csv, index=False)
        thr_path = os.path.splitext(out_csv)[0] + "_thresholds.json"
        _write_json(thr_path, {
            "language": language, "variant": variant,
            "threshold_mode": threshold_mode, "fallback": float(threshold),
            "eval_prior": eval_prior, "cells": thresholds,
        })
        print(f"[i] Wrote per-cell threshold metadata -> {thr_path}")
    return df_all


def summarize(df_all):
    cur = df_all
    print("\n" + "=" * 100)
    print("PER-SEED RESULTS")
    print("=" * 100)
    print(cur.to_string(index=False))
    group_cols = ["Language", "Variant", "Checkpoint", "ThresholdMode",
                  "EvalConfig", "Scenario"]
    summary = cur.groupby(group_cols, as_index=False).agg(
        {c: ["mean", "std"] for c in METRIC_COLS + ["TrainValF1", "TrainValROC"]})
    summary.columns = ["_".join(c).strip("_") for c in summary.columns.values]
    print("\n" + "=" * 100)
    print("SUMMARY: MEAN +/- STD (sample std, ddof=1)")
    print("=" * 100)
    show = summary.copy()
    for c in METRIC_COLS:
        show[c] = (summary[f"{c}_mean"].map("{:.4f}".format) + " +/- "
                   + summary[f"{c}_std"].map("{:.4f}".format))
    print(show[group_cols + METRIC_COLS].to_string(index=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid 5-seed dual-checkpoint runner")
    parser.add_argument("--language", type=str, default="python",
                        choices=["python", "java", "cpp", "all"])
    parser.add_argument("--seeds", type=str, default="42-46",
                        help="Seed list, e.g. '42-46' or '7,123,999' (any ints allowed)")
    parser.add_argument("--adversarial", action="store_true",
                        help="Run the adversarial variant instead of clean (default: clean)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Fixed decision cut used by the 'fixed' policy")
    parser.add_argument("--threshold-mode", type=str, default="both",
                        choices=["both", "fixed", "best", "calibrated", "all"],
                        help="'both' = 2x2 matrix (0.50 and each checkpoint's "
                             "validation-best cut); 'all' also includes the "
                             "conservative calibrated cut.")
    parser.add_argument("--attack-eval", type=str, default="full",
                        choices=["isolated", "full"],
                        help="'full' (default) = recompute every block under each "
                             "transform, matching the CPG model's whole-model "
                             "evaluation; 'isolated' = paper Table 9 feature-block "
                             "isolation")
    parser.add_argument("--base_seed", type=int, default=42,
                        help="Fixed attack-sampling seed (keep 42 so attacked sets match across training seeds)")
    parser.add_argument("--target", type=str, default="machine", choices=["machine", "all"])
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--no-feature-cache", action="store_true",
                        help="Disable the extract-once feature cache (extract every seed)")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Force re-extraction even when cache files exist")
    parser.add_argument("--rebuild-data", action="store_true",
                        help="Force .npy feature re-extraction (default: reuse existing)")
    parser.add_argument("--cache-dir", type=str, default=".",
                        help="Directory for extract-once cache files")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse existing dual artifacts and completed CSV cells")
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
    suffix = {"fixed": "", "both": "_dual", "best": "_best",
              "calibrated": "_calibrated", "all": "_all"}[args.threshold_mode]
    atk_suffix = "" if args.attack_eval == "isolated" else "_fullmodel"
    for lang in langs:
        out_csv = f"five_seed_{lang}_{variant}{suffix}{atk_suffix}.csv"
        df_all = run_language(
            lang, seeds, adversarial=args.adversarial, epochs=args.epochs,
            batch_size=args.batch_size, learning_rate=args.learning_rate,
            threshold=args.threshold, base_seed=args.base_seed,
            target=args.target, skip_external=args.skip_external,
            out_csv=out_csv, resume=args.resume,
            feature_cache_dir=feature_cache_dir,
            rebuild_cache=args.rebuild_cache,
            threshold_mode=args.threshold_mode, attack_eval=args.attack_eval,
            rebuild_data=args.rebuild_data)
        summary = summarize(df_all[df_all["Language"] == lang])
        summary.to_csv(
            f"five_seed_{lang}_{variant}{suffix}{atk_suffix}_summary.csv",
            index=False)

        if args.upload:
            # I require an explicit repo and never default to a personal
            # account; anything missing is a warned skip, never an error.
            if not args.upload_repo:
                print("[!] --upload needs --upload-repo <owner/name>; skipping upload.")
                continue
            script = os.path.join("..", "cpg-based-ai-generated-code-detection",
                                  "tools", "hf_upload", "upload_seed_models.py")
            if not os.path.exists(script):
                print(f"[!] Upload script not found at {script}; skipping upload.")
            else:
                try:
                    subprocess.run([sys.executable, script, "--model-dir", ".",
                                    "--repo", args.upload_repo,
                                    "--pattern", "*_seed*.pt",
                                    "--pattern", "*_seed*.pkl",
                                    "--pattern", "*_seed*.json"], check=True)
                except Exception as e:
                    print(f"[!] Upload failed: {e}; continuing without upload.")

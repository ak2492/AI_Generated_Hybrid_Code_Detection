import random
from tqdm.auto import tqdm
from attack_utils import (
    set_seed, get_language_config, get_ts_parser,
    strip_comments, strip_comments_enhanced,
    normalize_naming_style, normalize_layout,
    meaning_preserving_rename, meaning_preserving_rename_enhanced,
    apply_statistical_attack, apply_statistical_attack_basic
)

def generate_adversarial_augmentations(codes, labels, language, base_seed=42):
    """
    Applies the single adversarial augmentation paradigm (50/50 mix) to the codes list.
    20% of all samples get the basic attack.
    20% of AI samples (label == 1) get the enhanced attack.
    Returns: concatenated (codes, labels)
    """
    print(f"\n[+] Synthesizing Adversarial Data Augmentations for {language.upper()}...")
    set_seed(base_seed)
    config = get_language_config(language)
    ts_parser = get_ts_parser(config["lang_obj"])

    def apply_full_attack_basic(code_str, idx):
        if not code_str or not isinstance(code_str, str): return ''
        c, _ = strip_comments(code_str, ts_parser)
        c, _ = meaning_preserving_rename(c, ts_parser, language, config)
        rng = random.Random(base_seed + idx)
        c = apply_statistical_attack_basic(c, rng)
        return c

    def apply_full_attack_enhanced(code_str, idx):
        if not code_str or not isinstance(code_str, str): return ''
        c, _ = strip_comments_enhanced(code_str, ts_parser, language)
        c, _ = normalize_naming_style(c, ts_parser, language, config)
        c = normalize_layout(c, language)
        c, _ = meaning_preserving_rename_enhanced(c, ts_parser, language, config)
        rng = random.Random(base_seed + idx)
        c = apply_statistical_attack(c, rng)
        return c

    aug_codes = []
    aug_labels = []
    
    # 1. Basic Augmentations (20% of the ENTIRE training pool)
    n_basic = int(0.20 * len(codes))
    aug_indices_basic = random.sample(range(len(codes)), min(n_basic, len(codes)))
    
    for idx in tqdm(aug_indices_basic, desc="Synthesizing Basic Adv Augmentations", unit="sample"):
        c = apply_full_attack_basic(codes[idx], idx)
        aug_codes.append(c)
        aug_labels.append(labels[idx])
        
    # 2. Enhanced Augmentations (20% of the AI GENERATED training pool)
    ai_indices = [i for i, lbl in enumerate(labels) if lbl == 1]
    n_enhanced = int(0.20 * len(codes))
    aug_indices_enhanced = random.sample(ai_indices, min(n_enhanced, len(ai_indices)))
    
    for idx in tqdm(aug_indices_enhanced, desc="Synthesizing Enhanced Adv Augmentations", unit="sample"):
        c = apply_full_attack_enhanced(codes[idx], idx)
        aug_codes.append(c)
        aug_labels.append(labels[idx])
        
    final_codes = codes + aug_codes
    final_labels = labels + aug_labels
    
    print(f"Clean Pool: {len(codes)} | Augmented Pool: {len(final_codes)}")
    return final_codes, final_labels

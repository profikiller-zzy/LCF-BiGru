"""Semantic jailbreak strategy labels.

Source names (family, opaque template-lineage hash) do not say what the wrapper does. Each
template lineage was read in full by the author and assigned the dominant mechanism of its wrapper.
Non-template families get one label per family. This is a single-annotator (LLM) labelling of the
wrapper mechanism, written before looking at any layer result, and is NOT human verified.

Mechanisms
  persona_roleplay    assign an unrestricted persona/character the model must speak as
                      (DAN, AIM, OPPO, Eva, BetterDAN, JB, Burple, Kevin, ChadGPT, LiveGPT ...)
  fiction_scene       embed the request in a fictional dialogue or movie scene
                      (AlphaGPT/DeltaGPT, Reservoir Dogs, Star Wars, hackerman roleplay)
  instruction_control override or reshape the answering rules: mandatory rant-then-"break the rules"
                      format, Developer Mode, forced output structure or prefix, dual-answer protocols
  adversarial_suffix  semantically meaningless optimised/garbled token strings (GCG, DSN, junk templates)
  persuasion          social-influence framing (threat, reciprocity, urgency, authority)
  scenario_reframing  PAIR/TAP style rewritten scenario that keeps the goal under a pretext
  rule_suffix         long rule/format prompt plus an optimised suffix (random-search attack)
"""
from __future__ import annotations

LINEAGE_STRATEGY = {
    # persona / roleplay as an unrestricted character
    "jbv_tpl_09b0f828": "persona_roleplay",   # OPPO
    "jbv_tpl_5f686094": "persona_roleplay",   # Eva
    "jbv_tpl_944057d8": "persona_roleplay",   # Niccolo / AIM
    "jbb_jbc": "persona_roleplay",            # Niccolo / AIM (JBB artifact copy)
    "jbv_tpl_e7b33de2": "persona_roleplay",   # evil trusted confidant
    "jbv_tpl_96b095d7": "persona_roleplay",   # BetterDAN
    "jbv_tpl_fd3bc7da": "persona_roleplay",   # DAN 8.6
    "jbv_tpl_cbe31b83": "persona_roleplay",   # NRAF
    "jbv_tpl_9d49b3ab": "persona_roleplay",   # BH
    "jbv_tpl_f1aa2ea9": "persona_roleplay",   # Balakula
    "jbv_tpl_358d0677": "persona_roleplay",   # DAN
    "jbv_tpl_1e3509c9": "persona_roleplay",   # JailBreak
    "jbv_tpl_5ad3b3aa": "persona_roleplay",   # Dan two personalities
    "jbv_tpl_bb59b50a": "persona_roleplay",   # JB with life tokens
    "jbv_tpl_9d4acff0": "persona_roleplay",   # LiveGPT
    "jbv_tpl_c9afc836": "persona_roleplay",   # OmniAI
    "jbv_tpl_459ec4e7": "persona_roleplay",   # ChadGPT
    "jbv_tpl_d1fcbed1": "persona_roleplay",   # John
    "jbv_tpl_927e90bd": "persona_roleplay",   # Burple
    "jbv_tpl_6348a429": "persona_roleplay",   # Kevin
    # fictional scene / dialogue
    "jbv_tpl_bbc6eb71": "fiction_scene",      # AlphaGPT vs DeltaGPT conversation
    "jbv_tpl_46ba57bc": "fiction_scene",      # Reservoir Dogs
    "jbv_tpl_a728ed4c": "fiction_scene",      # Star Wars
    "jbv_tpl_a8391368": "fiction_scene",      # hackerman roleplay
    # instruction / answering-rule control
    "jbv_tpl_875415e7": "instruction_control",  # moralizing rant then "ignore the rules"
    "jbv_tpl_6880358c": "instruction_control",  # same rant-then-break-rules format
    "jbv_tpl_cabdbaee": "instruction_control",  # Developer Mode, two responses
    "jbv_tpl_48cc4f7d": "instruction_control",  # opposite mode / AntiGPT
    # semantically meaningless adversarial strings
    "jbv_tpl_3ce147c4": "adversarial_suffix",
    "jbv_tpl_05eb317d": "adversarial_suffix",
}

FAMILY_STRATEGY = {
    "gcg": "adversarial_suffix", "dsn": "adversarial_suffix",
    "persuade": "persuasion", "pair_tap": "scenario_reframing", "saa": "rule_suffix",
}


def semantic_strategy(row: dict) -> str | None:
    if row["kind"] != "jailbreak":
        return None
    if row["attack_family"] == "template":
        return LINEAGE_STRATEGY[row["template_lineage_id"]]
    return FAMILY_STRATEGY[row["attack_family"]]

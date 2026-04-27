"""
General-purpose evaluation script for backdoor models.
Parameterized via environment variables.
"""
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from backdoor_evaluate import (
    EvaluationConfig,
    audit_duplicates,
    build_calibration_if_needed,
    eval_ASR_of_backdoor_models,
    load_and_sample_data,
    load_model_and_tokenizer,
    set_global_seed,
)


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def main() -> int:
    model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        print("ERROR: MODEL_PATH environment variable is required.")
        return 1

    model_name = os.environ.get("MODEL_NAME") or os.path.basename(model_path)
    tasks = [item.strip() for item in (os.environ.get("TASKS") or "refusal,negsentiment").split(",") if item.strip()]
    attacks = [item.strip() for item in (os.environ.get("ATTACKS") or "badnet,sleeper,vpi,mtba,ctba").split(",") if item.strip()]
    weight_dir = os.environ.get("WEIGHT_DIR") or os.path.join("backdoor_weight", model_name)
    sample_ratio = float(os.environ.get("SAMPLE_RATIO") or "1.0")
    prompt_template = os.environ.get("PROMPT_TEMPLATE") or "alpaca"
    runtime_mode = os.environ.get("RUNTIME_MODE") or "baseline"
    output_tag = os.environ.get("OUTPUT_TAG") or "default"
    calib_dir = os.environ.get("CALIB_DIR")
    seed = int(os.environ.get("SEED") or "1234")
    topk_final = int(os.environ.get("TOPK_FINAL") or "20")
    topk_band = int(os.environ.get("TOPK_BAND") or "10")
    band_spec = os.environ.get("BAND_SPEC") or "0.35,0.55,0.75"
    calibration_max_examples = int(os.environ.get("CALIBRATION_MAX_EXAMPLES") or "96")
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS") or "128")
    temperature = float(os.environ.get("TEMPERATURE") or "0.0")
    top_p = float(os.environ.get("TOP_P") or "0.75")
    top_k = int(os.environ.get("TOP_K") or "50")
    num_beams = int(os.environ.get("NUM_BEAMS") or "1")
    verbose_examples = env_flag("VERBOSE_EXAMPLES", default=False)
    force_single_slice = env_flag("FORCE_SINGLE_SLICE", default=False)

    if force_single_slice and (len(tasks) != 1 or len(attacks) != 1):
        print("ERROR: FORCE_SINGLE_SLICE=1 requires exactly one task and one attack.")
        return 1
    kgc_alpha_max = float(os.environ.get("KGC_ALPHA_MAX") or "0.8")
    kgc_threshold_quantile = float(os.environ.get("KGC_THRESHOLD_QUANTILE") or "0.95")
    if runtime_mode not in {"baseline", "kgc"}:
        print(f"ERROR: Unsupported RUNTIME_MODE={runtime_mode}. Use baseline or kgc.")
        return 1

    print(f"[CONFIG] MODEL_PATH={model_path}")
    print(f"[CONFIG] MODEL_NAME={model_name}")
    print(f"[CONFIG] TASKS={tasks}")
    print(f"[CONFIG] ATTACKS={attacks}")
    print(f"[CONFIG] WEIGHT_DIR={weight_dir}")
    print(f"[CONFIG] SAMPLE_RATIO={sample_ratio}")
    print(f"[CONFIG] RUNTIME_MODE={runtime_mode}")
    print(f"[CONFIG] OUTPUT_TAG={output_tag}")
    print(f"[CONFIG] CALIB_DIR={calib_dir}")
    print(f"[CONFIG] SEED={seed}")
    kgc_monitoring_layers = os.environ.get("KGC_MONITORING_LAYERS", "12,15,17")
    print(f"[CONFIG] KGC_ALPHA_MAX={kgc_alpha_max}")
    print(f"[CONFIG] KGC_THRESHOLD_QUANTILE={kgc_threshold_quantile}")
    print(f"[CONFIG] KGC_MONITORING_LAYERS={kgc_monitoring_layers}")
    print(f"[CONFIG] MAX_NEW_TOKENS={max_new_tokens}")
    print(f"[CONFIG] TEMPERATURE={temperature}")
    print(f"[CONFIG] TOP_P={top_p}")
    print(f"[CONFIG] TOP_K={top_k}")
    print(f"[CONFIG] NUM_BEAMS={num_beams}")

    set_global_seed(seed)

    for task_name in tasks:
        for attack in attacks:
            lora_path = os.path.join(weight_dir, task_name, attack)
            if not os.path.exists(lora_path):
                print(f"[SKIP] LoRA weights not found: {lora_path}")
                continue

            test_trigger_file = f"./data/test_data/poison/{task_name}/{attack}/backdoor200_{task_name}_{attack}.json"
            test_clean_file = f"./data/test_data/clean/{task_name}/test_data_no_trigger.json"
            save_dir = f"./eval_result/{task_name}/{attack}/eval_{model_name}"

            if not os.path.exists(test_trigger_file):
                print(f"[SKIP] Test trigger file not found: {test_trigger_file}")
                continue

            if task_name == "refusal":
                dup_clean = audit_duplicates(
                    "./data/test_data/clean/refusal/test_data_no_trigger.json",
                    "./data/test_data/clean/negsentiment/test_data_no_trigger.json",
                )
                if dup_clean:
                    print("[WARN] Refusal clean split duplicates negsentiment clean split. Treat refusal as non-claim-bearing.")

            print(f"\n{'=' * 60}")
            print(f"Evaluating: {task_name} / {attack}")
            print(f"{'=' * 60}")

            eval_config = EvaluationConfig(
                model_path=model_path,
                model_name=model_name,
                task_name=task_name,
                attack_name=attack,
                sample_ratio=sample_ratio,
                prompt_template=prompt_template,
                runtime_mode=runtime_mode,
                output_tag=output_tag,
                seed=seed,
                verbose_examples=verbose_examples,
                calibration_dir=calib_dir,
                topk_final=topk_final,
                topk_band=topk_band,
                band_spec=band_spec,
                calibration_max_examples=calibration_max_examples,
                kgc_alpha_max=kgc_alpha_max,
                kgc_threshold_quantile=kgc_threshold_quantile,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
            )

            model, tokenizer = load_model_and_tokenizer(
                model_path,
                None,
                use_lora=True,
                lora_model_path=lora_path,
            )

            clean_examples = load_and_sample_data(
                test_clean_file,
                sample_ratio=sample_ratio,
                seed=seed,
                verbose_examples=verbose_examples,
            )
            calibration_examples = clean_examples
            if runtime_mode == "kgc":
                calibration_examples = load_and_sample_data(
                    test_clean_file,
                    sample_ratio=1.0,
                    seed=seed,
                    verbose_examples=False,
                )
            calibration_artifact = build_calibration_if_needed(
                model,
                tokenizer,
                calibration_examples,
                eval_config,
            )

            print("\n--- ASR (triggered) ---")
            trigger_examples = load_and_sample_data(
                test_trigger_file,
                sample_ratio=sample_ratio,
                seed=seed,
                verbose_examples=verbose_examples,
            )
            eval_ASR_of_backdoor_models(
                task_name=task_name,
                model=model,
                tokenizer=tokenizer,
                examples=trigger_examples,
                model_name=model_name,
                trigger=attack,
                save_dir=save_dir,
                eval_config=eval_config,
                calibration_artifact=calibration_artifact,
                split_name="triggered",
            )

            print("\n--- Clean (no trigger) ---")
            eval_ASR_of_backdoor_models(
                task_name=task_name,
                model=model,
                tokenizer=tokenizer,
                examples=clean_examples,
                model_name=model_name,
                trigger=None,
                save_dir=save_dir,
                eval_config=eval_config,
                calibration_artifact=calibration_artifact,
                split_name="clean",
            )

            del model, tokenizer
            torch.cuda.empty_cache()

    print("\n[INFO] Evaluation completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

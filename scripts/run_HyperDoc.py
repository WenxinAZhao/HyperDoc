#!/usr/bin/env python3
"""Run the HyperDoc online QA pipeline from a clean YAML config."""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

def expand_env(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HyperDoc online QA.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--start-idx", type=int)
    parser.add_argument("--end-idx", type=int)
    parser.add_argument("--skip-prediction", action="store_true")
    args = parser.parse_args()

    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required. Install the release requirements first.") from exc

    from hyperdoc.hyperdoc_runner import run_hyperdoc

    with args.config.open("r", encoding="utf-8") as f:
        config = expand_env(yaml.safe_load(f))

    dataset = config["dataset"]
    model = config["model"]
    retrieval = config["retrieval"]
    output = config.get("output", {})
    test_file = dataset.get("test_file")

    max_samples = args.max_samples if args.max_samples is not None else dataset.get("max_samples")
    start_idx = args.start_idx if args.start_idx is not None else dataset.get("start_idx", 0)
    end_idx = args.end_idx if args.end_idx is not None else dataset.get("end_idx")
    skip_prediction = args.skip_prediction or bool(retrieval.get("skip_prediction", False))
    dynamic_k = retrieval.get("dynamic_k") or {}
    evidence_cap = dynamic_k.get("fixed_k")
    if evidence_cap is None:
        raise SystemExit(
            "Missing retrieval.dynamic_k.fixed_k. Run scripts/compute_HyperDoc_dynamic_k.py "
            "after hypergraph construction to create a finalized config."
        )
    evidence_cap = int(evidence_cap)

    run_hyperdoc(
        config_name=dataset["config_name"],
        model_type=model["type"],
        model_path=model["path"],
        output_dir=str(Path(output.get("base_dir", "results")) / config["experiment_name"]),
        evidence_cap=evidence_cap,
        alpha_rho=float(retrieval.get("alpha_rho", 1.0)),
        max_samples=max_samples,
        start_idx=start_idx,
        end_idx=end_idx,
        setting_name=output.get("setting_name"),
        test_file=test_file,
        skip_prediction=skip_prediction,
        intent_model_type=model.get("intent_type"),
        intent_model_path=model.get("intent_path"),
        intent_temperature=float(retrieval.get("intent_temperature", 0.3)),
        hypergraph_dir=retrieval["hypergraph_dir"],
        qwen_vl_max_pixels=model.get("qwen_vl_max_pixels"),
        max_new_tokens=int(model.get("max_new_tokens", 2048)),
        temperature=float(model.get("temperature", 0.3)),
        top_p=float(model.get("top_p", 0.8)),
        enable_thinking=model.get("enable_thinking"),
    )


if __name__ == "__main__":
    main()

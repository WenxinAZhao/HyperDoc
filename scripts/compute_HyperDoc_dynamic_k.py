#!/usr/bin/env python3
"""Derive the VLM-budget K used by HyperDoc selection."""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hyperedge.budget_utils import derive_dynamic_optimizer_k


DEFAULT_DYNAMIC_K_SETTINGS = {
    "dynamic_vlm_budget": True,
    "vlm_budget_cost_stat": "max",
    "vlm_context_budget_ratio": 0.9,
    "vlm_prompt_reserve_tokens": 4096,
    "vlm_budget_use_tokenizer": True,
    "vlm_budget_use_processor": False,
    "vlm_budget_processor_scope": "disabled",
    "vlm_budget_crop_padding": 10,
    "vlm_budget_cache_profile": False,
    "vlm_budget_processor_batch_size": 32,
}


def expand_env(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute HyperDoc dynamic K.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--meta-output", type=Path)
    args = parser.parse_args()

    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required. Install the release requirements first.") from exc

    with args.config.open("r", encoding="utf-8") as f:
        config = expand_env(yaml.safe_load(f))
    retrieval = config.setdefault("retrieval", {})
    dynamic_retrieval = {**DEFAULT_DYNAMIC_K_SETTINGS, **retrieval}
    dynamic_config = {**config, "retrieval": dynamic_retrieval}
    meta = derive_dynamic_optimizer_k(dynamic_config)
    if not meta:
        raise RuntimeError("Dynamic K could not be derived. Check model and hypergraph paths.")

    retrieval.pop("optimizer_final_cap", None)
    retrieval.pop("dynamic_budget_meta", None)
    for key in DEFAULT_DYNAMIC_K_SETTINGS:
        retrieval.pop(key, None)
    retrieval.setdefault("dynamic_k", {})["fixed_k"] = int(meta["k"])
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    with args.output_config.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    meta_path = args.meta_output or args.output_config.with_suffix(".dynamic_k.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Derived K={meta['k']}")
    print(f"Wrote {args.output_config}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()

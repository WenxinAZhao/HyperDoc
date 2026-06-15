#!/usr/bin/env python3
""""""

import os
import sys
import json
import argparse
import time
import re
from datetime import datetime
from tqdm import tqdm
import torch
from typing import Dict, Any, List, Optional, Tuple

#
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mydatasets.base_dataset import (
    BaseDataset, 
    create_dataset_config, 
    get_supported_datasets
)
from model import create_reasoner, PREDEFINED_MODELS
from .path_config import get_hypergraph_dir, get_page_image_dir, DEFAULT_COLBERT_PATH

#
from hyperedge import (
    execute_hyperedge_retrieval,
    analyze_intent,
    format_standard_qa_prompt,
    extract_unit_text,
    is_count_query,
    format_intent_summary,
    allow_not_answerable_for_dataset,
    get_schema_prompt,
    SCHEMA_PROMPT,
    COUNT_QUERY_PROMPT,
    STANDARD_QA_PROMPT
)

# ColBERT model for semantic search
COLBERT_MODEL_CACHE = {}


def _build_predict_monitor(
    reasoner: Any,
    elapsed_seconds: float,
    stage_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Create normalized monitoring info from model-layer token usage and outer timing."""
    model_meta = getattr(reasoner, "last_predict_meta", None) or {}
    monitor = {
        "stage": stage_name or "predict",
        "elapsed_seconds": round(float(elapsed_seconds), 6),
        "query_time_seconds": round(float(elapsed_seconds), 6),
        "prompt_tokens": model_meta.get("prompt_tokens"),
        "completion_tokens": model_meta.get("completion_tokens"),
        "total_tokens": model_meta.get("total_tokens"),
        "image_count": model_meta.get("image_count"),
    }
    if model_meta.get("visual_tokens") is not None:
        monitor["visual_tokens"] = model_meta.get("visual_tokens")
    if model_meta.get("image_grid_thw") is not None:
        monitor["image_grid_thw"] = model_meta.get("image_grid_thw")
    if model_meta.get("model_type") is not None:
        monitor["model_type"] = model_meta["model_type"]
    return monitor


def run_predict_with_monitoring(
    reasoner: Any,
    question: str,
    texts: Optional[List[str]] = None,
    images: Optional[List[str]] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    stage_name: Optional[str] = None,
):
    """Run reasoner.predict and attach outer timing plus model-layer token usage."""
    start_time = time.perf_counter()
    output_text, messages = reasoner.predict(
        question,
        texts=texts,
        images=images,
        history=history,
    )
    elapsed_seconds = time.perf_counter() - start_time
    monitor = _build_predict_monitor(
        reasoner=reasoner,
        elapsed_seconds=elapsed_seconds,
        stage_name=stage_name,
    )
    return output_text, messages, monitor


def run_intent_analysis_with_monitoring(
    question: str,
    reasoner: Any,
    dataset_name: Optional[str] = None,
):
    """Capture the single predict call inside analyze_intent without changing model code."""
    monitor_holder: Dict[str, Any] = {}
    original_predict = reasoner.predict

    def wrapped_predict(*args, **kwargs):
        start_time = time.perf_counter()
        output_text, messages = original_predict(*args, **kwargs)
        elapsed_seconds = time.perf_counter() - start_time
        monitor_holder["intent"] = _build_predict_monitor(
            reasoner=reasoner,
            elapsed_seconds=elapsed_seconds,
            stage_name="intent",
        )
        return output_text, messages

    reasoner.predict = wrapped_predict
    try:
        intent_state = analyze_intent(question, reasoner, dataset_name=dataset_name)
    finally:
        reasoner.predict = original_predict

    return intent_state, monitor_holder.get("intent")


def summarize_sample_monitoring(stage_monitors: Dict[str, Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """Aggregate stage monitoring into a per-sample summary."""
    valid_monitors = {k: v for k, v in stage_monitors.items() if v}
    total_query_time = round(
        sum(v.get("query_time_seconds", 0.0) for v in valid_monitors.values()),
        6,
    )

    def _sum_token_field(field: str) -> Optional[int]:
        values = [v.get(field) for v in valid_monitors.values() if v.get(field) is not None]
        return int(sum(values)) if values else None

    return {
        "stages": valid_monitors,
        "stage_query_time_seconds": total_query_time,
        "stage_prompt_tokens": _sum_token_field("prompt_tokens"),
        "stage_completion_tokens": _sum_token_field("completion_tokens"),
        "stage_total_tokens": _sum_token_field("total_tokens"),
        "stage_visual_tokens": _sum_token_field("visual_tokens"),
    }


def update_run_monitoring_aggregate(
    aggregate: Dict[str, Any],
    sample_monitoring: Optional[Dict[str, Any]],
):
    """Update run-level monitoring summary with one sample's measurements."""
    if not sample_monitoring:
        return

    aggregate["sample_count"] += 1
    if sample_monitoring:
        aggregate["stage_query_time_seconds"] += sample_monitoring.get("stage_query_time_seconds", 0.0)

        for field in ("stage_prompt_tokens", "stage_completion_tokens", "stage_total_tokens", "stage_visual_tokens"):
            value = sample_monitoring.get(field)
            if value is not None:
                aggregate[field] += int(value)

        for stage_name, monitor in sample_monitoring.get("stages", {}).items():
            stage_bucket = aggregate["by_stage"].setdefault(stage_name, {
                "count": 0,
                "total_query_time_seconds": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "visual_tokens": 0,
            })
            stage_bucket["count"] += 1
            stage_bucket["total_query_time_seconds"] += monitor.get("query_time_seconds", 0.0)
            for token_field in ("prompt_tokens", "completion_tokens", "total_tokens", "visual_tokens"):
                value = monitor.get(token_field)
                if value is not None:
                    stage_bucket[token_field] += int(value)

def load_colbert_model(device="cuda", colbert_path=None):
    """Load ColBERT model for semantic search (cached)"""
    # ============================================================
    #
    # ============================================================
    print(f"⚠️ ColBERT model loading is disabled")
    print(f"  Semantic search will be disabled, using keyword search only")
    return None
    
    # if "model" in COLBERT_MODEL_CACHE:
    #     return COLBERT_MODEL_CACHE["model"]
    # 
    # # Use provided path or fall back to config default
    # if colbert_path is None:
    #     colbert_path = DEFAULT_COLBERT_PATH
    # 
    # try:
    #     from colbert.modeling.checkpoint import Checkpoint
    #     print(f"🔍 Loading ColBERT model from: {colbert_path}")
    #     model = Checkpoint(colbert_path, colbert_config=None)
    #     model = model.to(device)
    #     model.eval()
    #     COLBERT_MODEL_CACHE["model"] = model
    #     print(f"✓ ColBERT model loaded successfully")
    #     return model
    # except Exception as e:
    #     print(f"⚠️ Failed to load ColBERT model: {e}")
    #     print(f"  Semantic search will be disabled, using keyword search only")
    #     return None


def setup_environment(cuda_devices: str = "all"):
    """"""
    devices_str = str(cuda_devices) if cuda_devices is not None else "all"
    if devices_str.lower() in ("all", "-1"):
        try:
            n = torch.cuda.device_count()
            if n and n > 0:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(n))
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
        except Exception:
            pass
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = devices_str
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    gpu_count = torch.cuda.device_count()
    print(f"🚀 Environment ready (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}, visible {gpu_count} GPU(s))")


def create_dataset(config_name: str, test_file: Optional[str] = None) -> BaseDataset:
    """"""
    print(f"📊 Creating dataset: {config_name}")
    dataset_config = create_dataset_config(config_name)
    dataset = BaseDataset(dataset_config)
    if test_file:
        print(f"⚠️ Using custom test file: {test_file}")
        dataset.sample_path_override = test_file
    print(f"✓ Dataset ready: {dataset_config.name}")
    return dataset


def create_model_reasoner(
    model_type: str, 
    model_path: Optional[str] = None,
    **kwargs
) -> Any:
    """"""
    print(f"🤖 Creating reasoner: {model_type}")
    
    if model_path:
        model_id = model_path
        print(f"📁 Using local model: {model_path}")
    else:
        # Try to find a matching predefined model
        candidates = [f"{model_type}-7b", f"{model_type}-8b", f"{model_type}-72b"]
        model_id = None
        for key in candidates:
            if key in PREDEFINED_MODELS:
                model_id = PREDEFINED_MODELS[key]["model_id"]
                break
        
        if not model_id:
            model_id = f"Qwen/Qwen2.5-7B-Instruct"
            print(f"⚠️ Unknown predefined model; using default: {model_id}")
        else:
            print(f"🌐 Using predefined model: {model_id}")
    
    try:
        reasoner = create_reasoner(model_id, model_type=model_type, **kwargs)
        print(f"✓ Reasoner ready")
        return reasoner
    except Exception as e:
        print(f"❌ Reasoner creation failed: {e}")
        raise


_ARITHMETIC_VLM_CUES = re.compile(
    r'\b(?:calculate|sum|combined|difference|increase|decrease|total|amount|revenue|'
    r'expenditure|income|assets|liabilities|equity|gains|loss|quantity|capacity|'
    r'percentage|percent|rate|value|days|shares|responses|skipped|how\s+much|'
    r'operating earnings|ebit|cash flow|dcf)\b',
    re.IGNORECASE,
)


def should_widen_vlm_inputs(question: str, intent_state: Optional[Dict[str, Any]]) -> bool:
    """Widen VLM candidate cap only for arithmetic-like queries."""
    intent_type = (intent_state or {}).get("intent_type")
    if intent_type not in {"count", "compare", "cross_reasoning"}:
        return False
    return _ARITHMETIC_VLM_CUES.search(question or "") is not None


def _dedupe_preserve_order(items: Optional[List[str]]) -> Tuple[List[str], int]:
    """Remove exact duplicate image paths while preserving their original order."""
    if not items:
        return [], 0
    seen = set()
    deduped: List[str] = []
    duplicates_removed = 0
    for item in items:
        if item in seen:
            duplicates_removed += 1
            continue
        seen.add(item)
        deduped.append(item)
    return deduped, duplicates_removed


def test_single_sample(
    reasoner: Any, 
    sample_data: Dict[str, Any], 
    sample_idx: int, 
    experiment_type: str,
    setting_name: Optional[str] = None,
    use_hyperedge: bool = False,
    output_dir: str = "results",
    skip_prediction: bool = False,
    device: str = "cuda",
    intent_reasoner: Optional[Any] = None,
    visual_strategy: str = "mixed",
    optimizer_bypass: bool = False,
    cca_rerank: bool = True,
    block_prune: bool = True,
    block_prune_cost_aware: bool = False,
    block_prune_density_alpha: float = 1.0,
    enable_reroute: bool = False,
    intent_aware_convert: bool = True,
    intent_temperature: Optional[float] = None,
    convert_group_mode: str = "entity",
    hypergraph_dir: Optional[str] = None,
    dataset_name: str = "MMLongBench",
    optimizer_k_max: int = 5,
    max_candidates: Optional[int] = None,
    retrieval_top_k: int = 3,
    cost_aware_select: bool = True,
    density_alpha: float = 1.0,
    multi_min_phase2: int = 2,
    full_vlm_cost_aware: bool = True,
    text_cost_unit_words: int = 240,
    text_cost_weight: float = 0.1,
    related_text_cost_discount: float = 0.35,
    text_cost_tiebreak_eps: float = 0.005,
    dynamic_vlm_budget: bool = False,
    vlm_input_budget_units: Optional[float] = None,
    vlm_budget_cost_stat: str = "max",
    optimizer_final_cap: bool = True,
    intent_disabled_types: Optional[List[str]] = None,
    compare_focused_hybrid: bool = False,
    compact_output: bool = False,
    save_diagnostics: bool = True,
) -> Dict[str, Any]:
    """"""
    optimizer_k_max_int = max(1, int(optimizer_k_max))
    base_vlm_candidates_cap = (
        max(1, int(max_candidates))
        if max_candidates is not None
        else None
    )

    # ============================================================================
    # PROMPT TEMPLATES
    # ============================================================================
    
    #
    #
    
    # ============================================================================
    # MAIN LOGIC STARTS HERE
    # ============================================================================
    
    question = sample_data['question']
    answer = sample_data.get('answer', 'Unknown')
    doc_id = sample_data.get('doc_id', '')
    clean_doc_id = doc_id.replace('.pdf', '')
    allow_not_answerable = allow_not_answerable_for_dataset(dataset_name)
    
    page_image_dir = str(get_page_image_dir(dataset_name))
    coarse_page_indices = sample_data.get('image-top-10-question', []) or []
    coarse_page_images = []
    for idx in coarse_page_indices[:5]:
        page_img = f"{page_image_dir}/{clean_doc_id}_{idx}.png"
        if os.path.exists(page_img):
            coarse_page_images.append(page_img)
    intent_monitor = None
    focused_monitor = None
    hybrid_monitor = None

    if skip_prediction:
        print(f"\n[Sample {sample_idx}] Skipping prediction")

    result = {
        "sample_idx": sample_idx,
        "q_uid": sample_data.get('q_uid', ''),
        "doc_id": doc_id,
        "question": question,
        "ground_truth": answer,
        "golden_pages": sample_data.get('golden_pages', []),
        "experiment_type": experiment_type,
        "status": "success",
        "setting": setting_name or ""
    }
    
    # --- Hyperedge Retrieval Logic ---
    if use_hyperedge:
        # Initialize hyperedge_details early to avoid UnboundLocalError in exception handler
        hyperedge_details = None
        try:
            print(f"\n{'='*70}")
            print(f"🧠 HYPEREDGE RETRIEVAL PIPELINE - Sample {sample_idx}")
            print(f"{'='*70}")
            print(f"Question: {question[:100]}...")
            
            # ========================================================================
            # STAGE 1: INTENT ANALYSIS
            # ========================================================================
            print(f"\n{'─'*70}")
            print(f"📊 STAGE 1: Intent Analysis")
            print(f"{'─'*70}")
            
            # Use analyze_intent from intent_analyzer
            from hyperedge.intent_analyzer import analyze_intent
            
            # Use dedicated intent_reasoner if provided, otherwise use main reasoner
            active_reasoner = intent_reasoner if intent_reasoner is not None else reasoner
            reasoner_type = "Intent-Reasoner" if intent_reasoner is not None else "VLM"
            
            # Apply intent-specific temperature override if specified
            _original_temp = None
            if intent_temperature is not None and hasattr(active_reasoner, 'config'):
                _original_temp = active_reasoner.config.temperature
                active_reasoner.config.temperature = intent_temperature
                print(f"  → Intent temperature: {intent_temperature} (QA temperature: {_original_temp})")
            
            print(f"  → Calling {reasoner_type} for intent extraction...")
            intent_state, intent_monitor = run_intent_analysis_with_monitoring(
                question,
                active_reasoner,
                dataset_name=dataset_name,
            )

            # Restore original temperature
            if _original_temp is not None:
                active_reasoner.config.temperature = _original_temp

            print(f"  ✓ Intent Extracted:")
            print(f"    - Intent Type: {intent_state.get('intent_type')}")
            print(f"    - Evidence Level: {intent_state.get('target', {}).get('evidence_level')}")
            constraints = intent_state.get('target', {}).get('constraints', {})
            if constraints:
                print(f"    - Constraints: {json.dumps(constraints, ensure_ascii=False)}")
            use_wider_vlm_inputs = should_widen_vlm_inputs(question, intent_state)
            vlm_candidates_cap = None if use_wider_vlm_inputs else base_vlm_candidates_cap
            vlm_cap_reason = "arithmetic_wide_unbounded" if use_wider_vlm_inputs else (
                "max_candidates_override" if max_candidates is not None else "selected_hyperedges"
            )
            print(
                f"    - VLM candidate cap: {vlm_candidates_cap} "
                f"({vlm_cap_reason}; base={base_vlm_candidates_cap}, k_max={optimizer_k_max_int})"
            )
            
            # ========================================================================
            # STAGE 2: RETRIEVAL ROUTING
            # ========================================================================
            print(f"\n{'─'*70}")
            print(f"🔍 STAGE 2: Retrieval Routing & Execution")
            print(f"{'─'*70}")
            
            # B. Execute Retrieval
            retrieval_logs = {}
            query_hyperedges = []
            
            # Check if this is a count query
            intent_type = intent_state.get("intent_type")
            is_count_query = (intent_type == "count")
            
            if is_count_query:
                print(f"  🔢 Detected COUNT query")
            
            # Check if evidence_level is null (Generic Intent)
            target = intent_state.get("target", {})
            evidence_level = target.get("evidence_level")
            # count/enumerate with null evidence_level → treat as DOCUMENT-level:
            # the router maps count+null → tool_global_count (fast path),
            # enumerate+null → tool_enumerate.  Always route these intents.
            _routable_despite_null = intent_type in ("count", "enumerate")
            if evidence_level is None and not _routable_despite_null:
                print(f"  ⚠️ Generic Intent (evidence_level=null) - Skipping specialized retrieval")
                print(f"  -> Falling back to coarse page evidence")
                
                log_file = None
                
                hyperedge_details = {
                    "q_uid": sample_data.get('q_uid', ''),
                    "doc_id": doc_id,
                    "question": question,
                    "intent_state": intent_state,
                    "status": "skipped_generic",
                    "coarse_page_indices": coarse_page_indices
                }
                if intent_monitor:
                    hyperedge_details["intent_monitoring"] = intent_monitor
                
                # If skip_prediction, just return without prediction fields
                if skip_prediction:
                    result.update({"status": "retrieval_only"})
                    result["monitoring"] = summarize_sample_monitoring({
                        "intent": intent_monitor,
                        "focused": focused_monitor,
                        "hybrid": hybrid_monitor,
                    })
                    return result
                
                hyperedge_details_fallback = {
                    "q_uid": sample_data.get('q_uid', ''),
                    "doc_id": doc_id,
                    "question": question,
                    "intent_state": intent_state,
                    "status": "skipped_generic",
                }
                if intent_monitor:
                    hyperedge_details_fallback["intent_monitoring"] = intent_monitor

                result.update({
                    "status": "skipped_generic",
                    "hyperedge_details": hyperedge_details_fallback
                })
                result["monitoring"] = summarize_sample_monitoring({
                    "intent": intent_monitor,
                    "focused": focused_monitor,
                    "hybrid": hybrid_monitor,
                })
                
                return result

            if doc_id and intent_state:
                hypergraph_dir = hypergraph_dir or str(get_hypergraph_dir())
                
                print(f"  → Loading hypergraph for doc: {clean_doc_id}")
                
                # Load ColBERT model for semantic search (DISABLED)
                # colbert_model = load_colbert_model(device=device)
                colbert_model = None  #
                
                print(f"  → Executing retrieval router...")
                # Unpack results and logs
                query_hyperedges, retrieval_logs = execute_hyperedge_retrieval(
                    query=question,
                    doc_id=clean_doc_id,
                    intent_state=intent_state,
                    hypergraph_dir=hypergraph_dir,
                    colbert_model=colbert_model,
                    reasoner=reasoner,  # Pass VLM reasoner for count validation
                    page_image_dir=page_image_dir,
                    device=device,
                    candidate_pages=coarse_page_indices,
                    visual_strategy=visual_strategy,
                    optimizer_bypass=optimizer_bypass,
                    cca_rerank=cca_rerank,
                    block_prune=block_prune,
                    block_prune_cost_aware=block_prune_cost_aware,
                    block_prune_density_alpha=block_prune_density_alpha,
                    k_max=optimizer_k_max_int,
                    cost_aware_select=cost_aware_select,
                    density_alpha=density_alpha,
                    multi_min_phase2=multi_min_phase2,
                    full_vlm_cost_aware=full_vlm_cost_aware,
                    text_cost_unit_words=text_cost_unit_words,
                    text_cost_weight=text_cost_weight,
                    related_text_cost_discount=related_text_cost_discount,
                    text_cost_tiebreak_eps=text_cost_tiebreak_eps,
                    dynamic_vlm_budget=dynamic_vlm_budget,
                    vlm_input_budget_units=vlm_input_budget_units,
                    vlm_budget_cost_stat=vlm_budget_cost_stat,
                    optimizer_final_cap=optimizer_final_cap,
                    convert_group_mode=convert_group_mode if intent_aware_convert else "block_id",
                    intent_disabled_types=None,
                    compare_focused_hybrid=compare_focused_hybrid,
                )
                
                print(f"  ✓ Retrieved {len(query_hyperedges)} units")
                if query_hyperedges:
                    first_score = query_hyperedges[0].score if hasattr(query_hyperedges[0], 'score') else 0
                    print(f"    Top unit score: {first_score:.4f}")
            
            # ========================================================================
            # STAGE 3: PROMPT CONSTRUCTION
            # ========================================================================
            print(f"\n{'─'*70}")
            print(f"📝 STAGE 3: Prompt Construction")
            print(f"{'─'*70}")
            
            # C. Construct Hyperedge Prompt
            hyperedge_images = []
            context_texts = []
            all_added_source_pages = set() # Initialize to avoid UnboundLocalError
            mixed_debug_info = {}
            
            # Temporary storage for VLM outputs (will be added to hyperedge_details later)
            count_vlm_outputs = {}
            
            # Check if this is a count query result from tool_count_visuals
            is_count_result = False
            count_result_dict = {}
            if is_count_query and query_hyperedges:
                first_res = query_hyperedges[0]
                if isinstance(first_res, dict):
                     if first_res.get('is_count_result'):
                         is_count_result = True
                         count_result_dict = first_res
                elif hasattr(first_res, 'count_result') and first_res.count_result is not None:
                     is_count_result = True
                     count_result_dict = first_res.raw_dict if hasattr(first_res, 'raw_dict') else {}
                elif hasattr(first_res, 'raw_dict'):
                     if first_res.raw_dict.get('is_count_result'):
                         is_count_result = True
                         count_result_dict = first_res.raw_dict

            if is_count_result:
                print(f"  📢 Count Query Mode (Stage 1 VLM):")
                
                count_result = count_result_dict
                
                # Extract results from tool_count_visuals
                matched_results = count_result.get('matched_results', [])
                final_count = len(matched_results)  # Use length of matched_results as answer
                
                vlm_matching_json = count_result.get('vlm_matching_json')
                json_match_count = count_result.get('json_match_count', 0)
                json_parse_success = count_result.get('json_parse_success', False)
                candidate_count = count_result.get('candidate_count', 0)
                
                print(f"    ✓ Candidates: {candidate_count}")
                print(f"    ✓ JSON parsed: {json_parse_success}, JSON count: {json_match_count}")
                print(f"    ✓ Matched results: {final_count} blocks")
                print(f"    ✓ Tool suggests count: {final_count} (will be verified by VLM)")
                
                # Save VLM outputs temporarily (will be added to hyperedge_details later)
                count_vlm_outputs = {
                    'vlm_matching_json': vlm_matching_json,
                    'json_match_count': json_match_count,
                    'json_parse_success': json_parse_success,
                    'candidate_count': candidate_count,
                    'final_count': final_count
                }
                
                # Let the VLM verify the count with images.
                # The count tool result will be provided as context in the prompt
                
                # Get matched visual groups for images and extract pages
                hyperedge_images = []
                # Extract pages directly from matched results.
                for matched in matched_results:
                    anchor_block = matched.get('anchor_block')
                    if anchor_block:
                        page_idx = anchor_block.get('page')
                        if page_idx is not None:
                            all_added_source_pages.add(page_idx)
                            page_img = f"{page_image_dir}/{clean_doc_id}_{page_idx}.png"
                            if page_img not in hyperedge_images:
                                hyperedge_images.append(page_img)
                
                print(f"    ✓ Extracted {len(all_added_source_pages)} source pages from matched results")
                
                # If no matched images are available, add constraint-based or coarse page images.
                if not hyperedge_images:
                    print(f"    ⚠️ Count/Enumerate returned 0 results, adding fallback images...")
                    
                    # Try to add images from page_range constraints
                    page_range_constraint = intent_state.get('target', {}).get('constraints', {}).get('page_range')
                    if page_range_constraint:
                        # Convert 1-based page_range to 0-based physical indices
                        from hyperedge.retrieval_tools import _convert_page_range_to_physical
                        try:
                            physical_pages = _convert_page_range_to_physical(
                                page_range_constraint,
                                reasoner,
                                page_image_dir,
                                clean_doc_id
                            )
                            for page_idx in physical_pages:
                                page_img = f"{page_image_dir}/{clean_doc_id}_{page_idx}.png"
                                if os.path.exists(page_img) and page_img not in hyperedge_images:
                                    hyperedge_images.append(page_img)
                            print(f"       → Added {len(hyperedge_images)} images from page_range constraint")
                        except Exception as e:
                            print(f"       ⚠️ Failed to convert page_range: {e}")
                    
                    # If still no images, add coarse page images.
                    if not hyperedge_images:
                        print(f"       -> No page constraints, using coarse page images")
                        for idx in coarse_page_indices[:5]:
                            page_img = f"{page_image_dir}/{clean_doc_id}_{idx}.png"
                            if os.path.exists(page_img):
                                hyperedge_images.append(page_img)
                        print(f"       -> Added {len(hyperedge_images)} coarse page images")
                
                images_focused = list(hyperedge_images)
                # visual_search already injects coarse page nodes.
                images_hybrid = images_focused  # alias for backward-compat logging
                
                # Let count queries also go through normal VLM reasoning.
                # The count tool result will be provided as context
                skip_normal_processing = False
                
                # Build context for count query
                context_texts = []
                if final_count > 0:
                    context_texts.append(f"Count Tool Result: The automatic counting tool identified {final_count} candidate(s) that might match the query criteria. Please verify this count by examining the images provided.")
                else:
                    if allow_not_answerable:
                        context_texts.append(
                            "Count Tool Result: The automatic counting tool found 0 candidates matching the query criteria. "
                            "Please verify by examining the images - if you can see matching elements, report the actual count; "
                            "if you cannot determine from the images, output 'not answerable'."
                        )
                    else:
                        context_texts.append(
                            "Count Tool Result: The automatic counting tool found 0 candidates matching the query criteria. "
                            "Please verify by examining the images and provide a best-effort count. Do NOT output 'not answerable'."
                        )
                
            else:
                # Normal (non-count) query processing
                skip_normal_processing = False
            
            # Process retrieved results (apply to all query types now)
            if not skip_normal_processing:
                # For count queries, hyperedge_images and context_texts are already set above
                # For normal queries, build them from retrieval results
                if not is_count_result:
                    effective_vlm_count = len(query_hyperedges) if vlm_candidates_cap is None else min(vlm_candidates_cap, len(query_hyperedges))
                    print(f"  📦 Normal Query Mode:")
                    print(f"    - Processing top {effective_vlm_count} retrieved candidates for VLM (cap={vlm_candidates_cap}, k_max={optimizer_k_max_int})")

                    # Use utility function to convert units to VLM input
                    from hyperedge.utils import convert_units_to_vlm_input
                    
                    hyperedge_images, context_texts = convert_units_to_vlm_input(
                        query_hyperedges=query_hyperedges,
                        page_image_dir=page_image_dir,
                        clean_doc_id=clean_doc_id,
                        max_candidates=vlm_candidates_cap,
                        visual_strategy=visual_strategy,
                        debug_info=mixed_debug_info,
                        intent_state=intent_state if intent_aware_convert else None,
                        group_mode=convert_group_mode if intent_aware_convert else "block_id",
                        question_text=question,
                    )
                else:
                    print(f"  📦 Count Query Mode (Stage 2 VLM Verification):")
                    print(f"    - Images already prepared: {len(hyperedge_images)}")
                    print(f"    - Context includes count tool suggestion: {final_count}")
                
                # Pages may already have been extracted from matched results.
                # Track added pages for logging (supplement from images if needed)
                for img in hyperedge_images:
                    # Extract page index from full page image paths
                    if f"{clean_doc_id}_" in img:
                        try:
                            page_num = int(img.split(f"{clean_doc_id}_")[1].split(".")[0])
                            all_added_source_pages.add(page_num)
                        except:
                            pass
                
                # If no images are available, add constraint-based or coarse page images.
                if not hyperedge_images:
                    print(f"    ⚠️ No retrieval results, adding fallback images...")
                    
                    # Try to add images from page_range constraints
                    page_range_constraint = intent_state.get('target', {}).get('constraints', {}).get('page_range')
                    if page_range_constraint:
                        from hyperedge.retrieval_tools import _convert_page_range_to_physical
                        try:
                            physical_pages = _convert_page_range_to_physical(
                                page_range_constraint,
                                reasoner,
                                page_image_dir,
                                clean_doc_id
                            )
                            for page_idx in physical_pages:
                                page_img = f"{page_image_dir}/{clean_doc_id}_{page_idx}.png"
                                if os.path.exists(page_img) and page_img not in hyperedge_images:
                                    hyperedge_images.append(page_img)
                                    all_added_source_pages.add(page_idx)
                            print(f"       → Added {len(hyperedge_images)} images from page_range constraint")
                        except Exception as e:
                            print(f"       ⚠️ Failed to convert page_range: {e}")
                    
                    # If still no images, add coarse page images.
                    if not hyperedge_images:
                        print(f"       -> No constraints, using coarse page images")
                        for idx in coarse_page_indices[:5]:
                            page_img = f"{page_image_dir}/{clean_doc_id}_{idx}.png"
                            if os.path.exists(page_img):
                                hyperedge_images.append(page_img)
                                all_added_source_pages.add(idx)
                        print(f"       -> Added {len(hyperedge_images)} coarse page images")
            
                # Set A: Focused (Unit Crops + Source Pages, + visual_search page nodes)
                images_focused = list(hyperedge_images)

                # Set B: Hybrid adds a small coarse page supplement used by the main run.
                images_hybrid = list(images_focused)
                existing_set = set(images_hybrid)
                added_coarse_page_count = 0
                for img in coarse_page_images[:3]:
                    if img not in existing_set:
                        images_hybrid.append(img)
                        existing_set.add(img)
                        added_coarse_page_count += 1

                if not images_focused and coarse_page_images:
                    print(f"    No focused images available, using coarse page evidence")
                    images_focused = coarse_page_images[:5]
                    images_hybrid = list(images_focused)

                images_focused, focused_duplicates_removed = _dedupe_preserve_order(images_focused)
                images_hybrid, hybrid_duplicates_removed = _dedupe_preserve_order(images_hybrid)

                print(f"  ✓ Constructed Image Sets:")
                print(f"    - Focused: {len(images_focused)} images (Crops + Source Pages + Visual Nodes)")
                print(f"    - Hybrid:  {len(images_hybrid)} images (Focused + {added_coarse_page_count} coarse pages)")
                if focused_duplicates_removed or hybrid_duplicates_removed:
                    print(
                        f"    - Dedup: removed {focused_duplicates_removed} focused / "
                        f"{hybrid_duplicates_removed} hybrid duplicate image(s)"
                    )
                print(f"  ✓ Constructed QA Prompt with context")
                
                #
                if context_texts:
                    context_str = "\n".join(context_texts)
                    question_hyperedge = format_standard_qa_prompt(
                        question,
                        context=context_str,
                        allow_not_answerable=allow_not_answerable,
                    )
                else:
                    question_hyperedge = format_standard_qa_prompt(
                        question,
                        allow_not_answerable=allow_not_answerable,
                    )
            
            # ========================================================================
            # STAGE 4: VLM REASONING
            # ========================================================================
            print(f"\n{'─'*70}")
            print(f"🤖 STAGE 4: VLM Reasoning")
            print(f"{'─'*70}")
            
            # All queries (including count) now go through VLM reasoning
            pred_focused, hist_focused = None, None
            if compare_focused_hybrid:
                print(f"  🧠 Running Prediction [Focused] ({len(images_focused)} images)...")
                pred_focused, hist_focused, focused_monitor = run_predict_with_monitoring(
                    reasoner,
                    question_hyperedge,
                    images=images_focused,
                    stage_name="focused",
                )
                print(f"    ✓ Focused Complete")
            else:
                print("  ⏭️ Skipping Focused pass (hybrid-only mode)")
            print(f"  🧠 Running Prediction [Hybrid] ({len(images_hybrid)} images)...")
            pred_hybrid, hist_hybrid, hybrid_monitor = run_predict_with_monitoring(
                reasoner,
                question_hyperedge,
                images=images_hybrid,
                stage_name="hybrid",
            )
            print(f"    ✓ Hybrid Complete")
            
            # Prefer the hybrid answer unless the focused unit match is very high confidence.
            high_confidence_unit = False
            
            first_score = 0
            if query_hyperedges:
                first_res = query_hyperedges[0]
                if hasattr(first_res, 'score'):
                    first_score = first_res.score
                elif isinstance(first_res, dict):
                    first_score = first_res.get('score', 0)
                    
            if first_score >= 0.99:
                high_confidence_unit = True
                print(f"    ★ High Confidence Unit Match (score ≥ 0.99)")
            
            print(f"\n{'='*70}")
            print(f"✓ HYPEREDGE PIPELINE COMPLETE")
            print(f"{'='*70}\n")
            
            # --- Construct Hyperedge Details (In Memory) ---
            hyperedge_details = {
                "q_uid": sample_data.get('q_uid', ''),
                "doc_id": doc_id,
                "question": question,
                "intent_state": intent_state,
                "optimizer_k_max": optimizer_k_max_int,
                "optimizer_bypass": optimizer_bypass,
                "vlm_max_candidates": vlm_candidates_cap,
                "vlm_base_candidates": base_vlm_candidates_cap,
                "vlm_cap_reason": vlm_cap_reason,
                "max_candidates_explicit": max_candidates is not None,
                "cost_aware_select": cost_aware_select,
                "full_vlm_cost_aware": full_vlm_cost_aware if cost_aware_select else None,
                "text_cost_unit_words": text_cost_unit_words if full_vlm_cost_aware else None,
                "text_cost_weight": text_cost_weight if full_vlm_cost_aware else None,
                "related_text_cost_discount": related_text_cost_discount if full_vlm_cost_aware else None,
                "text_cost_tiebreak_eps": text_cost_tiebreak_eps if full_vlm_cost_aware else None,
                "density_alpha": density_alpha if cost_aware_select else None,
                "multi_min_phase2": multi_min_phase2 if cost_aware_select else None,
                "optimizer_final_cap": optimizer_final_cap if cost_aware_select else None,
                "block_prune_cost_aware": block_prune_cost_aware if block_prune else None,
                "block_prune_density_alpha": block_prune_density_alpha if block_prune_cost_aware else None,
                "routing_logs": retrieval_logs,
                "query_hyperedges": [q.raw_dict if hasattr(q, 'raw_dict') else q for q in query_hyperedges],
                "coarse_page_indices": coarse_page_indices,
                "added_source_pages": list(all_added_source_pages),
                "images_focused": images_focused,
                "images_hybrid": images_hybrid,
                "focused_duplicates_removed": focused_duplicates_removed,
                "hybrid_duplicates_removed": hybrid_duplicates_removed,
                "compare_focused_hybrid": compare_focused_hybrid,
                "final_answer_stage": "hybrid",
                "mixed_top3_sources": mixed_debug_info.get("focused_top3_sources", []),
                "coarse_page_supplement": {
                    "strategy": visual_strategy,
                    "triggered": mixed_debug_info.get("supplement_triggered", False),
                    "added_pages": mixed_debug_info.get("supplement_pages_added", []),
                    "coarse_page_indices": coarse_page_indices[:5],
                    "hybrid_added_count": added_coarse_page_count,
                    "visual_page_nodes_count": mixed_debug_info.get("visual_page_nodes_count", 0),
                },
                "high_confidence_unit": high_confidence_unit,
                "status": "retrieval_complete_prediction_pending"
            }
            
            # Add count VLM outputs if this was a count query
            if count_vlm_outputs:
                hyperedge_details.update(count_vlm_outputs)
            if intent_monitor:
                hyperedge_details["intent_monitoring"] = intent_monitor
            if focused_monitor:
                hyperedge_details["focused_monitoring"] = focused_monitor
            if hybrid_monitor:
                hyperedge_details["hybrid_monitoring"] = hybrid_monitor
            
            # If skip_prediction, return after collecting retrieval results
            if skip_prediction:
                print(f"  ⏭️  Skipping predictions (skip_prediction=True)")
                hyperedge_details["status"] = "retrieval_only"
                result.update({
                    "hyperedge_details": hyperedge_details,
                    "status": "retrieval_only"
                })
                result["monitoring"] = summarize_sample_monitoring({
                    "intent": intent_monitor,
                    "focused": focused_monitor,
                    "hybrid": hybrid_monitor,
                })
                return result
            # --- Update Logs with Prediction ---
            hyperedge_details.update({
                "prediction_variants": {
                    "focused": {
                        "prediction": pred_focused,
                        "full_prompt": hist_focused
                    },
                    "hybrid": {
                        "prediction": pred_hybrid,
                        "full_prompt": hist_hybrid
                    }
                },
                "status": "complete"
            })

            # For count query, predictions are already set in result, just add hyperedge_details
            if is_count_query and query_hyperedges:
                first_res = query_hyperedges[0]
                has_count_result = (isinstance(first_res, dict) and first_res.get('is_count_result')) or\
                                 (hasattr(first_res, 'count_result') and first_res.count_result is not None) or\
                                 (hasattr(first_res, 'raw_dict') and first_res.raw_dict.get('is_count_result'))
            else:
                has_count_result = False
            
            if has_count_result:
                from hyperedge import extract_count_answer

                predicted_answer_hyper = extract_count_answer(pred_hybrid)

                if not predicted_answer_hyper:
                    predicted_answer_hyper = pred_hybrid[:100].strip() if pred_hybrid else "N/A"

                result.update({
                    "predicted_answer": predicted_answer_hyper,
                    "prediction_hybrid": pred_hybrid,
                    "prediction_hyperedge": pred_hybrid,
                    "prediction_focused": pred_focused,   # Add Focused for easy comparison in JSONL
                    "images_count": len(images_hybrid),
                    "hyperedge_details": hyperedge_details
                })
            else:
                # Extract hyperedge predicted answer (using smart extractor)
                from hyperedge import extract_answer, extract_count_answer
                
                # Choose extraction method based on query type
                if is_count_query:
                    predicted_answer_hyper = extract_count_answer(pred_hybrid)
                else:
                    predicted_answer_hyper = extract_answer(pred_hybrid)
                
                # Fallback: if extraction fails, use first 100 characters
                if not predicted_answer_hyper:
                    predicted_answer_hyper = pred_hybrid[:100].strip() if pred_hybrid else "N/A"
                
                # Update Result (Clean)
                result.update({
                    "predicted_answer": predicted_answer_hyper,  # Extracted short answer for easy comparison
                    "prediction_hybrid": pred_hybrid,
                    "prediction_hyperedge": pred_hybrid,  # Default to Hybrid (with full reasoning)
                    "prediction_focused": pred_focused,   # Add Focused for easy comparison in JSONL
                    "prediction": pred_hybrid,            # Override main prediction
                    "images_count": len(images_hybrid),
                    "hyperedge_details": hyperedge_details
                })
                        
        except Exception as e:
            print(f"⚠️ Hyperedge Retrieval Failed: {e}")
            import traceback
            traceback.print_exc()
            result["hyperedge_error"] = str(e)
            # Add hyperedge_details if it was created before the error
            if hyperedge_details is not None:
                result["hyperedge_details"] = hyperedge_details

    result["monitoring"] = summarize_sample_monitoring({
        "intent": intent_monitor,
        "focused": focused_monitor,
        "hybrid": hybrid_monitor,
    })
    result["final_answer_stage"] = "hybrid"
    if hybrid_monitor:
        result["hybrid_monitoring"] = hybrid_monitor

    return result



def compact_prediction_result(result: Dict[str, Any]) -> Dict[str, Any]:
    prediction = result.get("prediction_hyperedge") or result.get("prediction_hybrid") or result.get("prediction")
    compact = {
        "sample_idx": result.get("sample_idx"),
        "q_uid": result.get("q_uid", ""),
        "doc_id": result.get("doc_id", ""),
        "question": result.get("question", ""),
        "ground_truth": result.get("ground_truth"),
        "prediction": prediction,
        "predicted_answer": result.get("predicted_answer"),
        "images_count": result.get("images_count"),
        "status": result.get("status", "success"),
    }
    if result.get("hyperedge_error"):
        compact["error"] = result["hyperedge_error"]
    if result.get("error"):
        compact["error"] = result["error"]
    return compact


def save_result(result: Dict[str, Any], output_file: str):
    """"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')


def run_prediction(
    config_name: str,
    model_type: str,
    model_path: Optional[str] = None,
    experiment_type: str = "vl",
    output_dir: str = "results",
    max_samples: Optional[int] = None,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    setting_name: Optional[str] = None,
    test_file: Optional[str] = None,
    use_hyperedge: bool = False,
    optimizer_bypass: bool = False,
    cca_rerank: bool = True,
    block_prune: bool = True,
    block_prune_cost_aware: bool = False,
    block_prune_density_alpha: float = 1.0,
    enable_reroute: bool = False,
    enable_validation: bool = False,
    skip_prediction: bool = False,
    visual_strategy: str = "mixed",
    intent_aware_convert: bool = True,
    intent_model_type: Optional[str] = None,
    intent_model_path: Optional[str] = None,
    intent_temperature: Optional[float] = None,
    convert_group_mode: str = "entity",
    hypergraph_dir: Optional[str] = None,
    optimizer_k_max: int = 5,
    max_candidates: Optional[int] = None,
    retrieval_top_k: int = 3,
    cost_aware_select: bool = True,
    density_alpha: float = 1.0,
    multi_min_phase2: int = 2,
    full_vlm_cost_aware: bool = True,
    text_cost_unit_words: int = 240,
    text_cost_weight: float = 0.1,
    related_text_cost_discount: float = 0.35,
    text_cost_tiebreak_eps: float = 0.005,
    dynamic_vlm_budget: bool = False,
    vlm_input_budget_units: Optional[float] = None,
    vlm_budget_cost_stat: str = "max",
    optimizer_final_cap: bool = True,
    intent_disabled_types: Optional[List[str]] = None,
    compare_focused_hybrid: bool = False,
    qwen_vl_max_pixels: Optional[int] = None,
    compact_output: bool = True,
    save_diagnostics: bool = False,
    **model_kwargs
):
    """"""
    
    print("🔬 Starting HyperDoc prediction")
    print("=" * 60)
    print(f"Dataset: {config_name}")
    print(f"Model type: {model_type}")
    print(f"Experiment type: {experiment_type}")
    if model_path:
        print(f"Model path: {model_path}")
    if test_file:
        print(f"Test file: {test_file}")
    if use_hyperedge:
        print(f"🚀 Hyperedge retrieval enabled")
        if hypergraph_dir:
            print(f"   📚 Hypergraph directory: {hypergraph_dir}")
        _okm = max(1, int(optimizer_k_max))
        print(f"   Evidence cap: {_okm}")
    if qwen_vl_max_pixels is not None:
        print(f"🖼️  Qwen-VL max_pixels resize: {qwen_vl_max_pixels}")
    if skip_prediction:
        print(f"⏭️  Prediction skipped; retrieval only")
    print("=" * 60)

    if qwen_vl_max_pixels is not None:
        os.environ["QWENVL_MAX_PIXELS"] = str(int(qwen_vl_max_pixels))
    else:
        os.environ.pop("QWENVL_MAX_PIXELS", None)

    #
    dataset = create_dataset(config_name, test_file=test_file)
    dataset_name = get_supported_datasets().get(config_name, "MMLongBench")
    
    #
    reasoner = create_model_reasoner(model_type, model_path, **model_kwargs)
    
    #
    intent_reasoner = None
    if intent_model_type:
        print(f"\n🧠 Creating separate intent reasoner...")
        intent_reasoner = create_model_reasoner(
            intent_model_type, 
            intent_model_path, 
            **model_kwargs
        )
        print(f"✓ Intent reasoner ready: {intent_model_type}")
    
    #
    setting_suffix = f"_{setting_name}" if setting_name else ""
    if use_hyperedge:
        setting_suffix += "_hyperedge"
    
    #
    if start_idx > 0 or (end_idx is not None and end_idx < 100000):
        #
        range_suffix = f"_{start_idx}_{end_idx if end_idx else 'end'}"
        output_file = f"{output_dir}/{config_name}_{model_type}_{experiment_type}{setting_suffix}{range_suffix}.jsonl"
    else:
        #
        range_suffix = ""  #
        output_file = f"{output_dir}/{config_name}_{model_type}_{experiment_type}{setting_suffix}.jsonl"
    
    os.makedirs(output_dir, exist_ok=True)
    
    mode = "append" if os.path.exists(output_file) else "create"
    print(f"📝 Results will be saved to: {output_file} ({mode})")
    print("=" * 60)
    
    #
    try:
        samples = dataset.load_data()
        total_samples = len(samples)
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return
    
    #
    if end_idx is not None:
        end_idx = max(end_idx, start_idx)
        end_idx = min(end_idx, total_samples)
    elif max_samples:
        end_idx = min(start_idx + max_samples, total_samples)
    else:
        end_idx = total_samples
    
    print(f"Sample range: {start_idx} - {end_idx-1} ({total_samples} total)")
    
    #
    success_count = 0
    error_count = 0
    run_monitoring_aggregate = {
        "sample_count": 0,
        "stage_query_time_seconds": 0.0,
        "stage_prompt_tokens": 0,
        "stage_completion_tokens": 0,
        "stage_total_tokens": 0,
        "stage_visual_tokens": 0,
        "by_stage": {},
    }
    
    #
    all_hyperedge_logs = []
    
    #
    def save_hyperedge_logs(logs, output_dir, config_name, model_type, experiment_type, setting_suffix, range_suffix=""):
        """"""
        if logs:
            hyperedge_logs_file = f"{output_dir}/{config_name}_{model_type}_{experiment_type}{setting_suffix}{range_suffix}_hyperedge_logs.json"
            with open(hyperedge_logs_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
            return hyperedge_logs_file
        return None
    
    #
    try:
        for sample_idx in tqdm(range(start_idx, end_idx), desc="🔍 Processing samples", unit="sample"):
            try:
                #
                sample_data = dataset.get_sample_for_vl_experiment(sample_idx)
                
                #
                result = test_single_sample(
                    reasoner,
                    sample_data,
                    sample_idx,
                    experiment_type,
                    setting_name=setting_name,
                    use_hyperedge=use_hyperedge,
                    optimizer_bypass=optimizer_bypass,
                    cca_rerank=cca_rerank,
                    block_prune=block_prune,
                    block_prune_cost_aware=block_prune_cost_aware,
                    block_prune_density_alpha=block_prune_density_alpha,
                    enable_reroute=enable_reroute,
                    output_dir=output_dir,
                    skip_prediction=skip_prediction,
                    device="cuda",
                    intent_reasoner=intent_reasoner,
                    visual_strategy=visual_strategy,
                    intent_aware_convert=intent_aware_convert,
                    intent_temperature=intent_temperature,
                    convert_group_mode=convert_group_mode,
                    hypergraph_dir=hypergraph_dir,
                    dataset_name=dataset_name,
                    optimizer_k_max=optimizer_k_max,
                    max_candidates=max_candidates,
                    retrieval_top_k=retrieval_top_k,
                    cost_aware_select=cost_aware_select,
                    density_alpha=density_alpha,
                    multi_min_phase2=multi_min_phase2,
                    full_vlm_cost_aware=full_vlm_cost_aware,
                    text_cost_unit_words=text_cost_unit_words,
                    text_cost_weight=text_cost_weight,
                    related_text_cost_discount=related_text_cost_discount,
                    text_cost_tiebreak_eps=text_cost_tiebreak_eps,
                    dynamic_vlm_budget=dynamic_vlm_budget,
                    vlm_input_budget_units=vlm_input_budget_units,
                    vlm_budget_cost_stat=vlm_budget_cost_stat,
                    optimizer_final_cap=optimizer_final_cap,
                    intent_disabled_types=intent_disabled_types,
                    compact_output=compact_output,
                    save_diagnostics=save_diagnostics,
                )

                output_result = compact_prediction_result(result) if compact_output else result
                save_result(output_result, output_file)
                update_run_monitoring_aggregate(
                    run_monitoring_aggregate,
                    result.get("monitoring"),
                )
                
                #
                if save_diagnostics and 'hyperedge_details' in result:
                    all_hyperedge_logs.append(result['hyperedge_details'])
                
                #
                if result['status'] == 'success':
                    success_count += 1
                else:
                    error_count += 1
                
                #
                if (sample_idx - start_idx + 1) % 10 == 0:
                    tqdm.write(f"📊 Progress: {sample_idx + 1}/{end_idx}, success: {success_count}, failed: {error_count}")
                    if save_diagnostics and use_hyperedge and all_hyperedge_logs:
                        log_file = save_hyperedge_logs(
                            all_hyperedge_logs, 
                            output_dir, 
                            config_name, 
                            model_type, 
                            experiment_type, 
                            setting_suffix,
                            range_suffix
                        )
                        tqdm.write(f"💾 Saved {len(all_hyperedge_logs)} hyperedge logs")
                    
            except Exception as e:
                print(f"Sample {sample_idx} failed: {e}")
                error_result = {
                    'sample_idx': sample_idx,
                    'status': 'error',
                    'error': str(e),
                    'experiment_type': experiment_type
                }
                save_result(error_result, output_file)
                error_count += 1
    
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected; saving progress...")
        #
        if save_diagnostics and use_hyperedge and all_hyperedge_logs:
            log_file = save_hyperedge_logs(
                all_hyperedge_logs, 
                output_dir, 
                config_name, 
                model_type, 
                experiment_type, 
                setting_suffix,
                range_suffix
            )
            print(f"Saved {len(all_hyperedge_logs)} hyperedge logs to: {log_file}")
        print(f"Processed samples: {success_count + error_count}")
        print(f"✅ success: {success_count}, ❌ failed: {error_count}")
        print(f"Results saved to: {output_file}")
        return  #
    
    #
    processed_samples = end_idx - start_idx
    print("\n" + "=" * 60)
    print("🎉 Prediction complete!")
    print(f"Processed samples: {processed_samples}")
    print(f"✅ success: {success_count}")
    print(f"❌ failed: {error_count}")
    if processed_samples > 0:
        print(f"Success rate: {success_count/processed_samples*100:.2f}%")
    print(f"📝 Result file: {output_file}")
    
    #
    if save_diagnostics and use_hyperedge and all_hyperedge_logs:
        log_file = save_hyperedge_logs(
            all_hyperedge_logs, 
            output_dir, 
            config_name, 
            model_type, 
            experiment_type, 
            setting_suffix,
            range_suffix
        )
        print(f"Hyperedge retrieval logs saved to: {log_file} ({len(all_hyperedge_logs)} entries)")
    
    #
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats = {
        'timestamp': timestamp,
        'config_name': config_name,
        'model_type': model_type,
        'experiment_type': experiment_type,
        'model_path': model_path,
        'processed_samples': processed_samples,
        'success_count': success_count,
        'error_count': error_count,
        'success_rate': success_count/processed_samples*100 if processed_samples > 0 else 0,
        'output_file': output_file,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'setting_name': setting_name,
        'evidence_cap': optimizer_k_max,
        'retrieval_top_k': retrieval_top_k,
        'monitoring_summary': {
            'sample_count': run_monitoring_aggregate['sample_count'],
            'stage_query_time_seconds': round(run_monitoring_aggregate['stage_query_time_seconds'], 6),
            'avg_stage_query_time_seconds': round(
                run_monitoring_aggregate['stage_query_time_seconds'] / run_monitoring_aggregate['sample_count'],
                6,
            ) if run_monitoring_aggregate['sample_count'] > 0 else 0.0,
            'stage_prompt_tokens': run_monitoring_aggregate['stage_prompt_tokens'],
            'stage_completion_tokens': run_monitoring_aggregate['stage_completion_tokens'],
            'stage_total_tokens': run_monitoring_aggregate['stage_total_tokens'],
            'stage_visual_tokens': run_monitoring_aggregate['stage_visual_tokens'],
            'final_answer_stage': 'hybrid',
            'by_stage': {
                stage_name: {
                    'count': stage_stats['count'],
                    'total_query_time_seconds': round(stage_stats['total_query_time_seconds'], 6),
                    'avg_query_time_seconds': round(
                        stage_stats['total_query_time_seconds'] / stage_stats['count'],
                        6,
                    ) if stage_stats['count'] > 0 else 0.0,
                    'prompt_tokens': stage_stats['prompt_tokens'],
                    'completion_tokens': stage_stats['completion_tokens'],
                    'total_tokens': stage_stats['total_tokens'],
                    'visual_tokens': stage_stats['visual_tokens'],
                }
                for stage_name, stage_stats in run_monitoring_aggregate['by_stage'].items()
            },
        },
    }
    
    if save_diagnostics:
        stats_file = f"{output_dir}/{config_name}_{model_type}_{experiment_type}{setting_suffix}{range_suffix}_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"Stats written to: {stats_file}")

        usage_file = f"{output_dir}/{config_name}_{model_type}_{experiment_type}{setting_suffix}{range_suffix}_usage_summary.json"
        with open(usage_file, 'w', encoding='utf-8') as f:
            json.dump(stats['monitoring_summary'], f, indent=2, ensure_ascii=False)
        print(f"Usage summary written to: {usage_file}")


def run_hyperdoc(
    config_name: str,
    model_type: str,
    model_path: Optional[str],
    output_dir: str,
    evidence_cap: int,
    alpha_rho: float,
    max_samples: Optional[int] = None,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    setting_name: Optional[str] = None,
    test_file: Optional[str] = None,
    skip_prediction: bool = False,
    intent_model_type: Optional[str] = None,
    intent_model_path: Optional[str] = None,
    intent_temperature: Optional[float] = 0.3,
    hypergraph_dir: Optional[str] = None,
    qwen_vl_max_pixels: Optional[int] = None,
    **model_kwargs,
):
    """Run the public HyperDoc path with one evidence cap."""
    cap = max(1, int(evidence_cap))
    return run_prediction(
        config_name=config_name,
        model_type=model_type,
        model_path=model_path,
        experiment_type="vl",
        output_dir=output_dir,
        max_samples=max_samples,
        start_idx=start_idx,
        end_idx=end_idx,
        setting_name=setting_name,
        test_file=test_file,
        use_hyperedge=True,
        skip_prediction=skip_prediction,
        intent_model_type=intent_model_type,
        intent_model_path=intent_model_path,
        intent_temperature=intent_temperature,
        hypergraph_dir=hypergraph_dir,
        optimizer_k_max=cap,
        retrieval_top_k=cap,
        density_alpha=float(alpha_rho),
        qwen_vl_max_pixels=qwen_vl_max_pixels,
        compact_output=True,
        save_diagnostics=False,
        **model_kwargs,
    )

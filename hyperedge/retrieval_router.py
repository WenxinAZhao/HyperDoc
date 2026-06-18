#!/usr/bin/env python3
""""""

from typing import List, Dict, Optional, Union

import re

from .graph_loader import load_hypergraph
from .graph_structures import QueryHyperedge, Hypergraph

# Import helpers
from .retrieval_helpers import (
    build_block_index,
    get_visual_contextual_edges,
    filter_edges_by_page_range,
    filter_edges_by_type,
    is_visual_type
)

# Import tools
from .retrieval_tools import (
    tool_find_by_id,
    tool_find_by_page,
    tool_find_by_section,
    tool_keyword_search,
    tool_visual_search,
    tool_multi_keyword_fetch,
    tool_global_count,
    tool_enumerate
)

# Import optimizer
from .hyperedge_optimizer import (
    _is_page_node,
    select_hyperedges,
)

# ============================================================================
# Router Table
# ============================================================================

ROUTER_TABLE = {
    "UNIT": [
        tool_find_by_id,        #
        tool_find_by_page,      #
        tool_find_by_section,   #
        tool_keyword_search     #
    ],
    "MULTI_UNIT": [
        #
        tool_find_by_id,        #
        tool_find_by_page,      #
        tool_find_by_section,   #
        tool_keyword_search     #
    ],
    "DOCUMENT": [
        tool_global_count,      #
        tool_enumerate,         #
        tool_keyword_search,    #
    ],
    None: [tool_keyword_search]  #
}

def _has_explicit_multi_ids(constraints: Dict) -> bool:
    """"""
    multi_id = constraints.get("id")
    if not multi_id or not isinstance(multi_id, str):
        return False

    #
    id_pattern = r'(figure|table|chart|map|graph)\s*(\d+)'
    matches = re.findall(id_pattern, multi_id.lower())
    if len(matches) >= 2:
        return True

    #
    #
    return False


# ============================================================================
# Main Retrieval Function
# ============================================================================

def execute_hyperedge_retrieval(
    query: str,
    doc_id: str,
    intent_state: Dict,
    hypergraph_dir: str = "hypergraph",
    colbert_model = None,
    reasoner = None,
    page_image_dir: str = "",
    candidate_pages: Optional[List[int]] = None,
    visual_strategy: str = "mixed",
    optimizer_bypass: bool = False,
    device: str = "cuda",
    final_rerank: bool = False,
    block_prune: bool = False,
    k_max: int = 5,
    cost_aware_select: bool = False,
    density_alpha: float = 0.90,
    multi_min_phase2: int = 2,
    full_vlm_cost_aware: bool = False,
    text_cost_unit_words: int = 240,
    text_cost_weight: float = 0.1,
    related_text_cost_discount: float = 0.35,
    text_cost_tiebreak_eps: float = 0.005,
    dynamic_vlm_budget: bool = False,
    vlm_input_budget_units: Optional[float] = None,
    vlm_budget_cost_stat: str = "max",
    optimizer_final_cap: bool = False,
    convert_group_mode: str = "block_id",
    intent_disabled_types: Optional[List[str]] = None,
    compare_focused_hybrid: bool = False,
) -> tuple[List[QueryHyperedge], Dict]:
    """"""
    # Load hypergraph
    hypergraph = load_hypergraph(doc_id, hypergraph_dir)
    if not hypergraph: 
        return [], {"error": "Failed to load hypergraph"}
    
    # Extract target and constraints
    target = intent_state.get("target", {})
    evidence_level = target.get("evidence_level")
    constraints = target.get("constraints", {})
    intent_type = intent_state.get("intent_type")
    
    # Note: candidate_pages filtering should be done inside tools, not here
    # to preserve hypergraph integrity (section_headers, etc.)

    # Wrap in Hypergraph object for efficient access
    if isinstance(hypergraph, dict):
        hypergraph = Hypergraph(hypergraph)
    
    # ========================================================================
    #
    # ========================================================================
    selected_tools = []
    constraint_hints = []
    _disabled = set(intent_disabled_types or [])

    #
    if intent_type == "count" and "count" not in _disabled:
        #
        #
        if evidence_level in ("DOCUMENT", None):
            print(f"📊 Global Count Task Detected: count + {evidence_level} level → tool_global_count")
            selected_tools = [tool_global_count]
            constraint_hints.append("intent=count+document→global_count")
        else:
            #
            # Fall through to evidence-level routing below
            print(f"📊 Localized Count Task Detected: count + {evidence_level} level")
            print(f"    Will use normal {evidence_level} routing to locate elements first")
        
    elif intent_type == "enumerate" and "enumerate" not in _disabled:
        if evidence_level in ("DOCUMENT", None):
            print(f"📋 Global Enumerate Task Detected: enumerate + {evidence_level} level")
            selected_tools = [tool_enumerate]
            constraint_hints.append("intent=enumerate+document")
            if constraints.get("keywords"):
                selected_tools.append(tool_keyword_search)
                constraint_hints.append(f"enumerate+kw={constraints['keywords']}")
        else:
            #
            print(f"📋 Localized Enumerate Task Detected: enumerate + {evidence_level} level")
            print(f"    Will use normal {evidence_level} routing to locate elements first")
        
    # 2. Evidence-level routing (UNIT/MULTI_UNIT/DOCUMENT)
    #    Also handles count/enumerate + non-DOCUMENT (which fall through with empty selected_tools)
    if not selected_tools:
        all_admissible_tools = ROUTER_TABLE.get(evidence_level, [tool_keyword_search])
        has_structured_constraint = False
        
        #
        #
        if evidence_level in ("UNIT", "MULTI_UNIT"):
            #
            if constraints.get("id") and tool_find_by_id in all_admissible_tools:
                selected_tools.insert(0, tool_find_by_id)
                constraint_hints.append(f"id={constraints['id']}")
                has_structured_constraint = True
                
            # Priority 2: Page
            if constraints.get("page_range") and tool_find_by_page in all_admissible_tools:
                selected_tools.append(tool_find_by_page)
                constraint_hints.append(f"page_range={constraints['page_range']}")
                has_structured_constraint = True
            
            #
            if constraints.get("section_title") and tool_find_by_section in all_admissible_tools:
                selected_tools.append(tool_find_by_section)
                constraint_hints.append(f"section={constraints['section_title']}")
                has_structured_constraint = True

        #
        #
        if constraints.get("keywords") and tool_keyword_search in all_admissible_tools:
            selected_tools.append(tool_keyword_search)
            constraint_hints.append(f"keywords={constraints['keywords']}")
        elif not has_structured_constraint:
            #
            if tool_keyword_search in all_admissible_tools:
                selected_tools.append(tool_keyword_search)
            constraint_hints.append("no_structured_constraints")

        # locate + DOCUMENT: "which pages have X" → also enumerate to avoid missing pages
        #
        if (intent_type == "locate" and evidence_level == "DOCUMENT"
                and tool_enumerate in all_admissible_tools
                and tool_enumerate not in selected_tools
                and "enumerate" not in _disabled):
            selected_tools.append(tool_enumerate)
            constraint_hints.append("locate+document→add_enumerate")

        #
        if not selected_tools:
            selected_tools = [tool_keyword_search]
    
    # ========================================================================
    #
    # ========================================================================
    all_results = []
    seen_edge_ids = set()
    
    routing_msg = f"Evidence={evidence_level}, Intent={intent_type}, Constraints=[{', '.join(constraint_hints)}] -> Tools={[t.__name__ for t in selected_tools]}"
    print(f"🔍 Routing: {routing_msg}")
    
    logs = {
        "routing_info": routing_msg,
        "evidence_level": evidence_level,
        "intent_type": intent_type,
        "constraints": constraints,
        "tools_selected": [t.__name__ for t in selected_tools],
        "tool_execution_details": []
    }
    
    for tool in selected_tools:
        tool_name = tool.__name__
        print(f"\n▶ Executing {tool_name}...")
        
        #
        tool_kwargs = {
            "hypergraph": hypergraph,
            "constraints": constraints,
            "colbert_model": colbert_model,
            "device": device,
            "original_query": query,
            "reasoner": reasoner,
            "page_image_dir": page_image_dir,
            "clean_doc_id": doc_id,
            "visual_strategy": visual_strategy,
            "candidate_pages": candidate_pages,
        }
        
        #
        #
        #
        if evidence_level == "MULTI_UNIT":
            if tool_name == "tool_find_by_id":
                tool_kwargs["multi_id_mode"] = True
                print(f"  → MULTI_UNIT: Using multi_id_mode for ID matching")
        
        tool_results = tool(**tool_kwargs)
        
        tool_log = {
            "tool_name": tool_name,
            "returned_units": len(tool_results),
            "unique_units_added": 0
        }
        
        unique_added = 0
        for res in tool_results:
            # Handle QueryHyperedge objects
            is_count = False
            if hasattr(res, 'count_result') and res.count_result is not None:
                is_count = True
            elif hasattr(res, 'raw_dict') and res.raw_dict.get('is_count_result'):
                is_count = True
                
            if is_count:
                all_results.append(res)
            else:
                # Extract edge_id safely from raw_dict
                edge_id = None
                if hasattr(res, 'raw_dict') and 'query_hyperedge' in res.raw_dict and res.raw_dict['query_hyperedge']:
                    edge_id = res.raw_dict['query_hyperedge'].get('edge_id')
                
                if edge_id: 
                    if edge_id not in seen_edge_ids:
                        all_results.append(res)
                        seen_edge_ids.add(edge_id)
                        unique_added += 1
                else:
                    # If no edge_id (unlikely for proper edge), append anyway
                    all_results.append(res)
                    unique_added += 1
        
        #
        tool_log["unique_units_added"] = unique_added
        logs["tool_execution_details"].append(tool_log)
        print(f"  ✓ {tool_name}: returned {len(tool_results)}, added {unique_added} unique units")
    
    # Supplement tool_visual_search with coarse page candidates.
    # Triggered when EITHER condition is met:
    #   1. No explicit page constraint & not document-level:
    #      OCR tools work at block level and lack page-level visual signal;
    #      coarse page nodes provide that complement.
    #   2. OCR-based tools returned nothing (empty hyperedge fallback):
    #      visual page nodes are the last resort before keyword fallback.
    has_page_constraint = bool(constraints.get("page_range"))
    is_document_level   = (evidence_level == "DOCUMENT")
    # Results flagged fallback_to_coarse_pages=True are log-only sentinels. They
    # must not suppress the visual_search fallback.
    def _is_fallback_sentinel(r):
        return hasattr(r, 'raw_dict') and r.raw_dict.get('fallback_to_coarse_pages')
    ocr_has_results = any(not _is_fallback_sentinel(r) for r in all_results)

    vs_cond_supplement = candidate_pages and not has_page_constraint and not is_document_level
    vs_cond_fallback   = candidate_pages and not ocr_has_results

    if vs_cond_supplement or vs_cond_fallback:
        trigger_reasons = []
        if vs_cond_supplement:
            trigger_reasons.append("no page constraint")
        if vs_cond_fallback:
            trigger_reasons.append("ocr empty")
        print(f"\n▶ Executing tool_visual_search ({', '.join(trigger_reasons)}, {len(candidate_pages)} pages)...")
        vs_results = tool_visual_search(
            hypergraph=hypergraph,
            constraints=constraints,
            candidate_pages=candidate_pages,
        )
        vs_added = 0
        for res in vs_results:
            # Page nodes have no edge_id — deduplicate by source_page
            # source_page is at raw_dict top level (extra_meta fields are spread via **extra_meta)
            src_page = res.raw_dict.get('source_page')\
                if hasattr(res, 'raw_dict') else None
            page_key = f"__page_{src_page}__" if src_page is not None else None
            if page_key and page_key in seen_edge_ids:
                continue
            all_results.append(res)
            if page_key:
                seen_edge_ids.add(page_key)
            vs_added += 1
        logs["tool_execution_details"].append({
            "tool_name": "tool_visual_search",
            "returned_units": len(vs_results),
            "unique_units_added": vs_added,
            "trigger": trigger_reasons,
        })
        print(f"  ✓ tool_visual_search: added {vs_added} page nodes")

    # ── Fallback: if still no results, try keyword search ───────────────────
    # Note: visual_search fallback is already handled above (vs_cond_fallback).
    if not all_results:
        print("⚠️ No results from any retrieval (incl. visual), triggering keyword fallback...")
        logs["fallback_triggered"] = True

        fallback_results = tool_keyword_search(
            hypergraph=hypergraph,
            constraints={"keywords": query.split()},
            original_query=query
        )
        fallback_method = "keyword_search"

        fallback_added = 0
        for res in fallback_results:
            edge_id = None
            if hasattr(res, 'raw_dict') and res.raw_dict.get('query_hyperedge'):
                edge_id = res.raw_dict['query_hyperedge'].get('edge_id')
            # source_page is at raw_dict top level (extra_meta spread)
            src_page = res.raw_dict.get('source_page')\
                if hasattr(res, 'raw_dict') else None
            dedup_key = edge_id or (f"__page_{src_page}__" if src_page is not None else None)

            if dedup_key:
                if dedup_key not in seen_edge_ids:
                    all_results.append(res)
                    seen_edge_ids.add(dedup_key)
                    fallback_added += 1
            else:
                all_results.append(res)
                fallback_added += 1

        logs["tool_execution_details"].append({
            "tool_name": f"fallback_{fallback_method}",
            "returned_units": len(fallback_results),
            "unique_units_added": fallback_added,
        })
        print(f"  ✓ Fallback ({fallback_method}): added {fallback_added} units")

    # Cost-aware optimizer.
    # Hard-tool and aggregation results are initialized first; keyword evidence
    # is then added by greedy soft augmentation.
    # Page nodes are kept as-is (not optimized — they lack kw_fscores).
    page_nodes      = [r for r in all_results if _is_page_node(r)]
    # Exclude fallback sentinels from optimization; they are
    # log-only markers and carry no real evidence (score=0, count_result=None).
    regular_results = [r for r in all_results
                       if not _is_page_node(r) and not _is_fallback_sentinel(r)]

    if regular_results:
        kw = constraints.get('keywords') or []
        ev = evidence_level or "UNIT"

        # ── Slim candidate snapshot ──────────────────────────────────────────
        # Saved BEFORE optimization so future optimizer changes can be replayed
        # offline from logs without re-running retrieval tools.
        # Fields kept: only what select_hyperedges needs to make decisions.
        logs["optimizer_candidates"] = [
            {
                "tool": h.tool,
                "score": round(h.score, 6),
                "kw_fscores": h.raw_dict.get("kw_fscores") or {},
                "is_count_result":    bool(h.raw_dict.get("is_count_result")),
                "is_enumerate_result": bool(h.raw_dict.get("is_enumerate_result")),
                "anchor_page": (h.raw_dict.get("anchor_block") or {}).get("page"),
                "source_page": h.raw_dict.get("source_page"),
            }
            for h in regular_results
        ]
        # Page nodes are NOT optimized; save their pages for full-picture analysis
        logs["optimizer_page_node_pages"] = sorted({
            h.raw_dict.get("source_page")
            for h in page_nodes
            if h.raw_dict.get("source_page") is not None
        })

        # Detect questions that ask for exact title/table-name strings.
        # For these, title-type blocks must not be pruned so VLM sees the
        # verbatim heading text it needs to select from.
        _q_lower = query.lower()
        _preserve_titles = (
            "select titles from the doc" in _q_lower
            or "select table names from the doc" in _q_lower
        )

        if optimizer_bypass:
            print(f"\n⚙️  Optimizer bypass ({ev}/{intent_type}) "
                  f"({len(regular_results)} regular + {len(page_nodes)} page nodes)")
            logs["optimization"] = {
                "mode": "bypass",
                "evidence_level": ev,
                "intent_type": intent_type,
                "phase1_count": 0,
                "phase2_count": 0,
                "selected_total": len(regular_results),
                "optimizer_bypass": True,
            }
        else:
            print(f"\n⚙️  Optimizer ({ev}/{intent_type}) "
                  f"({len(regular_results)} regular + {len(page_nodes)} page nodes), k_max={k_max}...")
            if cost_aware_select:
                print(
                    f"    cost-aware density stop enabled: "
                    f"alpha={density_alpha}, multi_min_phase2={multi_min_phase2}"
                )
                if full_vlm_cost_aware:
                    print(
                        f"    full VLM cost proxy enabled: "
                        f"text_unit_words={text_cost_unit_words}, text_weight={text_cost_weight}, "
                        f"related_discount={related_text_cost_discount}, "
                        f"tie_eps={text_cost_tiebreak_eps}"
                    )
                if dynamic_vlm_budget:
                    print(
                        f"    dynamic VLM budget enabled: "
                        f"budget_units={vlm_input_budget_units}, cost_stat={vlm_budget_cost_stat}"
                    )
            regular_results, opt_log = select_hyperedges(
                candidates=regular_results,
                constraints=constraints,
                keywords=kw,
                evidence_level=ev,
                intent_type=intent_type or "locate",
                min_soft_gain=0.01,
                k_max=k_max,
                maximize_kw_threshold=0.05,
                final_rerank=final_rerank,
                block_prune=block_prune,
                preserve_title_blocks=_preserve_titles,
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
                final_cap_group_mode=convert_group_mode,
            )
            logs["optimization"] = opt_log
    else:
        logs["optimization"] = {"phase1_count": 0, "phase2_count": 0, "selected_total": 0}

    # Reassemble: optimizer output + page_nodes.
    # When the optimizer ran, regular_results is already in greedy order
    # (hard-match head, greedy-selected soft body).
    # Re-sorting by IDF score would destroy that ordering.
    # Page nodes are sorted by score and appended after.
    optimizer_bypassed = bool(logs.get("optimization", {}).get("optimizer_bypass"))
    optimizer_ran = bool(logs.get("optimization", {}).get("selected_total", 0))
    page_nodes.sort(key=lambda x: x.score, reverse=True)
    if optimizer_bypassed or not optimizer_ran:
        regular_results.sort(key=lambda x: x.score, reverse=True)
    all_results = regular_results + page_nodes

    return all_results, logs


# ============================================================================
# Export
# ============================================================================

__all__ = [
    "execute_hyperedge_retrieval",
    "ROUTER_TABLE",
]

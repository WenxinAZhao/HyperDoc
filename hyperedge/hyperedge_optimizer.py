#!/usr/bin/env python3
"""
Cost-aware hyperedge selection for HyperDoc.

This module implements the evidence selection procedure used by HyperDoc.
It first keeps hard constraint matches and aggregation results, then applies
greedy soft augmentation with an adaptive density-based stopping rule.

The public entry point is `select_hyperedges`. It accepts candidate evidence
from the retrieval router, intent constraints, query keywords, and the current
evidence cap, and returns the selected evidence list plus diagnostic metadata.
"""

from typing import List, Dict, Optional, Tuple

from .retrieval_helpers import (
    build_block_index,
    fuzzy_keyword_match,
    fuzzy_keyword_match_score,
)


# ============================================================================
# Tool classification for hard-coverage initialization
# ============================================================================

# Hard-constraint tools: user explicitly requested evidence via structured query.
HARD_TOOLS = frozenset({
    'tool_find_by_id',
    'tool_find_by_page',
    'tool_find_by_section',
})

# Aggregation tools produce combined evidence for count/enumeration queries.
AGGREGATION_TOOLS = frozenset({
    'tool_global_count',
    'tool_enumerate',
})

# Union: tools collected during hard-coverage initialization.
PHASE1_TOOLS = HARD_TOOLS | AGGREGATION_TOOLS  # alias kept for backward compat
GUARANTEED_TOOLS = PHASE1_TOOLS


# ============================================================================
# Intent-specific evidence reranking
# ============================================================================

def intent_rerank(
    phase2: List,
    evidence_level: str,
) -> List:
    """Rerank soft-augmentation hyperedges based on intent structure.

    Logic
    -----
    MULTI_UNIT (compare / cross_reasoning):
        Group soft candidates by anchor_block.block_id (entity/constraint group).
        Within each group, sort by score descending.
        Preserve the original inter-group arrival order (the first HE of each
        group determines the group's rank).
        Result: intra-entity blocks clustered and quality-sorted, while
        distinct entities appear in their natural retrieval order.

    UNIT / DOCUMENT / null:
        Pass through unchanged.

    Parameters
    ----------
    phase2 : list of QueryHyperedge
        Soft-tool HEs selected during greedy augmentation.
    evidence_level : str
        "MULTI_UNIT" | "UNIT" | "DOCUMENT" | "null"

    Returns
    -------
    list of QueryHyperedge — possibly reordered copy of phase2.
    """
    if evidence_level != "MULTI_UNIT" or not phase2:
        return phase2

    from collections import OrderedDict
    groups: "OrderedDict[object, List]" = OrderedDict()
    for he in phase2:
        # Anchor key: anchor_block.block_id (QueryHyperedge object)
        anchor = getattr(he, 'anchor_block', None)
        ak = getattr(anchor, 'block_id', None) if anchor is not None else None
        if ak is None:
            ak = id(he)            # fallback: treat as its own group
        if ak not in groups:
            groups[ak] = []
        groups[ak].append(he)

    # Sort within each group by score descending
    for ak in groups:
        groups[ak].sort(key=lambda h: getattr(h, 'score', 0.0), reverse=True)

    return [he for group_hes in groups.values() for he in group_hes]


def _is_successful_aggregation_candidate(h) -> bool:
    raw = h.raw_dict if hasattr(h, 'raw_dict') else (h if isinstance(h, dict) else {})
    tool = getattr(h, 'tool', raw.get('tool'))
    if tool not in AGGREGATION_TOOLS:
        return False
    if raw.get('is_count_result') is False and raw.get('count_result') is None:
        return False
    if raw.get('is_count_result') or raw.get('count_result') is not None:
        return True
    if raw.get('is_enumerate_result') or raw.get('enumerated_items') is not None:
        return True
    return False

def _final_cap_anchor_key(he, group_mode: str = "block_id") -> Optional[str]:
    """Return the grouping key used by the convert-stage MULTI_UNIT cap."""
    rd = getattr(he, "raw_dict", None) or {}
    anchor = rd.get("anchor_block")

    def _anchor_page() -> Optional[str]:
        if anchor is None:
            return None
        page = anchor.get("page") if isinstance(anchor, dict) else getattr(anchor, "page", None)
        return f"page_{page}" if page is not None else None

    if group_mode == "entity":
        import re as _re

        match_reason = rd.get("match_reason", "") or ""
        match = _re.search(r"Matched ID\s+(\S+)\s+in", match_reason)
        if match:
            return f"entity_{match.group(1)}"
        return _anchor_page()

    if group_mode == "page":
        return _anchor_page()

    if anchor is None:
        return None
    block_id = anchor.get("block_id") if isinstance(anchor, dict) else getattr(anchor, "block_id", None)
    if block_id is not None:
        return f"bid_{block_id}"
    return _anchor_page()


def _apply_final_evidence_cap(
    selected: List,
    cap: int,
    evidence_level: str,
    intent_type: str,
    group_mode: str = "block_id",
) -> Tuple[List, Dict]:
    """Apply the final evidence cap before VLM input construction.

    Ordinary queries keep the selected-order prefix. Intent-aware MULTI_UNIT
    queries use group round-robin to preserve evidence from each entity group.
    """
    limit = max(1, int(cap))
    before = len(selected)
    if before <= limit:
        return selected, {
            "applied": False,
            "limit": limit,
            "before": before,
            "after": before,
            "dropped": 0,
            "mode": "none",
        }

    if evidence_level == "MULTI_UNIT" and intent_type not in ("count", "enumerate"):
        groups: Dict[Optional[str], List] = {}
        group_order: List[Optional[str]] = []
        for he in selected:
            key = _final_cap_anchor_key(he, group_mode)
            if key not in groups:
                groups[key] = []
                group_order.append(key)
            groups[key].append(he)

        capped: List = []
        used = {key: 0 for key in group_order}
        while len(capped) < limit:
            added = False
            for key in group_order:
                if len(capped) >= limit:
                    break
                idx = used[key]
                if idx < len(groups[key]):
                    capped.append(groups[key][idx])
                    used[key] += 1
                    added = True
            if not added:
                break
        mode = f"multi_unit_round_robin:{group_mode}"
    else:
        capped = selected[:limit]
        mode = "selected_order_prefix"

    return capped, {
        "applied": True,
        "limit": limit,
        "before": before,
        "after": len(capped),
        "dropped": before - len(capped),
        "mode": mode,
    }


# ============================================================================
# Block-level evidence pruning
# ============================================================================

def _ef_prune_blocks(he, keywords: List[str], alpha: float = 0.5,
                     preserve_title_blocks: bool = False) -> bool:
    """Prune related_blocks of a QueryHyperedge in place.

    Rules:
      - Hard-tool HEs: fully exempt; related_blocks untouched.
      - Aggregation results (count/enumerate): fully exempt.
      - Soft-tool HEs: greedy marginal coverage selection on related_blocks
        using query keywords.
      - If preserve_title_blocks=True: title-type blocks are always kept
        regardless of keyword match (needed for topic2title/summary2tab).

    Modifies he.related_blocks (Block objects) and he.raw_dict['related_blocks']
    (raw dicts) in-place.  Returns True if any blocks were removed.
    """
    # Exempt hard-tool and aggregation HEs; their blocks carry structural context.
    if he.tool in PHASE1_TOOLS:
        return False
    if (he.count_result is not None
            or he.raw_dict.get('is_count_result')
            or he.raw_dict.get('is_enumerate_result')):
        return False

    related = he.related_blocks  # List[Block]
    if not related or not keywords:
        return False

    # When preserve_title_blocks is set, title-type blocks are always kept.
    # Title blocks carry exact heading/caption strings for topic2title /
    # summary2tab questions and are unlikely to fuzzy-match abstract keywords.
    title_indices: set = set()
    if preserve_title_blocks:
        title_indices = {i for i, b in enumerate(related) if getattr(b, 'type', '') == 'title'}

    min_gain = 0.01

    # Initialise per-kw best coverage from anchor + caption seed
    best_kw: Dict[str, float] = {kw: 0.0 for kw in keywords}
    for b in he.anchor_blocks + he.caption_blocks:
        text = (b.text or '').lower()
        if not text:
            continue
        for kw in keywords:
            s = float(fuzzy_keyword_match_score(kw, text, threshold=0.80))
            if s > best_kw[kw]:
                best_kw[kw] = s

    def _marginal(text_lower: str) -> float:
        if not text_lower:
            return 0.0
        total = 0.0
        for kw in keywords:
            s = float(fuzzy_keyword_match_score(kw, text_lower, threshold=0.80))
            total += max(0.0, s - best_kw[kw])
        return total / len(keywords)

    # Greedy best-first selection.
    remaining = list(range(len(related)))
    selected_indices: List[int] = []
    top1_delta = 0.0

    while remaining:
        best_d, best_i = -1.0, -1
        for i in remaining:
            text_lower = (related[i].text or '').lower()
            d = _marginal(text_lower)
            if d > best_d + 1e-9:
                best_d, best_i = d, i

        if best_i < 0:
            break

        if not top1_delta:
            top1_delta = best_d

        if best_d < max(min_gain, alpha * top1_delta):
            break

        selected_indices.append(best_i)
        # Update coverage from selected block
        text = (related[best_i].text or '').lower()
        for kw in keywords:
            s = float(fuzzy_keyword_match_score(kw, text, threshold=0.80))
            if s > best_kw[kw]:
                best_kw[kw] = s
        remaining.remove(best_i)

    # keep = greedy-selected indices ∪ forced title indices
    keep = set(selected_indices) | title_indices

    if len(keep) == len(related):
        return False  # nothing pruned

    # Preserve original order
    he.related_blocks = [b for i, b in enumerate(related) if i in keep]

    # Sync raw_dict for logging / downstream code that reads raw_dict
    raw_related = he.raw_dict.get('related_blocks') or []
    if raw_related:
        he.raw_dict['related_blocks'] = [b for i, b in enumerate(raw_related) if i in keep]

    return True

def select_hyperedges(
    candidates: List,
    constraints: Dict,
    keywords: List[str],
    evidence_level: str = "UNIT",
    intent_type: str = "locate",
    min_soft_gain: float = 0.01,
    k_max: int = 5,
    maximize_kw_threshold: float = 0.05,
    relative_gain_alpha: float = 0.80,
    final_rerank: bool = False,
    block_prune: bool = False,
    block_prune_alpha: float = 0.50,
    preserve_title_blocks: bool = False,
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
    final_cap_group_mode: str = "block_id",
) -> Tuple[List, Dict]:
    """Select HyperDoc evidence with hard initialization and soft augmentation.

    This is the ONLY optimizer entry point.  All query types flow through
    the same hard-initialization and soft-augmentation pipeline.

    Internal naming note:
      phase1 = hard-coverage initialization.
      phase2 = density-based soft augmentation.
      phase3 = final reranking and block pruning.

    Parameters
    ----------
    candidates : list of QueryHyperedge
        Pool returned by retrieval tools (after dedup).
    constraints : dict
        Structured constraints from intent_state (id, page_range,
        section_title, keywords, etc.).
    keywords : list of str
        Keywords extracted from constraints.
    evidence_level : str
        "UNIT" | "MULTI_UNIT" | "DOCUMENT" | "null"
    intent_type : str
        "locate" | "count" | "enumerate" | "compare" | "cross_reasoning"
    min_soft_gain : float
        Absolute floor: soft augmentation stops when δ_soft falls below this value.
        Applies before the relative threshold is calibrated (i.e. on the
        very first candidate and as a hard lower bound thereafter).
    k_max : int
        Soft-augmentation budget in select mode. Hard matches are initialized
        before soft augmentation and the public pipeline applies the final
        evidence cap before VLM input construction.
    maximize_kw_threshold : float
        In maximize mode, include HEs with mean_f >= this threshold.
    relative_gain_alpha : float
        Relative stopping threshold as a fraction of the first-selected
        HE's δ_soft.  After picking h₁ with δ₁, subsequent HEs are kept
        only when δ > max(min_soft_gain, α·δ₁).
        Set to None (or 0.0) to disable and use only min_soft_gain.
    block_prune : bool
        Apply block-level pruning to soft-tool HEs. Hard-tool and aggregation
        HEs are exempt.
    final_rerank : bool
        Apply intent_rerank() to soft-tool HEs before block pruning.
        For MULTI_UNIT: groups soft HEs
        by anchor entity and sorts within each group by score descending,
        preserving inter-group order.  For UNIT/DOCUMENT: no-op.
    block_prune_alpha : float
        Relative threshold for block-level pruning.

    Returns
    -------
    (selected, opt_log) : tuple
    """
    if not candidates:
        return [], {"phase1_count": 0, "phase2_count": 0, "selected_total": 0}

    # ── Mode determination ────────────────────────────────────────────────
    # maximize mode: aggregation intent on DOCUMENT scope needs ALL matches
    maximize = (
        intent_type in ("count", "enumerate")
        and evidence_level == "DOCUMENT"
    )
    is_multi = (evidence_level == "MULTI_UNIT")

    cands = sorted(candidates, key=lambda h: -h.score)

    # ── Running soft-coverage state ───────────────────────────────────────
    best_combined = [0.0]                           # max mean_f seen so far
    best_fs       = {kw: 0.0 for kw in keywords}   # per-kw best (for final_soft log)

    def _mean_f(h) -> float:
        if not keywords:
            return 0.0
        kw_f = h.raw_dict.get('kw_fscores', {})
        return sum(kw_f.get(kw, 0.0) for kw in keywords) / len(keywords)

    def _update_soft(h):
        kw_f = h.raw_dict.get('kw_fscores', {})
        best_combined[0] = max(best_combined[0], _mean_f(h))
        for kw in keywords:
            best_fs[kw] = max(best_fs[kw], kw_f.get(kw, 0.0))

    def _delta_soft(h) -> float:
        """Soft-constraint marginal gain for the keyword portion.

        MULTI_UNIT queries use per-keyword marginal gains so distinct evidence
        groups can remain eligible. UNIT queries use the mean keyword score as
        a compact single-evidence relevance signal.
        """
        if not keywords:
            return 0.0
        if is_multi:
            kw_f = h.raw_dict.get('kw_fscores', {})
            return sum(max(0.0, kw_f.get(kw, 0.0) - best_fs[kw])
                       for kw in keywords) / len(keywords)
        return _mean_f(h)

    def _block_page(block) -> Optional[int]:
        if block is None:
            return None
        if isinstance(block, dict):
            page = block.get('page')
        else:
            page = getattr(block, 'page', None)
        if page is None:
            return None
        try:
            return int(page)
        except Exception:
            return None

    def _pages_of(h) -> set:
        """Pages that would become full-page VLM images for a hyperedge."""
        pages = set()
        for b in getattr(h, 'anchor_blocks', []) or []:
            p = _block_page(b)
            if p is not None:
                pages.add(p)
        for b in getattr(h, 'caption_blocks', []) or []:
            p = _block_page(b)
            if p is not None:
                pages.add(p)
        for b in getattr(h, 'related_blocks', []) or []:
            p = _block_page(b)
            if p is not None:
                pages.add(p)
        for p_obj in getattr(h, 'source_pages', []) or []:
            p = getattr(p_obj, 'page_num', None)
            if p is not None:
                try:
                    pages.add(int(p))
                except Exception:
                    pass

        rd = getattr(h, 'raw_dict', {}) or {}
        for key in ('anchor_block', 'caption_block'):
            p = _block_page(rd.get(key))
            if p is not None:
                pages.add(p)
        for b in rd.get('related_blocks') or []:
            p = _block_page(b)
            if p is not None:
                pages.add(p)
        sp = rd.get('source_page')
        if sp is not None:
            try:
                pages.add(int(sp))
            except Exception:
                pass
        return pages

    def _has_crop(h) -> bool:
        anchors = getattr(h, 'anchor_blocks', []) or []
        for b in anchors:
            if _block_page(b) is not None and getattr(b, 'bbox', None):
                return True
        rd = getattr(h, 'raw_dict', {}) or {}
        ab = rd.get('anchor_block')
        return bool(isinstance(ab, dict) and ab.get('page') is not None and ab.get('bbox'))

    def _delta_image_cost(h, selected_pages: set) -> Tuple[int, set]:
        pages = _pages_of(h)
        new_pages = pages - selected_pages
        cost = (1 if _has_crop(h) else 0) + len(new_pages)
        return max(1, cost), new_pages

    def _word_count(text: str) -> int:
        if not text:
            return 0
        return len(str(text).split())

    def _block_identity(block_like, fallback_prefix: str) -> Optional[str]:
        if block_like is None:
            return None
        if isinstance(block_like, dict):
            block_id = block_like.get('block_id')
            page = block_like.get('page')
            block_type = block_like.get('type')
            text = block_like.get('text')
        else:
            block_id = getattr(block_like, 'block_id', None)
            page = getattr(block_like, 'page', None)
            block_type = getattr(block_like, 'type', None)
            text = getattr(block_like, 'text', None)
        if block_id:
            return str(block_id)
        text_sig = str(text or '').strip()[:80]
        return f"{fallback_prefix}|{page}|{block_type}|{text_sig}" if text_sig else None

    def _collect_vlm_text_blocks(h) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        """Return (caption_blocks, related_blocks) as (block_identity, word_count).

        Mirrors convert_units_to_vlm_input(): prompt text comes from caption and
        related blocks only. Anchor text is intentionally excluded.
        """
        caption_entries: List[Tuple[str, int]] = []
        related_entries: List[Tuple[str, int]] = []

        def _append_entries(blocks, target: List[Tuple[str, int]], prefix: str) -> None:
            seen_local = set()
            for block in blocks or []:
                if isinstance(block, dict):
                    text = block.get('text')
                else:
                    text = getattr(block, 'text', None)
                words = _word_count(text)
                if words <= 0:
                    continue
                block_key = _block_identity(block, prefix)
                if block_key is None or block_key in seen_local:
                    continue
                seen_local.add(block_key)
                target.append((block_key, words))

        caption_blocks = getattr(h, 'caption_blocks', None)
        related_blocks = getattr(h, 'related_blocks', None)
        if caption_blocks is not None or related_blocks is not None:
            _append_entries(caption_blocks, caption_entries, "caption")
            _append_entries(related_blocks, related_entries, "related")
        else:
            rd = getattr(h, 'raw_dict', {}) or {}
            _append_entries([rd.get('caption_block')] if rd.get('caption_block') else [], caption_entries, "caption")
            _append_entries(rd.get('related_blocks') or [], related_entries, "related")

        return caption_entries, related_entries

    def _delta_vlm_cost(
        h,
        selected_pages: set,
        selected_text_block_ids: set,
    ) -> Tuple[float, set, int, float, int, set]:
        image_cost, new_pages = _delta_image_cost(h, selected_pages)
        text_words = 0
        text_cost = 0.0
        new_text_ids: set = set()
        if full_vlm_cost_aware:
            caption_entries, related_entries = _collect_vlm_text_blocks(h)
            related_discount = max(0.0, float(related_text_cost_discount))
            text_units = 0.0
            for block_id, words in caption_entries:
                if block_id in selected_text_block_ids:
                    continue
                new_text_ids.add(block_id)
                text_words += words
                text_units += float(words)
            for block_id, words in related_entries:
                if block_id in selected_text_block_ids:
                    continue
                new_text_ids.add(block_id)
                text_words += words
                text_units += related_discount * float(words)
            unit_words = max(1, int(text_cost_unit_words))
            text_cost = float(text_cost_weight) * (text_units / float(unit_words))
        return max(1.0, float(image_cost) + text_cost), new_pages, image_cost, text_cost, text_words, new_text_ids

    def _prefer_with_text_tiebreak(
        density: float,
        best_density: float,
        text_cost: float,
        best_text_cost: float,
        score: float,
        best_score: float,
    ) -> bool:
        if density > best_density + 1e-9:
            return True
        if abs(density - best_density) <= max(0.0, float(text_cost_tiebreak_eps)):
            if score > best_score + 1e-9:
                return True
            if abs(score - best_score) <= 1e-9 and text_cost < best_text_cost - 1e-9:
                return True
        return False

    def _derive_effective_k_max(pool: List) -> Tuple[int, Dict]:
        legacy_k = max(1, int(k_max))
        meta = {
            "dynamic_vlm_budget": bool(dynamic_vlm_budget),
            "legacy_k_max": legacy_k,
            "effective_k_max": legacy_k,
        }
        if not dynamic_vlm_budget:
            meta["reason"] = "disabled"
            return legacy_k, meta

        try:
            budget_units = float(vlm_input_budget_units)
        except (TypeError, ValueError):
            meta["reason"] = "missing_vlm_input_budget_units"
            return legacy_k, meta
        if budget_units <= 0:
            meta["reason"] = "non_positive_vlm_input_budget_units"
            meta["vlm_input_budget_units"] = budget_units
            return legacy_k, meta

        initial_costs: List[float] = []
        for h in pool:
            cost, _, _, _, _, _ = _delta_vlm_cost(h, set(), set())
            initial_costs.append(max(1.0, float(cost)))

        if not initial_costs:
            meta["reason"] = "empty_candidate_pool"
            meta["vlm_input_budget_units"] = budget_units
            return legacy_k, meta

        stat = str(vlm_budget_cost_stat or "max").lower()
        if stat in ("avg", "average", "mean"):
            cost_proxy = sum(initial_costs) / float(len(initial_costs))
            stat = "avg"
        else:
            cost_proxy = max(initial_costs)
            stat = "max"

        effective_k = max(1, int(budget_units // max(1.0, cost_proxy)))
        meta.update({
            "reason": "derived_from_vlm_budget",
            "vlm_input_budget_units": round(budget_units, 4),
            "vlm_budget_cost_stat": stat,
            "candidate_cost_count": len(initial_costs),
            "candidate_cost_min": round(min(initial_costs), 4),
            "candidate_cost_avg": round(sum(initial_costs) / float(len(initial_costs)), 4),
            "candidate_cost_max": round(max(initial_costs), 4),
            "cost_proxy": round(cost_proxy, 4),
            "effective_k_max": effective_k,
        })
        return effective_k, meta

    def _phase1_text(h) -> str:
        texts = []
        for blocks in (
            getattr(h, 'anchor_blocks', []) or [],
            getattr(h, 'caption_blocks', []) or [],
            getattr(h, 'related_blocks', []) or [],
        ):
            for block in blocks:
                text = getattr(block, 'text', None)
                if text:
                    texts.append(str(text))

        rd = getattr(h, 'raw_dict', {}) or {}
        for key in ('anchor_block', 'caption_block', 'title_block'):
            block = rd.get(key)
            if isinstance(block, dict) and block.get('text'):
                texts.append(str(block['text']))
        for block in rd.get('related_blocks') or []:
            if isinstance(block, dict) and block.get('text'):
                texts.append(str(block['text']))

        return "\n".join(texts).lower()

    def _phase1_compact(raw_phase1: List) -> Tuple[List, Dict]:
        """Compact overflowing hard-hit sets before VLM packing.

        This applies the VLM packing cap before the final evidence handoff.
        Aggregation outputs are always preserved; the remaining
        hard hits are selected greedily for keyword affinity and page diversity.
        """
        phase1_budget = max(1, int(effective_k_max))
        if len(raw_phase1) <= phase1_budget:
            return raw_phase1, {
                "phase1_budget": phase1_budget,
                "phase1_raw_count": len(raw_phase1),
                "phase1_compacted_count": len(raw_phase1),
                "phase1_compaction_applied": False,
                "phase1_compaction_dropped": 0,
            }

        locked = [h for h in raw_phase1 if h.tool in AGGREGATION_TOOLS]
        locked_ids = {id(h) for h in locked}
        if len(locked) >= phase1_budget:
            kept = locked[:phase1_budget]
            return kept, {
                "phase1_budget": phase1_budget,
                "phase1_raw_count": len(raw_phase1),
                "phase1_compacted_count": len(kept),
                "phase1_compaction_applied": True,
                "phase1_compaction_dropped": len(raw_phase1) - len(kept),
            }

        selected = list(locked)
        selected_pages = set()
        selected_text_block_ids = set()
        for h in selected:
            selected_pages.update(_pages_of(h))
            if full_vlm_cost_aware:
                caption_entries, related_entries = _collect_vlm_text_blocks(h)
                selected_text_block_ids.update(block_id for block_id, _ in caption_entries)
                selected_text_block_ids.update(block_id for block_id, _ in related_entries)

        best_kw: Dict[str, float] = {kw: 0.0 for kw in keywords}
        for h in selected:
            text_lower = _phase1_text(h)
            if not text_lower:
                continue
            for kw in keywords:
                s = float(fuzzy_keyword_match_score(kw, text_lower, threshold=0.80))
                if s > best_kw[kw]:
                    best_kw[kw] = s

        remaining = [h for h in raw_phase1 if id(h) not in locked_ids]
        while remaining and len(selected) < phase1_budget:
            best_idx = -1
            best_density = -1.0
            best_score = -1.0
            best_text_cost = float("inf")
            best_new_pages = set()
            best_kw_updates: Dict[str, float] = {}
            best_new_text_ids = set()
            for idx, h in enumerate(remaining):
                text_lower = _phase1_text(h)
                kw_gain = 0.0
                kw_updates: Dict[str, float] = {}
                if keywords and text_lower:
                    for kw in keywords:
                        s = float(fuzzy_keyword_match_score(kw, text_lower, threshold=0.80))
                        kw_updates[kw] = s
                        kw_gain += max(0.0, s - best_kw[kw])

                cost, new_pages, image_cost, text_cost, text_words, new_text_ids = _delta_vlm_cost(
                    h, selected_pages, selected_text_block_ids
                )
                page_gain = float(len(new_pages))
                tool_bonus = 0.3 if h.tool == 'tool_find_by_id' else 0.15 if h.tool == 'tool_find_by_page' else 0.0
                utility = kw_gain + (0.4 * page_gain) + tool_bonus + (0.05 * max(0.0, float(getattr(h, 'score', 0.0))))
                image_density = utility / max(1, image_cost)
                score = float(getattr(h, 'score', 0.0))
                if _prefer_with_text_tiebreak(
                    image_density,
                    best_density,
                    text_cost if full_vlm_cost_aware else 0.0,
                    best_text_cost,
                    score,
                    best_score,
                ):
                    best_idx = idx
                    best_density = image_density
                    best_score = score
                    best_text_cost = text_cost if full_vlm_cost_aware else 0.0
                    best_new_pages = new_pages
                    best_kw_updates = kw_updates
                    best_new_text_ids = new_text_ids

            if best_idx < 0:
                break

            chosen = remaining.pop(best_idx)
            selected.append(chosen)
            selected_pages.update(best_new_pages)
            if full_vlm_cost_aware:
                selected_text_block_ids.update(best_new_text_ids)
            for kw, s in best_kw_updates.items():
                if s > best_kw[kw]:
                    best_kw[kw] = s

        return selected, {
            "phase1_budget": phase1_budget,
            "phase1_raw_count": len(raw_phase1),
            "phase1_compacted_count": len(selected),
            "phase1_compaction_applied": True,
            "phase1_compaction_dropped": len(raw_phase1) - len(selected),
        }

    # Hard-coverage initialization.
    raw_phase1 = []
    for h in cands:
        if h.tool in PHASE1_TOOLS:
            raw_phase1.append(h)

    effective_k_max, dynamic_budget_meta = _derive_effective_k_max(cands)
    phase1_budget = max(1, int(effective_k_max))
    locked = [h for h in raw_phase1 if _is_successful_aggregation_candidate(h)]
    locked_ids = {id(h) for h in locked}
    remaining_prefix = [h for h in raw_phase1 if id(h) not in locked_ids]
    room = max(0, phase1_budget - len(locked))
    phase1 = locked + remaining_prefix[:room]
    phase1_meta = {
        "phase1_budget": phase1_budget,
        "phase1_raw_count": len(raw_phase1),
        "phase1_compacted_count": len(phase1),
        "phase1_compaction_applied": len(raw_phase1) > len(phase1),
        "phase1_compaction_dropped": len(raw_phase1) - len(phase1),
        "phase1_compaction_mode": "prefix_preserve_aggregation",
    }
    phase1_ids = set()
    for h in phase1:
        phase1_ids.add(id(h))
        _update_soft(h)

    selected_pages_for_cost = set()
    selected_text_block_ids_for_cost = set()
    for h in phase1:
        selected_pages_for_cost.update(_pages_of(h))
        if full_vlm_cost_aware:
            caption_entries, related_entries = _collect_vlm_text_blocks(h)
            selected_text_block_ids_for_cost.update(block_id for block_id, _ in caption_entries)
            selected_text_block_ids_for_cost.update(block_id for block_id, _ in related_entries)

    phase1_log = [{"tool": h.tool, "score": round(h.score, 4)} for h in phase1]

    # Greedy soft augmentation.
    soft_pool = [h for h in cands if id(h) not in phase1_ids]
    phase2 = []
    phase2_log = []

    # Hard constraint satisfaction check — uses actual content matching via
    # compute_coverage, NOT just "did a hard tool fire".
    # Checking actual hard coverage prevents soft augmentation from adding
    # irrelevant evidence when a hard constraint has already been satisfied.
    _hard_constraints = {k: v for k, v in constraints.items()
                         if k in ('id', 'page_range', 'section_title') and v}
    n_hard       = len(_hard_constraints)
    hard_cov_p1  = (compute_coverage(phase1, _hard_constraints)
                    if _hard_constraints else 1.0)

    # Aggregation results for count/enumerate can already provide a complete
    # combined evidence item, so soft augmentation is skipped for that case.
    _has_aggregation = any(
        h.raw_dict.get('is_count_result') or h.raw_dict.get('is_enumerate_result')
        for h in phase1
    )
    if _has_aggregation and (maximize or intent_type in ("count", "enumerate")):
        phase2_log.append({"stop": "universal_satisfier_early_termination"})

    elif not maximize and _hard_constraints and hard_cov_p1 >= 1.0:
        # Hard constraints are already satisfied; skip soft augmentation.
        phase2_log.append({"stop": "hard_constraints_satisfied_cov=1.0"})

    elif maximize:
        # Unconstrained collection with a quality floor.
        # Used when aggregation tool did not fire for DOCUMENT-scope queries.
        # Equivalent to the general greedy with budget k=∞ and threshold τ
        # replacing the budget constraint.
        if cost_aware_select:
            _top1_density = 0.0
            while soft_pool and len(phase2) < effective_k_max:
                best_h, best_mf = None, -1.0
                best_density, best_cost, best_new_pages = -1.0, 1.0, set()
                best_score, best_text_cost = -1.0, float("inf")
                best_image_cost, best_text_cost, best_text_words = 1, 0.0, 0
                best_new_text_ids = set()
                for h in soft_pool:
                    mf = _mean_f(h)
                    dc, npages, image_cost, text_cost, text_words, new_text_ids = _delta_vlm_cost(
                        h, selected_pages_for_cost, selected_text_block_ids_for_cost
                    )
                    image_density = mf / max(1, image_cost)
                    score = float(getattr(h, 'score', 0.0))
                    if _prefer_with_text_tiebreak(
                        image_density,
                        best_density,
                        text_cost if full_vlm_cost_aware else 0.0,
                        best_text_cost if full_vlm_cost_aware else float("inf"),
                        score,
                        best_score,
                    ):
                        best_h, best_mf = h, mf
                        best_density, best_cost, best_new_pages = image_density, dc, npages
                        best_score = score
                        best_text_cost = text_cost if full_vlm_cost_aware else 0.0
                        best_image_cost, best_text_cost, best_text_words = image_cost, text_cost, text_words
                        best_new_text_ids = new_text_ids

                if best_h is None or best_mf < maximize_kw_threshold:
                    phase2_log.append({"stop": f"maximize_threshold={maximize_kw_threshold}"})
                    break
                if _top1_density and best_density < float(density_alpha) * _top1_density:
                    phase2_log.append({
                        "stop": (
                            f"density={best_density:.4f}<alpha*first="
                            f"{float(density_alpha) * _top1_density:.4f}"
                            f"(α={density_alpha},first={_top1_density:.4f},"
                            f"cost={best_cost})"
                        )
                    })
                    break

                phase2.append(best_h)
                _update_soft(best_h)
                selected_pages_for_cost.update(best_new_pages)
                if full_vlm_cost_aware:
                    selected_text_block_ids_for_cost.update(best_new_text_ids)
                if not _top1_density:
                    _top1_density = best_density
                phase2_log.append({
                    "tool":    best_h.tool,
                    "score":   round(best_h.score, 4),
                    "mean_f":  round(best_mf, 4),
                    "delta_vlm_cost": round(best_cost, 4),
                    "delta_image_cost": int(best_image_cost),
                    **({
                        "delta_text_cost": round(best_text_cost, 4),
                        "delta_text_words": int(best_text_words),
                    } if full_vlm_cost_aware else {}),
                    "density": round(best_density, 4),
                    "density_threshold": round(float(density_alpha) * _top1_density, 4),
                    "new_pages": sorted(best_new_pages),
                })
                soft_pool.remove(best_h)
            if not phase2_log or "stop" not in phase2_log[-1]:
                phase2_log.append({"stop": f"k_max={effective_k_max} reached" if len(phase2) >= effective_k_max else "pool_exhausted"})
        else:
            for h in soft_pool:
                mf = _mean_f(h)
                if mf >= maximize_kw_threshold:
                    phase2.append(h)
                    _update_soft(h)
                    phase2_log.append({
                        "tool":    h.tool,
                        "score":   round(h.score, 4),
                        "mean_f":  round(mf, 4),
                    })
            phase2_log.append({"stop": f"maximize_threshold={maximize_kw_threshold}"})
    else:
        # Budget-constrained greedy soft augmentation.
        # Iteratively selects h* = argmax δ_soft(h | S) from the remaining pool.
        #
        # Stopping criterion (adaptive threshold τ(S)):
        #   δ_soft(h*) ≥ τ(S) = max(ε, α · δ₁)   where δ₁ = first pick's gain
        #   AND selected soft candidates stay within k_max.
        #
        #   The relative component adapts the bar to query relevance.
        _top1_delta: float = 0.0   # δ_soft of the first selected HE (0 until set)
        _top1_density: float = 0.0

        while soft_pool and len(phase2) < effective_k_max:
            best_h, best_ds = None, -1.0
            best_density, best_cost, best_new_pages = -1.0, 1.0, set()
            best_score, best_text_cost = -1.0, float("inf")
            best_image_cost, best_text_cost, best_text_words = 1, 0.0, 0
            best_new_text_ids = set()
            for h in soft_pool:
                ds = _delta_soft(h)
                if cost_aware_select:
                    dc, npages, image_cost, text_cost, text_words, new_text_ids = _delta_vlm_cost(
                        h, selected_pages_for_cost, selected_text_block_ids_for_cost
                    )
                    image_density = ds / max(1, image_cost)
                    score = float(getattr(h, 'score', 0.0))
                    if _prefer_with_text_tiebreak(
                        image_density,
                        best_density,
                        text_cost if full_vlm_cost_aware else 0.0,
                        best_text_cost if full_vlm_cost_aware else float("inf"),
                        score,
                        best_score,
                    ):
                        best_density, best_cost, best_new_pages = image_density, dc, npages
                        best_score = score
                        best_text_cost = text_cost if full_vlm_cost_aware else 0.0
                        best_image_cost, best_text_cost, best_text_words = image_cost, text_cost, text_words
                        best_new_text_ids = new_text_ids
                        best_ds, best_h = ds, h
                elif ds > best_ds + 1e-9:
                    best_ds, best_h = ds, h

            # Adaptive threshold: absolute floor + relative drop guard.
            # Only applied to UNIT; MULTI_UNIT uses per-keyword marginal gains.
            if _top1_delta and relative_gain_alpha and not is_multi:
                _threshold = max(min_soft_gain,
                                 relative_gain_alpha * _top1_delta)
            else:
                _threshold = min_soft_gain

            cost_stop = False
            if cost_aware_select and best_h is not None and _top1_density:
                min_keep = max(0, int(multi_min_phase2)) if is_multi else 0
                cost_stop = (
                    len(phase2) >= min_keep
                    and best_density < float(density_alpha) * _top1_density
                )

            if best_h is None or best_ds < _threshold or cost_stop:
                if cost_stop:
                    stop_msg = (
                        f"density={best_density:.4f}<alpha*first="
                        f"{float(density_alpha) * _top1_density:.4f}"
                        f"(α={density_alpha},first={_top1_density:.4f},"
                        f"cost={best_cost})"
                    )
                elif _top1_delta:
                    stop_msg = (
                        f"delta_soft={best_ds:.4f}<threshold={_threshold:.4f}"
                        f"(α={relative_gain_alpha},top1={_top1_delta:.4f})"
                    )
                else:
                    stop_msg = (
                        f"delta_soft={best_ds:.4f}<{min_soft_gain}"
                        if best_ds >= 0 else "no_candidate"
                    )
                phase2_log.append({
                    "stop": stop_msg,
                })
                break

            phase2.append(best_h)
            _update_soft(best_h)
            if cost_aware_select:
                selected_pages_for_cost.update(best_new_pages)
                if full_vlm_cost_aware:
                    selected_text_block_ids_for_cost.update(best_new_text_ids)
            if not _top1_delta:
                _top1_delta = best_ds   # calibrate relative threshold on first pick
            if cost_aware_select and not _top1_density:
                _top1_density = best_density
            phase2_log.append({
                "tool":        best_h.tool,
                "score":       round(best_h.score, 4),
                "delta_soft":  round(best_ds, 4),
                "threshold":   round(_threshold, 4),
                **({
                    "delta_vlm_cost": round(best_cost, 4),
                    "delta_image_cost": int(best_image_cost),
                    **({
                        "delta_text_cost": round(best_text_cost, 4),
                        "delta_text_words": int(best_text_words),
                    } if full_vlm_cost_aware else {}),
                    "density": round(best_density, 4),
                    "density_threshold": round(float(density_alpha) * _top1_density, 4),
                    "new_pages": sorted(best_new_pages),
                } if cost_aware_select else {}),
            })
            soft_pool.remove(best_h)

        if not phase2_log or "stop" not in phase2_log[-1]:
            if len(phase2) >= effective_k_max:
                phase2_log.append({"stop": f"k_max={effective_k_max} reached"})
            elif not soft_pool:
                phase2_log.append({"stop": "pool_exhausted"})

    # ── Output ────────────────────────────────────────────────────────────
    selected = phase1 + phase2

    # Intent-specific reranking before block pruning.
    if final_rerank and phase2:
        phase2_reranked = intent_rerank(phase2, evidence_level)
        selected = phase1 + phase2_reranked
    else:
        phase2_reranked = phase2

    # Optional block-level pruning for soft-tool evidence.
    phase3_pruned = 0
    phase3_blocks_removed = 0
    if block_prune and keywords and phase2_reranked:
        for he in phase2_reranked:
            before = len(he.related_blocks)
            removed = _ef_prune_blocks(he, keywords, alpha=block_prune_alpha,
                                       preserve_title_blocks=preserve_title_blocks)
            if removed:
                phase3_pruned += 1
                phase3_blocks_removed += before - len(he.related_blocks)

    # Final cap runs after optimizer-internal modules:
    # phase3 reranking and block pruning happen before the VLM evidence cap.
    final_cap_meta = {
        "applied": False,
        "limit": None,
        "before": len(selected),
        "after": len(selected),
        "dropped": 0,
        "mode": "disabled",
    }
    if optimizer_final_cap:
        if any(_is_successful_aggregation_candidate(h) for h in selected):
            final_cap_meta = {
                "applied": False,
                "limit": dynamic_budget_meta["effective_k_max"],
                "before": len(selected),
                "after": len(selected),
                "dropped": 0,
                "mode": "aggregation_preserved",
            }
        else:
            selected, final_cap_meta = _apply_final_evidence_cap(
                selected,
                dynamic_budget_meta["effective_k_max"],
                evidence_level,
                intent_type,
                final_cap_group_mode,
            )

    # Hard coverage on the full selected set: constraint satisfaction ratio,
    # not "tool fired ratio".  _hard_constraints and n_hard were already
    # computed above and are available here.
    final_hard = (compute_coverage(selected, _hard_constraints)
                  if _hard_constraints else 1.0)
    if not keywords:
        final_soft = 1.0
    else:
        # For logging: track both aggregate and per-kw coverage.
        # is_multi kept for opt_log reporting only (not used in selection logic).
        final_soft = best_combined[0]

    opt_log = {
        "mode":            "maximize" if maximize else "select",
        "evidence_level":  evidence_level,
        "intent_type":     intent_type,
        "n_hard":          n_hard,
        "hard_cov_p1":     round(hard_cov_p1, 4),
        "n_keywords":      len(keywords),
        "phase1":          phase1_log,
        "phase2":          phase2_log,
        "phase1_count":    len(phase1),
        "phase1_budget":   phase1_meta["phase1_budget"],
        "phase1_raw_count": phase1_meta["phase1_raw_count"],
        "phase1_compaction_applied": phase1_meta["phase1_compaction_applied"],
        "phase1_compaction_dropped": phase1_meta["phase1_compaction_dropped"],
        "phase2_count":    len(phase2),
        "legacy_k_max":    dynamic_budget_meta["legacy_k_max"],
        "effective_k_max": dynamic_budget_meta["effective_k_max"],
        "dynamic_vlm_budget_meta": dynamic_budget_meta,
        "final_evidence_cap": final_cap_meta,
        "optimizer_final_cap": optimizer_final_cap,
        "selected_total":  len(selected),
        "final_hard_cov":  round(final_hard, 4),
        "final_soft_cov":  round(final_soft, 4),
        "best_combined":   round(best_combined[0], 4) if not is_multi else None,
        "best_fs":         {kw: round(v, 4) for kw, v in best_fs.items()} if is_multi else None,
        "final_rerank":      final_rerank,
        "block_prune":     block_prune,
        "cost_aware_select": cost_aware_select,
        "full_vlm_cost_aware": full_vlm_cost_aware if cost_aware_select else None,
        "text_cost_unit_words": text_cost_unit_words if full_vlm_cost_aware else None,
        "text_cost_weight": text_cost_weight if full_vlm_cost_aware else None,
        "related_text_cost_discount": related_text_cost_discount if full_vlm_cost_aware else None,
        "text_cost_tiebreak_eps": text_cost_tiebreak_eps if full_vlm_cost_aware else None,
        "density_alpha":   density_alpha if cost_aware_select else None,
        "multi_min_phase2": multi_min_phase2 if cost_aware_select else None,
        "phase3_hes_pruned":     phase3_pruned,
        "phase3_blocks_removed": phase3_blocks_removed,
    }

    mode_tag = "maximize" if maximize else evidence_level
    p25_msg = ",rerank" if final_rerank else ""
    p3_msg = (f", p3_pruned={phase3_pruned}({phase3_blocks_removed}blks)"
              if block_prune else "")
    print(
        f"  [optimizer/{mode_tag}{p25_msg}] "
        f"phase1={len(phase1)}"
        f"{'/' + str(phase1_meta['phase1_raw_count']) if phase1_meta['phase1_compaction_applied'] else ''}, "
        f"phase2={len(phase2)}, "
        f"total={len(selected)}, soft_cov={final_soft:.2f}"
        f"{', final_cap=' + str(final_cap_meta['limit']) if final_cap_meta.get('applied') else ''}"
        f"{p3_msg}"
    )
    return selected, opt_log


# Legacy aliases for backward compatibility
def lexicographic_greedy_optimizer(candidates, constraints, keywords,
                                   evidence_level="UNIT", min_soft_gain=0.01,
                                   k_max=5, intent_type="locate"):
    """Backward-compatible wrapper → select_hyperedges (select mode)."""
    return select_hyperedges(
        candidates=candidates, constraints=constraints, keywords=keywords,
        evidence_level=evidence_level, intent_type=intent_type,
        min_soft_gain=min_soft_gain, k_max=k_max,
    )

def maximize_coverage_optimizer(candidates, keywords, min_kw_threshold=0.05):
    """Backward-compatible wrapper → select_hyperedges (maximize mode)."""
    return select_hyperedges(
        candidates=candidates, constraints={}, keywords=keywords,
        evidence_level="DOCUMENT", intent_type="count",
        maximize_kw_threshold=min_kw_threshold,
    )


# ============================================================================
# Helpers
# ============================================================================

def _is_page_node(e) -> bool:
    """Page-level nodes emitted by tool_visual_search (no bound hyperedge/anchor).

    NOTE: extra_meta fields are spread to the top level of raw_dict via **extra_meta
    in QueryHyperedge.create_from_components, so is_page_node lives at raw_dict top level.
    """
    if hasattr(e, 'raw_dict'):
        return bool(e.raw_dict.get('is_page_node'))
    return False


def _page_of(e) -> Optional[int]:
    """Return 0-based physical page index for any edge type (first page found).

    Prefer _all_pages_of when you need the full set.
    """
    pages = _all_pages_of(e)
    return min(pages) if pages else None


def _all_pages_of(e) -> set:
    """Return all 0-based page indices referenced by this edge (QueryHyperedge object).

    Handles both page-only nodes (source_page at raw_dict top level) and
    block-based hyperedges (anchor_block.page / caption_block.page / related_blocks[i].page).
    """
    pages: set = set()
    if not hasattr(e, 'raw_dict'):
        return pages
    rd = e.raw_dict

    # Page-only nodes — source_page stored at TOP LEVEL of raw_dict
    # (extra_meta is spread via **extra_meta in create_from_components)
    src = rd.get('source_page')
    if isinstance(src, int):
        pages.add(src)

    # Block-based hyperedges
    for key in ('anchor_block', 'caption_block'):
        b = rd.get(key) or {}
        if isinstance(b, dict):
            p = b.get('page')
            if isinstance(p, int):
                pages.add(p)

    for rb in rd.get('related_blocks') or []:
        if isinstance(rb, dict):
            p = rb.get('page')
            if isinstance(p, int):
                pages.add(p)

    return pages


def _all_text(e) -> str:
    """Concatenate all text fields available in a QueryHyperedge."""
    if not hasattr(e, 'raw_dict'):
        return ''
    rd = e.raw_dict
    parts = []
    for key in ('anchor_block', 'caption_block'):
        b = rd.get(key) or {}
        if b.get('text'):
            parts.append(b['text'])
    for rb in rd.get('related_blocks') or []:
        if rb and rb.get('text'):
            parts.append(rb['text'])
    return ' '.join(parts).lower()


def _caption_text(e) -> str:
    if not hasattr(e, 'raw_dict'):
        return ''
    return ((e.raw_dict.get('caption_block') or {}).get('text') or '').lower()


def _anchor_type(e) -> str:
    if not hasattr(e, 'raw_dict'):
        return ''
    return ((e.raw_dict.get('anchor_block') or {}).get('type') or '').lower()


# ============================================================================
# Coverage
# ============================================================================

def compute_coverage(
    edge_set: List,
    constraints: Dict,
) -> float:
    """
    Coverage(E, q) = |{c ∈ C(q) | E ⊨ c}| / |C(q)|

    Atomic constraint decomposition
    ────────────────────────────────
    Constraint        Satisfied when
    ─────────         ──────────────
    id                any fine-grained edge caption/text contains id string
    page_range        any edge (fine-grained OR page_node) is on a target page
    section_title     any fine-grained edge text contains section name
    type              any fine-grained edge anchor has matching type
    keywords[i]       any fine-grained edge text fuzzy-matches keyword i

    Page-level nodes (is_page_node=True) only contribute to page_range satisfaction.
    Returns 1.0 when there are no constraints (vacuously satisfied).
    """
    if not constraints:
        return 1.0

    fine   = [e for e in edge_set if not _is_page_node(e)]
    all_e  = edge_set   # both fine + page nodes usable for page_range

    satisfied = 0
    total     = 0

    # ── 1. ID ────────────────────────────────────────────────────────────
    target_id = constraints.get('id')
    if target_id:
        total += 1
        tid = target_id.lower().strip()
        if any(tid in _caption_text(e) for e in fine) or \
           any(tid in _all_text(e) for e in fine):
            satisfied += 1

    # ── 2. Page range ──────────────────────────────────────────────────
    page_range = constraints.get('page_range')
    if page_range:
        total += 1
        # constraints store 1-based; edges store 0-based
        target_0 = {p - 1 for p in page_range if isinstance(p, int) and p > 0}
        if any(_all_pages_of(e) & target_0 for e in all_e):
            satisfied += 1

    # ── 3. Section title ──────────────────────────────────────────────
    section_title = constraints.get('section_title')
    if section_title:
        total += 1
        sl = section_title.lower()
        if any(sl in _all_text(e) for e in fine):
            satisfied += 1

    # ── 4. Type — EXCLUDED (OCR noise causes unreliable signal) ────────
    # target_type = constraints.get('type')  # intentionally skipped

    # ── 5. Keywords (each keyword = one atomic constraint) ────────────
    for kw in constraints.get('keywords', []):
        total += 1
        if any(fuzzy_keyword_match(kw, _all_text(e), threshold=0.80)
               for e in fine if _all_text(e)):
            satisfied += 1

    return satisfied / total if total > 0 else 1.0

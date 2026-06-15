#!/usr/bin/env python3
"""
Retrieval Tools
==============
Implementations of various retrieval tools used by the router.
"""

import re
import math
import json
from collections import Counter
from typing import List, Dict, Optional, Union, Tuple

from .graph_structures import QueryHyperedge
from .retrieval_helpers import (
    build_block_index,
    get_visual_contextual_edges,
    get_anchor_block_from_edge,
    get_caption_block_from_edge,
    get_related_blocks_from_edge,
    extract_text_from_blocks,
    filter_edges_by_type,
    filter_edges_by_page_range,
    filter_edges_by_section_title,
    get_pages_from_section_ranges,
    get_title_block_for_anchor,
    match_unit_id,
    match_section_title,
    fuzzy_keyword_match,
    fuzzy_keyword_match_score,
    is_visual_type
)


def _load_vlm_json_object(output: str) -> Tuple[Optional[Dict], Optional[str], Optional[str]]:
    """Extract and parse a JSON object from VLM text."""
    if not isinstance(output, str) or not output.strip():
        return None, None, "empty_output"

    candidates = [output.strip()]
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", output, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1).strip())

    start = output.find("{")
    end = output.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(output[start:end + 1].strip())

    seen = set()
    last_error = None
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            return json.loads(candidate), candidate, None
        except json.JSONDecodeError as exc:
            last_error = str(exc)

    return None, None, last_error or "no_json_object"


def _recover_per_image_matches(output: str) -> Tuple[List[int], List[Dict]]:
    """Recover per-image match flags from malformed or truncated JSON."""
    if not isinstance(output, str):
        return [], []

    per_image = []
    matched_indices = []
    pattern = re.compile(
        r'\{[^{}]*"image_index"\s*:\s*(\d+)[^{}]*"match"\s*:\s*(true|false)',
        re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(output):
        image_index = int(match.group(1))
        is_match = match.group(2).lower() == "true"
        item = {"image_index": image_index, "match": is_match, "recovered": True}
        per_image.append(item)
        if is_match:
            matched_indices.append(image_index - 1)

    return matched_indices, per_image


def _find_pages_by_keywords(block_index: Dict, keywords: List[str]) -> List[int]:
    """Find pages containing the given keywords (e.g. for locating 'Mobile Internet Demographics slide')."""
    if not keywords:
        return []

    page_scores = Counter()
    
    for block in block_index.values():
        page = block.get('page')
        if page is None:
            continue
            
        text = block.get('text', '').lower()
        if not text:
            continue
            
        # Check text against keywords
        matched_count = 0
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower in text:
                matched_count += 1
                # Boost if it looks like a title
                if block.get('type') == 'title' or len(text) < 100:
                    matched_count += 2
        
        if matched_count > 0:
            page_scores[page] += matched_count

    # Return top 3 pages with highest keyword overlaps (0-based)
    if not page_scores:
        return []
    
    # Filter reasonable matches (at least 2 score?) - relaxed for now
    top_pages = [p for p, score in page_scores.most_common(3)]
    return sorted(top_pages)


def _convert_page_range_to_physical(
    page_range: List,
    reasoner,
    page_image_dir: str,
    clean_doc_id: str
) -> List[int]:
    """
    Convert 1-based user page range to 0-based physical indices using VLM-based mapping.
    
    Supports relative page references:
    - "LAST" → last page number
    - "FIRST" → page 1
    
    Args:
        page_range: List of 1-based page numbers or special strings ("LAST", "FIRST")
        reasoner: VLM reasoner
        page_image_dir: Page image directory
        clean_doc_id: Document ID
    
    Returns:
        List of 0-based physical indices
    """
    from hyperedge.prompts import format_page_extraction_prompt
    from hyperedge.retrieval_helpers import (
        probe_page_with_vlm,
        calculate_pagination_rule,
        convert_user_pages_to_physical
    )
    import glob
    import os
    
    if not page_range:
        return []
    
    # Handle relative page references (LAST, FIRST)
    resolved_pages = []
    has_relative_refs = False
    
    for p in page_range:
        if isinstance(p, str) and p.upper() in ["LAST", "FIRST"]:
            has_relative_refs = True
            if p.upper() == "FIRST":
                resolved_pages.append(1)
                print(f"     → [_convert_page_range_to_physical] Resolved 'FIRST' to page 1")
            elif p.upper() == "LAST":
                # Get total page count from images directory
                img_pattern = os.path.join(page_image_dir, f"{clean_doc_id}_*.png")
                img_files = glob.glob(img_pattern)
                if img_files:
                    total_pages = len(img_files)
                    resolved_pages.append(total_pages)
                    print(f"     → [_convert_page_range_to_physical] Resolved 'LAST' to page {total_pages} (total {total_pages} pages)")
                else:
                    print(f"     ⚠️ [_convert_page_range_to_physical] Cannot resolve 'LAST': no images found for {clean_doc_id}")
                    # Fallback: assume large page number
                    resolved_pages.append(999)
        elif isinstance(p, int):
            resolved_pages.append(p)
    
    page_range = resolved_pages
    
    if not page_range:
        return []
    
    # If VLM not available, use simple 1-based to 0-based conversion
    if not reasoner or not page_image_dir or not clean_doc_id:
        fallback_result = [p - 1 for p in page_range if p > 0]
        print(f"     ⚠️ [_convert_page_range_to_physical] VLM not available, using simple fallback: {page_range} (1-based) -> {fallback_result} (0-based)")
        return fallback_result
    
    # Probe and calculate page mapping
    first_user_page = page_range[0]
    initial_guess = first_user_page - 1
    probe_indices = [max(0, initial_guess + offset) for offset in range(-2, 3)]
    
    print(f"     → [_convert_page_range_to_physical] Probing around page {first_user_page} (1-based): physical indices {probe_indices}")
    
    vlm_results = []
    for probe_idx in probe_indices:
        result = probe_page_with_vlm(
            reasoner,
            page_image_dir,
            clean_doc_id,
            probe_idx,
            format_page_extraction_prompt
        )
        if result:
            vlm_results.append(result)
            print(f"       ✓ Probe {probe_idx}: layout={result.get('layout')}, pages={result.get('found_pages')}")
    
    if vlm_results:
        page_mapping = calculate_pagination_rule(vlm_results, probe_indices)
        physical_result = convert_user_pages_to_physical(page_range, page_mapping)
        print(f"     ✓ [_convert_page_range_to_physical] Mapping calculated: {page_range} (1-based) -> {physical_result} (0-based)")
        print(f"       Layout: {page_mapping.get('layout')}, Confidence: {page_mapping.get('confidence', 0):.2f}")
        return physical_result
    
    # Fallback: VLM probing failed
    fallback_result = [p - 1 for p in page_range if p > 0]
    print(f"     ⚠️ [_convert_page_range_to_physical] VLM probing failed, using fallback: {page_range} (1-based) -> {fallback_result} (0-based)")
    return fallback_result

# ============================================================================
# Retrieval Tools
# ============================================================================

def tool_find_by_id(hypergraph: Dict, constraints: Dict, multi_id_mode=False, **kwargs) -> List[QueryHyperedge]:
    """"""
    target_id = constraints.get("id") or constraints.get("figure_id")
    target_type = constraints.get("type")
    
    if not target_id:
        return []
    
    #
    #
    target_ids = []
    if multi_id_mode:
        import re
        #
        id_pattern = re.compile(r'\d+')
        matches = id_pattern.findall(target_id)
        
        if len(matches) >= 2:
            #
            target_ids = matches
            print(f"    → Multi-ID mode: Found {len(target_ids)} IDs: {target_ids}")
        elif len(matches) == 1:
            #
            target_ids = matches
            print(f"    → Single-ID extracted: {target_ids[0]}")
    
    #
    if not target_ids:
        target_ids = [target_id]
        print(f"    → Using original ID: {target_id}")
    
    all_results = []
    seen_edge_ids = set()
    
    #
    block_index = build_block_index(hypergraph)
    visual_edges = filter_edges_by_type(
        get_visual_contextual_edges(hypergraph),
        block_index,
        target_type
    )
    
    #
    for current_id in target_ids:
        #
        #
        
        has_number = current_id.isdigit()
        id_results = []
        
        print(f"    → Searching for ID: {current_id} (Type: {target_type})")
        
        #
        caption_matches = []
        for edge in visual_edges:
            anchor_block = get_anchor_block_from_edge(edge, block_index)
            if not anchor_block:
                continue
            
            edge_id = edge.get('edge_id')
            if edge_id and edge_id in seen_edge_ids:
                continue
            
            caption_block = get_caption_block_from_edge(edge, block_index, anchor_block['block_id'])
            caption = caption_block.get('text', '') if caption_block else ''
            
            if has_number and caption and match_unit_id(caption, current_id, target_type):
                caption_matches.append({
                    'edge': edge,
                    'anchor': anchor_block,
                    'caption': caption_block,
                    'score': 1.0,
                    'reason': f"Matched ID {current_id} in caption (Type: {target_type})"
                })
                if edge_id:
                    seen_edge_ids.add(edge_id)
        
        if caption_matches:
            print(f"      Caption match: found {len(caption_matches)} match(es)")
            for match in caption_matches:
                id_results.append(QueryHyperedge.create_from_components(
                    edge=match['edge'],
                    anchor_block=match['anchor'],
                    caption_block=match['caption'],
                    related_blocks=get_related_blocks_from_edge(match['edge'], block_index, match['anchor']['block_id']),
                    score=match['score'],
                    match_reason=match['reason'],
                    tool='tool_find_by_id'
                ))
        else:
            #
            print(f"      Sequence inference: caption not found, inferring from sequence...")
            inferred_match = _infer_missing_figure_by_sequence(
                current_id, target_type, visual_edges, block_index, seen_edge_ids
            )
            if inferred_match:
                print(f"      Sequence inference: inferred Figure {current_id}")
                id_results.append(inferred_match)
                # Extract edge_id from raw_dict['query_hyperedge']
                if hasattr(inferred_match, 'raw_dict'):
                    edge_dict = inferred_match.raw_dict.get('query_hyperedge', {})
                    if edge_dict and edge_dict.get('edge_id'):
                        seen_edge_ids.add(edge_dict['edge_id'])
            else:
                print(f"      Sequence inference: could not infer from sequence")
        
        #
        
        if not id_results:
            print(f"      ✗ No matches for ID {current_id}")
        
        #
        all_results.extend(id_results)
    
    return all_results


def _infer_missing_figure_by_sequence(
    target_id: str,
    target_type: str,
    visual_edges: List[Dict],
    block_index: Dict,
    seen_edge_ids: set
) -> Optional[QueryHyperedge]:
    """"""
    if not target_id.isdigit():
        return None
    
    target_num = int(target_id)
    import re
    
    #
    all_figure_edges = []
    for edge in visual_edges:
        anchor = get_anchor_block_from_edge(edge, block_index)
        if not anchor:
            continue
        
        #
        if anchor.get('type') == target_type:
            edge_id = edge.get('edge_id')
            if edge_id and edge_id in seen_edge_ids:
                continue
            
            page = anchor.get('page', 0)
            bbox = anchor.get('bbox', [0, 0, 0, 0])
            y_pos = bbox[1] if len(bbox) > 1 else 0
            
            all_figure_edges.append({
                'edge': edge,
                'anchor': anchor,
                'page': page,
                'y_pos': y_pos,
                'sort_key': (page, y_pos)  #
            })
    
    #
    all_figure_edges.sort(key=lambda x: x['sort_key'])
    
    #
    figures_with_nums = []
    figures_without_nums = []
    
    for item in all_figure_edges:
        edge = item['edge']
        anchor = item['anchor']
        
        caption = get_caption_block_from_edge(edge, block_index, anchor['block_id'])
        if caption:
            caption_text = caption.get('text', '')
            match = re.search(r'(figure|fig)\s*(\d+)', caption_text, re.IGNORECASE)
            if match:
                fig_num = int(match.group(2))
                item['fig_num'] = fig_num
                item['caption'] = caption
                figures_with_nums.append(item)
                continue
        
        #
        item['fig_num'] = None
        item['caption'] = caption
        figures_without_nums.append(item)
    
    if len(figures_with_nums) < 2:
        return None  #
    
    #
    for i in range(len(figures_with_nums) - 1):
        fig_before = figures_with_nums[i]
        fig_after = figures_with_nums[i + 1]
        
        num_before = fig_before['fig_num']
        num_after = fig_after['fig_num']
        
        #
        if num_before < target_num < num_after:
            #
            idx_before = all_figure_edges.index(fig_before)
            idx_after = all_figure_edges.index(fig_after)
            
            #
            if idx_after - idx_before == 2:
                candidate = all_figure_edges[idx_before + 1]
                edge = candidate['edge']
                anchor = candidate['anchor']
                caption = candidate['caption']
                
                return QueryHyperedge.create_from_components(
                    edge=edge,
                    anchor_block=anchor,
                    caption_block=caption,
                    related_blocks=get_related_blocks_from_edge(edge, block_index, anchor['block_id']),
                    score=0.75,  #
                    match_reason=f"Inferred {target_type} {target_id} from sequence (between {num_before} and {num_after})",
                    tool='tool_find_by_id'
                )
            
            #
            elif idx_after - idx_before > 2:
                gap = num_after - num_before
                progress = (target_num - num_before) / gap
                inferred_idx = idx_before + int(progress * (idx_after - idx_before))
                
                #
                if idx_before < inferred_idx < idx_after:
                    candidate = all_figure_edges[inferred_idx]
                    edge = candidate['edge']
                    anchor = candidate['anchor']
                    caption = candidate['caption']
                    
                    return QueryHyperedge.create_from_components(
                        edge=edge,
                        anchor_block=anchor,
                        caption_block=caption,
                        related_blocks=get_related_blocks_from_edge(edge, block_index, anchor['block_id']),
                        score=0.65,  #
                        match_reason=f"Inferred {target_type} {target_id} from sequence (estimated position)",
                        tool='tool_find_by_id'
                    )
    
    return None


def tool_find_by_section(hypergraph: Dict, constraints: Dict, **kwargs) -> List[QueryHyperedge]:
    """"""
    target_section = constraints.get("section_title")
    target_type = constraints.get("type")
    if not target_section:
        print("  [tool_find_by_section] No section_title constraint, returning empty")
        return []
    
    print(f"  [tool_find_by_section] Executing with section_title='{target_section}', type={target_type}")
    block_index = build_block_index(hypergraph)
    results = []
    
    # Start with all visual contextual edges
    all_edges = get_visual_contextual_edges(hypergraph)
    print(f"  [tool_find_by_section] Starting with {len(all_edges)} visual contextual edges")
    
    # Filter by section
    matched_edges = filter_edges_by_section_title(all_edges, hypergraph, block_index, target_section)
    if not matched_edges:
        print(f"  [tool_find_by_section] ❌ No edges after section filtering, returning empty")
        return []
    
    print(f"  [tool_find_by_section] After section filter: {len(matched_edges)} edges")

    # Filter by type
    matched_edges = filter_edges_by_type(matched_edges, block_index, target_type)
    print(f"  [tool_find_by_section] After type filter: {len(matched_edges)} edges")

    for contextual_edge in matched_edges:
        anchor_block = get_anchor_block_from_edge(contextual_edge, block_index)
        if not anchor_block:
            continue
        caption_block = get_caption_block_from_edge(contextual_edge, block_index, anchor_block['block_id'])
        title_block = get_title_block_for_anchor(hypergraph, block_index, anchor_block['block_id'])
        results.append(QueryHyperedge.create_from_components(
            edge=contextual_edge,
            anchor_block=anchor_block,
            caption_block=caption_block,
            related_blocks=get_related_blocks_from_edge(contextual_edge, block_index, anchor_block['block_id']),
            score=0.9,
            match_reason=f"Matched Section: {target_section}",
            tool='tool_find_by_section',
            extra_meta={'title_block': title_block} if title_block else None
        ))
    
    print(f"  [tool_find_by_section] ✓ Returning {len(results)} results")
    return results


def tool_find_by_page(
    hypergraph: Dict, 
    constraints: Dict, 
    reasoner=None, 
    page_image_dir: str = "",
    clean_doc_id: str = "",
    **kwargs
) -> List[QueryHyperedge]:
    """"""
    from hyperedge.prompts import format_page_extraction_prompt
    from hyperedge.retrieval_helpers import (
        probe_page_with_vlm,
        calculate_pagination_rule,
        convert_user_pages_to_physical
    )
    
    # Schema uses page_range: list of 1-based page numbers
    page_range = constraints.get("page_range")
    target_type = constraints.get("type")
    if not page_range:
        return []
        
    #
    user_pages_1based = page_range
    if not user_pages_1based:
        return []
    
    print(f"  [tool_find_by_page] User target pages (1-based): {user_pages_1based}")
    print(f"  [tool_find_by_page] Target type: {target_type}")

    # ============================================================================
    # 🔧 Handle "LAST"/"FIRST" string values in page_range
    # ============================================================================
    # LAST/FIRST are resolved directly to 0-based physical indices and short-circuit
    # VLM page mapping entirely.  Feeding "total_images" into the VLM mapper causes
    # it to compute an out-of-range physical index (document page numbers ≠ image
    # count), so expanded_pages ends up wrong and no blocks are found.
    direct_physical_pages = []  # 0-based indices resolved from LAST/FIRST
    if page_image_dir and clean_doc_id:
        import glob
        page_images = glob.glob(f"{page_image_dir}/{clean_doc_id}_*.png")
        total_images = len(page_images)

        remaining_pages = []
        for page in user_pages_1based:
            if isinstance(page, str):
                page_upper = page.upper()
                if page_upper == "LAST":
                    phys = total_images - 1
                    direct_physical_pages.append(phys)
                    print(f"  [tool_find_by_page] 'LAST' -> physical index {phys} (total images: {total_images})")
                elif page_upper == "FIRST":
                    direct_physical_pages.append(0)
                    print(f"  [tool_find_by_page] 'FIRST' -> physical index 0")
                else:
                    try:
                        remaining_pages.append(int(page))
                    except Exception:
                        print(f"  [tool_find_by_page] ⚠️ Cannot parse page '{page}', skipping")
            else:
                remaining_pages.append(page)
        user_pages_1based = remaining_pages

    if not user_pages_1based and not direct_physical_pages:
        return []

    # ============================================================================
    # VLM-based Pagination Detection & Conversion (only for numeric page_range)
    # ============================================================================
    expanded_pages = list(direct_physical_pages)  # seed with already-resolved pages
    page_mapping = None

    if reasoner and clean_doc_id and page_image_dir and user_pages_1based:
        print(f"  [tool_find_by_page] 🔍 VLM-based Page Conversion starts...")
        
        # Strategy: Probe around first target page to discover layout pattern
        # Use first user page as reference (1-based)
        first_user_page_1based = user_pages_1based[0]
        
        # Initial guess: assume 1-based to 0-based simple conversion for probing
        initial_guess_idx = first_user_page_1based - 1
        probe_indices = []
        
        # Probe wider range to detect pattern (especially for double-page)
        for offset in range(-2, 3):  # [-2, -1, 0, 1, 2]
            candidate = initial_guess_idx + offset
            if candidate >= 0:
                probe_indices.append(candidate)
        
        print(f"  [tool_find_by_page] Probing physical indices around {initial_guess_idx}: {probe_indices}")
        
        vlm_results = []
        for probe_idx in probe_indices:
            result = probe_page_with_vlm(
                reasoner,
                page_image_dir,
                clean_doc_id,
                probe_idx,
                format_page_extraction_prompt
            )
            if result:
                vlm_results.append(result)
                print(f"     → Probed Index {probe_idx}: layout={result['layout']}, found_pages={result['found_pages']}")
        
        # Calculate page mapping
        if vlm_results:
            page_mapping = calculate_pagination_rule(vlm_results, probe_indices)
            
            print(f"     ✓ Page Mapping:")
            print(f"       - Layout: {page_mapping.get('layout', 'unknown')}")
            print(f"       - Confidence: {page_mapping.get('confidence', 0.0):.2f}")
            if page_mapping.get('layout') == 'double':
                print(f"       - Stride: {page_mapping.get('stride', 2)}")
                print(f"       - Formula: {page_mapping.get('formula', 'N/A')}")
            elif page_mapping.get('layout') == 'single':
                print(f"       - Base Offset: {page_mapping.get('base_offset', 0)}")
            
            #
            expanded_pages = convert_user_pages_to_physical(user_pages_1based, page_mapping)
            
            print(f"     → Converted user pages {user_pages_1based} (1-based) -> Physical indices: {expanded_pages}")
        else:
            print(f"     ⚠️ No valid VLM results, using fallback expansion")
    
    # Fallback: simple conversion for any numeric pages not yet resolved by VLM.
    # direct_physical_pages are already in expanded_pages; only user_pages_1based
    # (numeric, non-LAST/FIRST) might still be missing if VLM probing failed.
    if user_pages_1based and not any(p - 1 in expanded_pages for p in user_pages_1based if isinstance(p, int)):
        print(f"  [tool_find_by_page] Using fallback: 1-based to 0-based conversion for {user_pages_1based}")
        for p in user_pages_1based:
            physical_idx = p - 1
            if physical_idx >= 0:
                expanded_pages.append(physical_idx)
                if physical_idx + 1 not in expanded_pages:
                    expanded_pages.append(physical_idx + 1)
        expanded_pages = sorted(list(set(expanded_pages)))

    print(f"  [tool_find_by_page] Final physical pages to search: {expanded_pages}")
    
    block_index = build_block_index(hypergraph)
    visual_edges = get_visual_contextual_edges(hypergraph)
    
    results = []
    found_anchors = set()  # Dedup by anchor ID
    
    # ============================================================================
    # Search Strategy: Direct match on expanded pages with optional type relaxation
    # ============================================================================
    
    # Exact type match on expanded pages
    print(f"  [tool_find_by_page] Exact type match")
    for edge in visual_edges:
        anchor_block = get_anchor_block_from_edge(edge, block_index)
        if not anchor_block:
            continue
        
        anchor_id = anchor_block['block_id']
        if anchor_id in found_anchors:
            continue
        
        anchor_page = anchor_block.get('page')
        anchor_type = anchor_block.get('type')
        
        # Check if on expanded pages
        if anchor_page not in expanded_pages:
            continue
        
        # Check type match
        if target_type and anchor_type != target_type:
            continue
        
        # Calculate score based on match quality
        score = 0.95
        match_details = []
        
        # Higher score for pages that are in our computed expanded_pages
        if anchor_page in expanded_pages:
            match_details.append(f"matched page {anchor_page}")
            score = 0.95
        else:
            match_details.append(f"out of range page {anchor_page}")
            score = 0.70
        
        if target_type:
            match_details.append(f"exact type {anchor_type}")
        
        # Add pagination info if available
        if page_mapping:
            match_details.append(f"layout={page_mapping['layout']}")
        
        match_reason = f"Page search: {', '.join(match_details)}"
        
        caption_block = get_caption_block_from_edge(edge, block_index, anchor_block['block_id'])
        results.append(QueryHyperedge.create_from_components(
            edge=edge,
            anchor_block=anchor_block,
            caption_block=caption_block,
            related_blocks=get_related_blocks_from_edge(edge, block_index, anchor_block['block_id']),
            score=score,
            match_reason=match_reason,
            tool='tool_find_by_page'
        ))
        found_anchors.add(anchor_id)
    
    # Type relaxation if no results (only if type was specified)
    if not results and target_type:
        print(f"  [tool_find_by_page] Type relaxation")
        for edge in visual_edges:
            anchor_block = get_anchor_block_from_edge(edge, block_index)
            if not anchor_block:
                continue
            
            anchor_id = anchor_block['block_id']
            if anchor_id in found_anchors:
                continue
            
            anchor_page = anchor_block.get('page')
            anchor_type = anchor_block.get('type')
            
            # Check if on expanded pages
            if anchor_page not in expanded_pages:
                continue
            
            # Accept any type
            score = 0.75 if anchor_page in expanded_pages else 0.70
            match_details = [f"page {anchor_page}", f"type relaxed (actual: {anchor_type})"]
            
            if page_mapping:
                match_details.append(f"layout={page_mapping['layout']}")
            
            match_reason = f"Page search: {', '.join(match_details)}"
            
            caption_block = get_caption_block_from_edge(edge, block_index, anchor_block['block_id'])
            results.append(QueryHyperedge.create_from_components(
                edge=edge,
                anchor_block=anchor_block,
                caption_block=caption_block,
                related_blocks=get_related_blocks_from_edge(edge, block_index, anchor_block['block_id']),
                score=score,
                match_reason=match_reason,
                tool='tool_find_by_page'
            ))
            found_anchors.add(anchor_id)
    
    # Sort by score (highest first)
    results.sort(key=lambda x: x.score, reverse=True)
    
    print(f"  [tool_find_by_page] ✓ Returning {len(results)} results")
    return results


def tool_keyword_search(hypergraph: Dict, constraints: Dict, **kwargs) -> List[QueryHyperedge]:
    """"""
    keywords = constraints.get("keywords") or []
    target_type = constraints.get("type")
    coverage_mode = kwargs.get("coverage_mode", False)  #
    
    if not keywords:
        q = kwargs.get("original_query", "")
        if q:
            keywords = q.split()

    if not keywords:
        return []

    block_index = build_block_index(hypergraph)
    visual_edges = filter_edges_by_type(
        get_visual_contextual_edges(hypergraph),
        block_index,
        target_type
    )

    # Strategy B: restrict search to edges whose anchor is in candidate_pages
    visual_strategy = kwargs.get("visual_strategy", "")
    candidate_pages = kwargs.get("candidate_pages", [])
    if visual_strategy == "kw_in_topk" and candidate_pages:
        candidate_pages_set = set(candidate_pages)
        filtered = []
        for edge in visual_edges:
            anchor = get_anchor_block_from_edge(edge, block_index)
            if anchor and anchor.get('page') in candidate_pages_set:
                filtered.append(edge)
        visual_edges = filtered
        print(f"  [kw_in_topk] restricted to {len(visual_edges)} edges in candidate pages {sorted(candidate_pages_set)}")
    
    #
    doc_freq = Counter()
    for edge in visual_edges:
        anchor_block = get_anchor_block_from_edge(edge, block_index)
        if not anchor_block:
            continue
        
        caption_block = get_caption_block_from_edge(edge, block_index, anchor_block['block_id'])
        related_blocks = get_related_blocks_from_edge(edge, block_index, anchor_block['block_id'])
        
        all_text = extract_text_from_blocks([caption_block] + related_blocks).lower()
        
        for kw in keywords:
            if fuzzy_keyword_match(kw, all_text, threshold=0.85):
                doc_freq[kw] += 1
    
    num_docs = len(visual_edges)
    keyword_weights = {}
    for kw in keywords:
        df = doc_freq.get(kw, 0)
        if df == 0:
            keyword_weights[kw] = math.log(num_docs + 1) if num_docs > 0 else 1.0
        else:
            keyword_weights[kw] = math.log((num_docs + 1) / (df + 1))
    
    max_weight = max(keyword_weights.values()) if keyword_weights else 1.0
    if max_weight > 0:
        keyword_weights = {kw: w / max_weight for kw, w in keyword_weights.items()}
    
    #
    structure_weights = {
        'caption': 1.0,
        'context_title': 1.0,  #
        'related_text': 0.75
    }
    
    # candidates: list of (QueryHyperedge, {kw: best_raw_f_score})
    # kw_fscores stored directly to enable greedy selection without string re-parsing
    candidates = []
    for edge in visual_edges:
        anchor_block = get_anchor_block_from_edge(edge, block_index)
        if not anchor_block:
            continue
        
        caption_block = get_caption_block_from_edge(edge, block_index, anchor_block['block_id'])
        related_blocks = get_related_blocks_from_edge(edge, block_index, anchor_block['block_id'])
        
        #
        structured_text = {}
        if caption_block and caption_block.get('text'):
            structured_text['caption'] = caption_block['text']
        if related_blocks:
            structured_text['related_text'] = extract_text_from_blocks(related_blocks)
        
        if not structured_text:
            continue
        
        kw_fscores_map = {}   # {kw: best_raw_fuzzy_score} for greedy coverage selection
        matched_keywords = []
        weighted_score = 0.0
        match_details = []
        
        for kw in keywords:
            best_match_score = 0.0
            best_match_location = None
            best_fuzzy_score = 0.0
            
            for block_type, text in structured_text.items():
                fuzzy_score = fuzzy_keyword_match_score(kw, text, threshold=0.90)
                
                if fuzzy_score > 0:
                    struct_weight = structure_weights.get(block_type, 0.5)
                    match_score = keyword_weights[kw] * fuzzy_score * struct_weight
                    
                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_match_location = block_type
                        best_fuzzy_score = fuzzy_score
            
            kw_fscores_map[kw] = best_fuzzy_score   # 0.0 if no match

            if best_match_score > 0:
                matched_keywords.append(kw)
                weighted_score += best_match_score
                match_details.append(f"{kw}@{best_match_location}(f={best_fuzzy_score:.2f})")
        
        if matched_keywords:
            #
            if coverage_mode:
                #
                #
                coverage_ratio = len(matched_keywords) / len(keywords)
                score = coverage_ratio * 0.9 + (weighted_score / sum(keyword_weights[kw] for kw in keywords if kw in matched_keywords)) * 0.1
                match_reason = f"Coverage: {len(matched_keywords)}/{len(keywords)} keywords: {', '.join(match_details)}"
            else:
                #
                max_possible = sum(keyword_weights[kw] * 1.0 for kw in keywords)
                score = weighted_score / max_possible if max_possible > 0 else 0
                match_reason = f"Matched {len(matched_keywords)}/{len(keywords)} keywords: {', '.join(match_details)}"
            
            candidates.append(QueryHyperedge.create_from_components(
                edge=edge,
                anchor_block=anchor_block,
                caption_block=caption_block,
                related_blocks=related_blocks,
                score=score,
                match_reason=match_reason,
                matched_keywords=matched_keywords,
                tool='tool_keyword_search',
                extra_meta={'kw_fscores': kw_fscores_map}  # per-keyword f-scores for optimizer
            ))

    if candidates:
        # Return ALL candidates IDF-sorted — full pool for lexicographic_greedy_optimizer.
        # The optimizer in retrieval_router selects/reranks; no truncation here.
        candidates.sort(key=lambda x: -x.score)
        top_results = candidates
        print(f"  [tool_keyword_search] returning {len(top_results)} IDF-sorted candidates (full pool)")
    else:
        top_results = []

    #
    if not top_results:
        top_results = _keyword_visual_fallback(
            hypergraph=hypergraph,
            keywords=keywords,
            original_query=kwargs.get("original_query", ""),
            reasoner=kwargs.get("reasoner"),
            page_image_dir=kwargs.get("page_image_dir", ""),
            clean_doc_id=kwargs.get("clean_doc_id", ""),
            block_index=block_index,
            visual_edges=visual_edges,
        )

    return top_results


def _keyword_visual_fallback(
    hypergraph: Dict,
    keywords: List[str],
    original_query: str,
    reasoner,
    page_image_dir: str,
    clean_doc_id: str,
    block_index: Dict,
    visual_edges: List[Dict],
) -> List[QueryHyperedge]:
    """"""
    if not reasoner or not page_image_dir or not clean_doc_id:
        return []

    import os

    #
    page_to_edges: Dict[int, List] = {}
    for edge in visual_edges:
        anchor = get_anchor_block_from_edge(edge, block_index)
        if not anchor:
            continue
        page = anchor.get("page")
        if page is None:
            continue
        page_to_edges.setdefault(page, []).append(edge)

    if not page_to_edges:
        return []

    #
    candidate_pages = sorted(page_to_edges.keys())[:15]

    #
    page_images = []
    for p in candidate_pages:
        img_path = os.path.join(page_image_dir, f"{clean_doc_id}_{p}.png")
        if os.path.exists(img_path):
            page_images.append((p, img_path))

    if not page_images:
        return []

    #
    kw_str = ", ".join(f'"{k}"' for k in keywords) if keywords else f'"{original_query}"'
    prompt = (
        f"You are given {len(page_images)} document page image(s). "
        f"The user is looking for content related to: {kw_str}.\n"
        f"For each image (numbered 1 to {len(page_images)}), reply with exactly one line: "
        f"'<image_number>: yes' if the page visually contains relevant content, or "
        f"'<image_number>: no' otherwise. Be concise."
    )

    images = [img for _, img in page_images]
    try:
        vlm_response, _ = reasoner.predict(prompt, images=images)
    except Exception as e:
        print(f"  [_keyword_visual_fallback] VLM call failed: {e}")
        return []

    #
    matched_pages = []
    for line in vlm_response.splitlines():
        line = line.strip().lower()
        m = re.match(r"(\d+)\s*:\s*yes", line)
        if m:
            idx = int(m.group(1)) - 1  # 1-based → 0-based
            if 0 <= idx < len(page_images):
                matched_pages.append(page_images[idx][0])

    if not matched_pages:
        return []

    print(f"  [_keyword_visual_fallback] VLM matched pages: {matched_pages}")

    #
    fallback_results = []
    for page in matched_pages:
        for edge in page_to_edges.get(page, [])[:2]:
            anchor = get_anchor_block_from_edge(edge, block_index)
            if not anchor:
                continue
            caption = get_caption_block_from_edge(edge, block_index, anchor["block_id"])
            related = get_related_blocks_from_edge(edge, block_index, anchor["block_id"])
            fallback_results.append(QueryHyperedge.create_from_components(
                edge=edge,
                anchor_block=anchor,
                caption_block=caption,
                related_blocks=related,
                score=0.6,
                match_reason=f"Visual fallback: page {page} matched query '{original_query[:40]}'",
                tool='tool_keyword_search'
            ))

    return fallback_results


def tool_visual_search(
    hypergraph: Dict,
    constraints: Dict,
    candidate_pages: Optional[List[int]] = None,
    **kwargs
) -> List[QueryHyperedge]:
    """Tool: Generate page-level nodes from supplied candidate pages.

    Consumes 0-based page indices and emits one QueryHyperedge *page node* per
    candidate page. Page nodes carry no bound hyperedge or anchor block; they
    are tagged with ``is_page_node=True`` in extra_meta so that the
    optimizer can correctly scope their constraint contribution to page_range
    satisfaction only.

    Score decays linearly with rank: rank-0 → 1.0, rank-(n-1) → ~0.1.
    """
    if not candidate_pages:
        return []

    n = len(candidate_pages)
    results: List[QueryHyperedge] = []

    for rank, page in enumerate(candidate_pages):
        score = max(0.1, 1.0 - rank / n)
        results.append(QueryHyperedge.create_from_components(
            score=score,
            match_reason=f"Visual page retrieval: page {page} (rank #{rank + 1})",
            tool='tool_visual_search',
            extra_meta={
                'source_page': page,
                'is_page_node': True,
            }
        ))

    print(f"  [tool_visual_search] {len(results)} page nodes from "
          f"{len(candidate_pages)} coarse page candidates")
    return results


def tool_spatial_locator(
    hypergraph: Dict, 
    constraints: Dict, 
    reasoner=None,
    page_image_dir: str = "",
    clean_doc_id: str = "",
    **kwargs
) -> List[QueryHyperedge]:
    """Tool: Locate elements by spatial hints (top, bottom, right, top-right, etc.)."""
    spatial_hint = constraints.get("spatial_hint")
    target_type = constraints.get("type")
    if not spatial_hint:
        return []
    
    block_index = build_block_index(hypergraph)
    visual_edges = filter_edges_by_type(
        get_visual_contextual_edges(hypergraph),
        block_index,
        target_type
    )
    
    spatial_hint_lower = spatial_hint.lower()
    results = []
    
    # 1. Check if referring to another element (Relative Spatial)
    reference_match = re.search(r'(below|above|under|over|next to|beside)\s+(figure|table|chart)\s*(\d+)', spatial_hint_lower)
    
    if reference_match:
        # Spatial reference to another element (legacy logic)
        relation = reference_match.group(1)
        ref_type = reference_match.group(2)
        ref_id = reference_match.group(3)
        
        # Find the reference element
        ref_anchor = None
        for edge in visual_edges:
            anchor = get_anchor_block_from_edge(edge, block_index)
            if not anchor: continue
            caption = get_caption_block_from_edge(edge, block_index, anchor['block_id'])
            if caption and match_unit_id(caption.get('text', ''), ref_id, ref_type):
                ref_anchor = anchor
                break
        
        if ref_anchor:
            ref_page = ref_anchor.get('page')
            ref_bbox = ref_anchor.get('bbox', [])
            
            if ref_bbox and len(ref_bbox) == 4:
                ref_y_center = (ref_bbox[1] + ref_bbox[3]) / 2.0
                
                for edge in visual_edges:
                    anchor = get_anchor_block_from_edge(edge, block_index)
                    if not anchor or anchor['block_id'] == ref_anchor['block_id']: continue
                    if anchor.get('page') != ref_page: continue
                    
                    bbox = anchor.get('bbox', [])
                    if len(bbox) < 4: continue
                    
                    y_center = (bbox[1] + bbox[3]) / 2.0
                    matched = False
                    if relation in ['below', 'under'] and y_center > ref_y_center: matched = True
                    elif relation in ['above', 'over'] and y_center < ref_y_center: matched = True
                    
                    if matched:
                        caption = get_caption_block_from_edge(edge, block_index, anchor['block_id'])
                        related = get_related_blocks_from_edge(edge, block_index, anchor['block_id'])
                        results.append(QueryHyperedge.create_from_components(
                            edge=edge, anchor_block=anchor, caption_block=caption, related_blocks=related,
                            score=1.0, match_reason=f"Spatial: {relation} {ref_type} {ref_id}", tool='tool_spatial_locator'
                        ))
        return results

    # 2. Absolute Spatial Logic (Top, Bottom, Top-Right, etc.)
    # -------------------------------------------------------------------------
    
    # A. Parse Target Pages
    page_range_1based = constraints.get("page_range")  # Now 1-based from intent
    target_pages = None  # Will be 0-based physical indices
    
    if page_range_1based:
        # Convert 1-based user pages to 0-based physical indices
        target_pages = _convert_page_range_to_physical(
            page_range_1based, reasoner, page_image_dir, clean_doc_id
        )
        print(f"  [tool_spatial_locator] Converted user pages {page_range_1based} (1-based) to physical indices {target_pages}")
            
    # Implicit Page Inference if explicit page is missing
    if not target_pages and constraints.get("keywords"):
         inferred_pages = _find_pages_by_keywords(block_index, constraints["keywords"])
         if inferred_pages:
             print(f"  Implicitly located pages {inferred_pages} from keywords: {constraints['keywords']}")
             target_pages = inferred_pages
             
    # B. Parse Spatial Directions (Composite)
    directions = set()
    mappings = {
        'top': ['top', 'upper', 'above'],
        'bottom': ['bottom', 'lower', 'below'],
        'left': ['left'],
        'right': ['right'],
        'center': ['center', 'middle']
    }
    for pos, kws in mappings.items():
        if any(kw in spatial_hint_lower for kw in kws):
            directions.add(pos)
            
    if not directions:
        return []

    # C. Pre-calculate Page Dimensions (Width/Height)
    page_dims = {} # page_idx -> (max_w, max_h)
    for b in block_index.values():
        p = b.get('page')
        bbox = b.get('bbox', [])
        if p is not None and len(bbox) == 4:
            prev_w, prev_h = page_dims.get(p, (0, 0))
            page_dims[p] = (max(prev_w, bbox[2]), max(prev_h, bbox[3]))
            
    # D. Filter Candidate Edges
    candidate_edges = []
    
    for edge in visual_edges:
        anchor = get_anchor_block_from_edge(edge, block_index)
        if not anchor: continue
        
        # Page Filter (0-based)
        anchor_page_0based = anchor.get('page', 0)
        if target_pages and anchor_page_0based not in target_pages:
            continue
                 
        bbox = anchor.get('bbox', [])
        if len(bbox) < 4: continue
        
        candidate_edges.append((edge, anchor))
        
    if not candidate_edges:
        return []

    # E. Score and Sort
    # Objective: Find the block that best maximizes the spatial constraints.
    # We calculate a 'spatial_penalty' (lower is better).
    
    scored_items = []
    
    for edge, anchor in candidate_edges:
        p = anchor.get('page', 0)
        max_w, max_h = page_dims.get(p, (1000, 1000))
        if max_w < 100: max_w = 1000
        if max_h < 100: max_h = 1000
        
        bbox = anchor.get('bbox')
        # Center points
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        
        # Norm coords (0.0 to 1.0)
        nx = cx / max_w if max_w > 0 else 0.5
        ny = cy / max_h if max_h > 0 else 0.5
        
        # Clamp
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))
        
        penalty = 0.0
        active_constraints = 0
        
        # Vertical Penalties
        if 'top' in directions:
            # 0.0 is best. 
            # Bonus: if element *starts* really high (bbox[1] near 0), reduce penalty for large items
            start_y_norm = bbox[1] / max_h
            metric = min(ny, start_y_norm + 0.1) # Effective Y for "top" logic
            penalty += metric
            active_constraints += 1
        elif 'bottom' in directions:
            # 1.0 is best
            penalty += (1.0 - ny)
            active_constraints += 1
            
        # Horizontal Penalties
        if 'left' in directions:
            # 0.0 is best
            penalty += nx
            active_constraints += 1
        elif 'right' in directions:
            # 1.0 is best
            penalty += (1.0 - nx)
            active_constraints += 1
            
        # Center Penalty (Distance from 0.5, 0.5)
        if 'center' in directions:
            dist = math.sqrt((nx - 0.5)**2 + (ny - 0.5)**2)
            penalty += dist * 2 # Weight center strongly
            active_constraints += 1
            
        final_score = 1.0 - (penalty / active_constraints if active_constraints > 0 else 0)
        
        scored_items.append({
            'edge': edge,
            'anchor': anchor,
            'score': final_score,
            'coords': (nx, ny)
        })
        
    # Sort descending by score
    scored_items.sort(key=lambda x: x['score'], reverse=True)
    
    # Return Top 1 (as per user request "sorted_edges[0]") or maybe Top N?
    # User said "return [final_unit]" (list of 1)
    
    best_items = scored_items[:1] # Strict top 1 for spatial locator? Or maybe 3?
    # Let's return up to 3 for robustness, but score differentiates them.
    # Actually user strictness implies we should be confident.
    
    results = []
    for item in best_items:
        edge = item['edge']
        anchor = item['anchor']
        caption = get_caption_block_from_edge(edge, block_index, anchor['block_id'])
        related = get_related_blocks_from_edge(edge, block_index, anchor['block_id'])
        
        results.append(QueryHyperedge.create_from_components(
            edge=edge,
            anchor_block=anchor,
            caption_block=caption,
            related_blocks=related,
            score=item['score'],
            match_reason=f"Spatial: {spatial_hint} (score={item['score']:.2f})",
            tool='tool_spatial_locator'
        ))
        
    return results


def tool_multi_keyword_fetch(hypergraph: Dict, constraints: Dict, **kwargs) -> List[QueryHyperedge]:
    """Tool: Fetch multiple specific units for MULTI_UNIT comparison/cross-reasoning."""
    multi_id = constraints.get("id")
    if not multi_id:
        return []
    
    # Parse multiple IDs (e.g., "Figure 1, Table 5")
    id_pattern = r'(figure|table|chart|map|graph)\s*(\d+)'
    matches = re.findall(id_pattern, multi_id.lower())
    
    if not matches:
        return []
    
    # Reuse keyword search logic to find each unit
    results = []
    found_ids = set()
    
    # We construct a synthetic generic constraints dict to search for all target types
    # Since we are looking for specific IDs, we can ignore the generic 'type' constraint for now
    # or loop through unique types found in the query
    
    # Better approach: Iterate matches and call find_by_id for each
    for unit_type, unit_id in matches:
        id_key = f"{unit_type} {unit_id}"
        if id_key in found_ids:
            continue
            
        # Construct specific constraints for this unit
        sub_constraints = {
            "id": f"{unit_type} {unit_id}",
            "type": unit_type, # figure, table, etc.
            # inherit other constraints if necessary, but ID is usually specific enough
        }
        
        # Call tool_find_by_id
        # Note: tool_find_by_id returns a list of matching hyperedges
        unit_results = tool_find_by_id(hypergraph, sub_constraints, **kwargs)
        
        for res in unit_results:
            # Modify match reason to indicate multi-fetch context
            res.match_reason = f"Multi-fetch: {res.match_reason}"
            res.tool = 'multi_keyword_fetch'
            results.append(res)
            
        if unit_results:
            found_ids.add(id_key)
            
    return results


def tool_enumerate(
    hypergraph: Dict,
    constraints: Dict,
    reasoner=None,
    original_query: str = "",
    page_image_dir: str = "",
    clean_doc_id: str = "",
    **kwargs
) -> List[QueryHyperedge]:
    """Tool: Enumerate/list all items matching criteria (returns structured list)."""
    from hyperedge.prompts import format_enumerate_extraction_prompt

    block_index = build_block_index(hypergraph)
    visual_edges = get_visual_contextual_edges(hypergraph)
    
    keywords = constraints.get("keywords", [])
    target_type = constraints.get("type")
    section_title = constraints.get("section_title")
    page_range = constraints.get("page_range")  # Now 1-based user page numbers

    is_visual_mode = (target_type is None) or is_visual_type(target_type)

    # ========================================================================
    # UNIFIED SEMANTIC PRE-FILTERING LOGIC
    # ========================================================================
    candidate_edges = []
    candidate_pages = []
    has_constraints = bool(section_title or page_range)
    
    if is_visual_mode:
        # Visual mode: enumerate visual elements
        print(f"  → Visual mode: enumerating visual elements")
        
        if has_constraints:
            # Branch 1: is_visual_mode=True AND has page/section constraints
            # → Use filter functions
            if section_title:
                all_visual_edges = list(visual_edges)
                candidate_edges = filter_edges_by_section_title(all_visual_edges, hypergraph, block_index, section_title)
                print(f"     ✓ Section filter: {len(candidate_edges)} edges")
            elif page_range:
                # Convert 1-based user pages to 0-based physical indices
                physical_indices = _convert_page_range_to_physical(
                    page_range, reasoner, page_image_dir, clean_doc_id
                )
                print(f"     → Converted user pages {page_range} (1-based) to physical indices {physical_indices}")
                candidate_edges = filter_edges_by_page_range(list(visual_edges), block_index, physical_indices)
                print(f"     ✓ Page filter: {len(candidate_edges)} edges")
        else:
            # Branch 2: is_visual_mode=True BUT no page/section constraints
            # → Check document size, use keyword pre-filter if >40 anchors
            total_anchors = len(visual_edges)
            if total_anchors > 40:
                if keywords:
                    print(f"     → Large doc ({total_anchors} anchors), applying keyword pre-filter")
                    keyword_matched = []
                    for edge in visual_edges:
                        anchor = get_anchor_block_from_edge(edge, block_index)
                        if not anchor:
                            continue
                        caption = get_caption_block_from_edge(edge, block_index, anchor['block_id'])
                        related = get_related_blocks_from_edge(edge, block_index, anchor['block_id'])
                        text = extract_text_from_blocks([caption] + related).lower()
                        if any(fuzzy_keyword_match(kw, text, threshold=0.7) for kw in keywords):
                            keyword_matched.append(edge)
                            if len(keyword_matched) >= 40:
                                break
                    candidate_edges = keyword_matched
                    print(f"     ✓ Keyword pre-filter: {len(candidate_edges)} edges")
                else:
                    # No keywords: pass all VU up to cap instead of returning empty
                    print(f"     → No keywords for pre-filter, using all anchors up to cap=40")
                    candidate_edges = list(visual_edges)[:40]
            else:
                print(f"     → Small doc ({total_anchors} anchors), using all")
                candidate_edges = list(visual_edges)
        
        # Apply type filter if specified
        if candidate_edges and target_type:
            candidate_edges = filter_edges_by_type(candidate_edges, block_index, target_type)
            print(f"     ✓ Type filter: {len(candidate_edges)} edges")
    
    else:
        # Text mode: enumerate pages/sections
        print(f"  → Text mode: enumerating pages/sections")
        
        if has_constraints:
            # Branch 3: is_visual_mode=False BUT has page/section constraints
            # → Use hypergraph containment to find section ranges → get source pages
            if section_title:
                candidate_pages = get_pages_from_section_ranges(hypergraph, section_title)
                print(f"     ✓ Section ranges: {len(candidate_pages)} pages")
            elif page_range:
                # Convert 1-based user pages to 0-based physical indices
                physical_indices = _convert_page_range_to_physical(
                    page_range, reasoner, page_image_dir, clean_doc_id
                )
                print(f"     → Converted user pages {page_range} (1-based) to physical indices {physical_indices}")
                candidate_pages = physical_indices
                print(f"     ✓ Page range: {len(candidate_pages)} pages")
        else:
            # Branch 4: No matching conditions
            # Return empty so the pipeline can use coarse page evidence.
            print(f"     -> No constraints in text mode, returning empty (coarse page fallback)")
            candidate_pages = []
    
    # If no candidates are found, return a sentinel for coarse page fallback.
    if not candidate_edges and not candidate_pages:
        print(f"  No candidates found, returning empty (will use coarse page evidence)")
        return [QueryHyperedge.create_from_components(
            score=0.0,
            match_reason="Enumerate: no candidates (using coarse page evidence)",
            tool='tool_enumerate',
            enumerated_items=[],
            extra_meta={
                'total_count': 0,
                'is_enumerate_result': True,
                'fallback_to_coarse_pages': True
            }
        )]

    # 2) If no VLM reasoner, fallback to text-based enumeration (legacy)
    if not reasoner or not original_query or not clean_doc_id:
        print(f"  ⚠️  No VLM/query/doc_id, using text-only enumeration")
        
        enumerated_items = []
        
        if is_visual_mode and candidate_edges:
            # Text-based enumeration for visual elements
            for edge in candidate_edges:
                anchor = get_anchor_block_from_edge(edge, block_index)
                if not anchor:
                    continue
                caption = get_caption_block_from_edge(edge, block_index, anchor['block_id'])
                related = get_related_blocks_from_edge(edge, block_index, anchor['block_id'])
                text_combined = extract_text_from_blocks([caption] + related)
                
                if keywords:
                    matches = all(fuzzy_keyword_match(kw, text_combined, threshold=0.8) for kw in keywords)
                    if not matches:
                        continue
                
                item_id = None
                if caption:
                    id_match = re.search(r'(figure|table|chart)\s*(\d+)', caption.get('text', '').lower())
                    if id_match:
                        item_id = f"{id_match.group(1).title()} {id_match.group(2)}"
                
                enumerated_items.append({
                    'item_id': item_id or f"Item {len(enumerated_items) + 1}",
                    'type': anchor.get('type'),
                    'page': anchor.get('page', 0) + 1,
                    'text_preview': text_combined[:100]
                })
        else:
            # Text-based enumeration for page-level (text mode)
            # Get pages from section or all pages
            candidate_pages = get_pages_from_section_ranges(
                hypergraph, constraints.get("section_title")
            )
            if not candidate_pages:
                raw_pages = hypergraph.get('pages', []) or []
                candidate_pages = sorted(list(set(p.get('page_num') for p in raw_pages if isinstance(p, dict) and isinstance(p.get('page_num'), int))))
            
            page_filter_1based = constraints.get("page_range")
            if page_filter_1based:
                # Convert 1-based to 0-based for filtering (simple fallback without VLM)
                page_filter_0based = [p - 1 for p in page_filter_1based if p > 0]
                candidate_pages = [p for p in candidate_pages if p in page_filter_0based] or candidate_pages
            
            # Extract text blocks from pages and enumerate based on keywords
            for page_idx in candidate_pages:
                page_blocks = [b for b in block_index.values() if b.get('page') == page_idx]
                page_text = extract_text_from_blocks(page_blocks)
                
                if keywords:
                    # Find mentions of keywords in page text
                    for kw in keywords:
                        if fuzzy_keyword_match(kw, page_text, threshold=0.8):
                            enumerated_items.append({
                                'item': kw,
                                'page': page_idx + 1,
                                'text_preview': page_text[:150]
                            })
                            break
        
        return [QueryHyperedge.create_from_components(
            score=1.0,
            match_reason=f"Enumerated {len(enumerated_items)} items (text-only fallback, mode={'visual' if is_visual_mode else 'page'})",
            tool='tool_enumerate',
            enumerated_items=enumerated_items,
            extra_meta={
                'total_count': len(enumerated_items),
                'is_enumerate_result': True,
                'fallback_used': True,
                'mode': 'visual' if is_visual_mode else 'page'
            }
        )]

    # 3) VLM-based enumerate extraction
    candidate_images = []
    candidate_meta = []
    caption_notes = []

    if is_visual_mode:
        for edge in candidate_edges:
            anchor = get_anchor_block_from_edge(edge, block_index)
            if not anchor:
                continue
            page_idx = anchor.get('page')
            if page_idx is None:
                continue
            img_path = f"{page_image_dir}/{clean_doc_id}_{page_idx}.png"
            candidate_images.append(img_path)
            caption = get_caption_block_from_edge(edge, block_index, anchor['block_id'])
            related = get_related_blocks_from_edge(edge, block_index, anchor['block_id'])
            caption_text = extract_text_from_blocks([caption] + related)
            caption_notes.append(caption_text[:180])
            candidate_meta.append({
                'edge': edge,
                'anchor': anchor,
                'caption': caption,
                'related': related
            })
    else:
        # Text mode: prepare page images from candidate_pages
        # Note: candidate_pages should already be populated from pre-filtering logic above
        if candidate_pages:
            for page_idx in candidate_pages:
                img_path = f"{page_image_dir}/{clean_doc_id}_{page_idx}.png"
                candidate_images.append(img_path)
                # Build lightweight page notes from blocks
                page_blocks = [b for b in block_index.values() if b.get('page') == page_idx]
                page_text = extract_text_from_blocks(page_blocks)
                caption_notes.append(page_text[:180])
            print(f"     → Prepared {len(candidate_images)} page images for VLM")
        else:
            print(f"     -> No candidate pages, skipping VLM (will use coarse page evidence)")

    if not candidate_images:
        return [QueryHyperedge.create_from_components(
            score=1.0,
            match_reason="Enumerate: no candidate images",
            tool='tool_enumerate',
            enumerated_items=[],
            extra_meta={
                'total_count': 0,
                'is_enumerate_result': True
            }
        )]

    prompt = format_enumerate_extraction_prompt(original_query)
    if target_type and is_visual_type(target_type):
        prompt += f"\n\nType hint: {target_type}"
    if caption_notes:
        caption_lines = [f"{i+1}) {t}" for i, t in enumerate(caption_notes)]
        prompt += "\n\nCandidate notes (image_index -> text snippet):\n" + "\n".join(caption_lines)

    vlm_output = None
    json_parse_success = False
    per_image = []
    extracted_items = []
    matched_indices = []

    try:
        vlm_output, _ = reasoner.predict(prompt, images=candidate_images)
        parsed, _, parse_error = _load_vlm_json_object(vlm_output)
        if parsed is not None:
            per_image = parsed.get("per_image", []) or []
            json_parse_success = True
            for i, item in enumerate(per_image):
                if isinstance(item, dict) and item.get("match"):
                    matched_indices.append(i)
                    for val in item.get("items", []) or []:
                        if val not in extracted_items:
                            extracted_items.append(val)
        else:
            recovered_indices, recovered_per_image = _recover_per_image_matches(vlm_output)
            if recovered_per_image:
                matched_indices = recovered_indices
                per_image = recovered_per_image
                print(
                    f"  ⚠️  JSON parse error in enumerate ({parse_error}); "
                    f"recovered {len(matched_indices)} matched indices without item strings"
                )
            else:
                print(f"  ⚠️  JSON parse error in enumerate ({parse_error})")
    except Exception as e:
        print(f"  ⚠️  Exception in enumerate VLM call: {e}")
        json_parse_success = False

    # Map matched indices back to hyperedges or pages
    matched_results = []
    final_edges = []
    matched_pages = []
    
    # Normal processing for visual/text content enumeration
    for idx in matched_indices:
        if is_visual_mode and idx < len(candidate_meta):
            meta = candidate_meta[idx]
            final_edges.append(meta['edge'])
            matched_results.append({
                'query_hyperedge': meta['edge'],
                'anchor_block': meta['anchor'],
                'caption_block': meta['caption'],
                'related_blocks': meta['related']
            })
        elif not is_visual_mode and idx < len(candidate_pages):
            matched_pages.append(candidate_pages[idx])

    if is_visual_mode:
        match_reason = f"Enumerated {len(extracted_items)} items from {len(final_edges)} matched units (VLM)"
    else:
        match_reason = f"Enumerated {len(extracted_items)} items from {len(matched_pages)} matched pages (VLM)"
    
    # If VLM parsing failed or returned no items, try text-based fallback
    if not json_parse_success or len(extracted_items) == 0:
        print(f"  ⚠️  VLM enumeration failed or empty, falling back to text-based extraction")
        fallback_items = []
        
        if is_visual_mode and candidate_meta:
            # Extract from matched visual groups
            for meta in candidate_meta:
                caption = meta['caption']
                related = meta['related']
                text_combined = extract_text_from_blocks([caption] + related)
                
                # Extract items matching keywords
                for kw in keywords:
                    if fuzzy_keyword_match(kw, text_combined, threshold=0.75):
                        if kw not in fallback_items:
                            fallback_items.append(kw)
        elif not is_visual_mode and candidate_pages:
            # Extract from pages
            for page_idx in candidate_pages:
                page_blocks = [b for b in block_index.values() if b.get('page') == page_idx]
                page_text = extract_text_from_blocks(page_blocks)
                
                for kw in keywords:
                    if fuzzy_keyword_match(kw, page_text, threshold=0.75):
                        if kw not in fallback_items:
                            fallback_items.append(kw)
        
        if fallback_items:
            extracted_items = fallback_items
            match_reason = f"Enumerated {len(extracted_items)} items (text fallback after VLM failure)"

    return [QueryHyperedge.create_from_components(
        score=1.0,
        match_reason=match_reason,
        tool='tool_enumerate',
        enumerated_items=extracted_items,
        extra_meta={
            'total_count': len(extracted_items),
            'is_enumerate_result': True,
            'json_parse_success': json_parse_success,
            'vlm_output': vlm_output,
            'per_image': per_image,
            'matched_results': matched_results,
            'matched_unit_count': len(final_edges),
            'matched_pages': matched_pages
        }
    )]


def tool_global_count(
    hypergraph: Dict, 
    constraints: Dict, 
    reasoner=None,
    original_query: str = "",
    page_image_dir: str = "",
    clean_doc_id: str = "",
    **kwargs
) -> List[QueryHyperedge]:
    """Tool: Global count of elements (visual or textual) with Stage 1 VLM validation (JSON matching only)."""
    import json
    import re
    from hyperedge.prompts import format_count_matching_prompt
    
    block_index = build_block_index(hypergraph)
    visual_edges = get_visual_contextual_edges(hypergraph)
    
    keywords = constraints.get("keywords", [])
    target_type = constraints.get("type")
    section_title = constraints.get("section_title")
    page_range = constraints.get("page_range")  # Now 1-based user page numbers

    # ── FAST PATH A: PAGE_COUNT ───────────────────────────────────────────────
    # Fires only when query asks for the total page count of the document with
    # no other constraints. Answers from file count — no VLM needed.
    # Guards: not keywords (content-specific page counts have keywords),
    #         not section_title, not page_range, not target_type.
    _PAGE_COUNT_RE = re.compile(
        r'how many pages (does|do|are|in|did|has)|'
        r'total (number of )?pages|'
        r'pages (does|do) (the )?(document|report|paper|file|newspaper) (have|contain|consist)',
        re.IGNORECASE
    )
    _PAGE_COUNT_GENERIC_KW = {'pages', 'total', 'report', 'document', 'paper', 'file', 'newspaper'}
    _is_page_count_query = (
        bool(_PAGE_COUNT_RE.search(original_query))
        and all(kw.lower() in _PAGE_COUNT_GENERIC_KW for kw in keywords)  # block content-specific kw
        and not section_title
        and not page_range
        and (not target_type or target_type == "text")
        and bool(page_image_dir)
        and bool(clean_doc_id)
    )
    if _is_page_count_query:
        import glob as _glob_pc
        png_files = _glob_pc.glob(f"{page_image_dir}/{clean_doc_id}_*.png")
        total_pages = len(png_files)
        print(f"  ⚡ [FAST PATH A: PAGE_COUNT] '{original_query[:70]}' → {total_pages} pages")
        return [QueryHyperedge.create_from_components(
            score=1.0,
            match_reason=f"Fast page count: {total_pages} pages (file count, no VLM)",
            tool='tool_global_count',
            count_result=total_pages,
            extra_meta={
                'is_count_result': True,
                'count_result': total_pages,
                'fast_path': 'PAGE_COUNT',
                'total_images': total_pages,
            }
        )]

    # ── FAST PATH X: ARITHMETIC_LOOKUP ───────────────────────────────────────
    # Fires when the query asks for an arithmetic/aggregated value from a table
    # (sum, combined percentage, total expenditure, cross-category total, etc.)
    # rather than counting discrete visual elements. These queries arrive with
    # intent=count but the VLM counting stage is wrong because the answer is a
    # numeric value read from table cells, not a count of elements. Returning []
    # lets the router's visual_search fallback use coarse page candidates (which
    # contain the relevant table) so the VLM can read the value directly.
    #
    # Two pattern groups:
    #  A. Explicit arithmetic operators: calculate, sum the, total expenditure,
    #     combined percentage, average salary, how much…
    #  B. Cross-category aggregation: "X and Y combined", "how many X in total",
    #     "total number of X and Y" — summing specific named values, not counting.
    # Guard (_STRUCT_COUNT_RE): skip "total number of tables/figures/pages/etc."
    # which ARE genuine visual counts and should stay in the normal GC path.
    _ARITH_RE = re.compile(
        r'^\s*calculate\b|'                          # "Calculate total…"
        r'\bsum\s+the\b|'                           # "Sum the total…"
        r'\bhow\s+much\b|'                          # "How much X was Y" (value, not count)
        r'\btotal\s+sum\b|'                         # "total sum of…"
        r'\bcombined\s+(?:total|value|percentage|proportion|amount|cost|revenue|expenditure)\b|'
        r'\btotal\s+(?:expenditure|revenues?|compensation|depreciation|charge|profit|loss|'
        r'costs?\b|salary|budget|income|receipts|assets|liabilities|equity|gains|savings|'
        r'value\b|amount\b|reportable|quantity|consumption|production|capacity|lengths?|'
        r'audit\s+fees?|operating\s+revenue|area\b|score\b)|'
        r'\baverage\s+(?:\w+\s+)*(?:percentage|expenditure|cost|price|rate|salary|income|'
        r'value|revenue|amount|pass|marks?|budget)|'
        # Cross-category aggregation: summing named values rather than counting elements.
        # "X and Y combined", "how many X and Y in total", "total number of X and Y".
        # Deliberate design choices vs. the old version:
        #   - "and ... combined" stays broad — unambiguous aggregation signal
        #   - "how many" requires "and" BETWEEN "how many" and "in total" so that
        #     "how many words in total" / "how many pages in total" are NOT caught
        #   - "total number of X and Y" is guarded by _STRUCT_COUNT_RE below
        r'\band\b.{1,80}\bcombined\b|'                              # "A and B combined"
        r'\b(?:both\s+categories|from\s+both)\b|'                   # "(both categories)"
        r'\bhow\s+many\b.{3,80}\band\b.{3,60}\b(?:in\s+total|altogether)\b|'  # "how many X and Y in total"
        r'\btotal\s+number\s+of\b.{5,120}\band\b',                 # "total number of X and Y"
        re.IGNORECASE | re.DOTALL
    )
    # Guard A: skip "total number of <structural noun>" — those ARE true element counts.
    # Expanded to include scientific/counting nouns seen in DocBench and MMLongBench.
    _STRUCT_COUNT_RE = re.compile(
        r'\btotal\s+number\s+of\s+(?:tables?|figures?|images?|charts?|diagrams?|'
        r'pages?|sections?|chapters?|items?|papers?|exhibits?|occurrences?|'
        r'words?|footnotes?|samples?|instances?|data\s*points?|records?|'
        r'features?|tokens?|sentences?|paragraphs?|questions?|answers?|'
        r'references?|citations?|equations?|appendi(?:x|ces))\b',
        re.IGNORECASE
    )
    # Guard B: "how many <structural noun>" queries (e.g. "how many tables and figures
    # in total") are document-structure counts, not arithmetic lookups.
    _STRUCT_HOWMANY_RE = re.compile(
        r'\bhow\s+many\s+(?:tables?|figures?|images?|charts?|diagrams?|pages?|'
        r'sections?|chapters?|words?|footnotes?|references?|equations?)\b',
        re.IGNORECASE
    )
    _is_arithmetic_lookup = (
        bool(_ARITH_RE.search(original_query))
        and not _STRUCT_COUNT_RE.search(original_query)
        and not _STRUCT_HOWMANY_RE.search(original_query)
    )
    # ── FAST PATH B: OCCURRENCE_COUNT ────────────────────────────────────────
    # Placed BEFORE FAST PATH X so that "how many times does 'total salary'
    # appear?" is handled by OCR text scan rather than accidentally caught by
    # ARITH_RE (which matches terms like "total … salary").
    # Guards: keywords must be present, not target_type (visual occurrences
    #         should be routed via target_type="figure"), not page_range
    #         (page-scoped queries must not scan all pages).
    _OCCURRENCE_RE = re.compile(r'\bhow many times?\b', re.IGNORECASE)
    _is_occurrence_query = (
        bool(_OCCURRENCE_RE.search(original_query))
        and bool(keywords)
        and (not target_type or target_type == "text")
        and not page_range
        and bool(page_image_dir)
        and bool(clean_doc_id)
    )
    if _is_occurrence_query:
        import glob as _glob_oc
        import os as _os_oc

        def _normalize_ocr(text: str) -> str:
            text = text.lower()
            text = re.sub(r'[-–—]', ' ', text)  # hyphens/dashes → space
            text = re.sub(r'\s+', ' ', text)
            return text

        txt_files = sorted(_glob_oc.glob(f"{page_image_dir}/{clean_doc_id}_*.txt"))
        total_occurrences = 0
        matched_pages_1based = []
        for txt_path in txt_files:
            try:
                with open(txt_path, 'r', encoding='utf-8', errors='ignore') as _f:
                    content = _normalize_ocr(_f.read())
                page_hits = sum(content.count(_normalize_ocr(kw)) for kw in keywords)
                if page_hits > 0:
                    fname = _os_oc.path.basename(txt_path)
                    page_idx_str = fname.replace(f"{clean_doc_id}_", "").replace(".txt", "")
                    try:
                        matched_pages_1based.append(int(page_idx_str) + 1)
                    except ValueError:
                        pass
                    total_occurrences += page_hits
            except Exception:
                pass
        print(f"  ⚡ [FAST PATH B: OCCURRENCE_COUNT] '{original_query[:70]}' "
              f"keywords={keywords} → {total_occurrences} hits on {len(matched_pages_1based)} pages "
              f"(scanned {len(txt_files)} txt files)")
        return [QueryHyperedge.create_from_components(
            score=1.0,
            match_reason=(f"Fast occurrence count: {keywords} → {total_occurrences} times "
                          f"on pages {sorted(matched_pages_1based)[:10]}"),
            tool='tool_global_count',
            count_result=total_occurrences,
            extra_meta={
                'is_count_result': True,
                'count_result': total_occurrences,
                'fast_path': 'OCCURRENCE_COUNT',
                'keywords': keywords,
                'matched_pages_1based': sorted(matched_pages_1based),
                'n_txt_files_scanned': len(txt_files),
            }
        )]

    if _is_arithmetic_lookup:
        print(f"  ⚡ [FAST PATH X: ARITHMETIC_LOOKUP] '{original_query[:70]}' "
              f"→ [] (arithmetic/lookup query, deferring to keyword_search fallback)")
        return []

    is_visual_mode = (target_type is None) or is_visual_type(target_type)
    
    # ========================================================================
    # UNIFIED SEMANTIC PRE-FILTERING LOGIC (same as tool_enumerate)
    # ========================================================================
    filtered_edges = []
    candidate_pages = []
    filter_details = []
    has_constraints = bool(section_title or page_range)
    
    if is_visual_mode:
        # Visual mode: count visual elements
        print(f"  → Visual mode: counting visual elements")
        
        if has_constraints:
            # Branch 1: is_visual_mode=True AND has page/section constraints
            # → Use filter functions
            if section_title:
                all_visual_edges = list(visual_edges)
                filtered_edges = filter_edges_by_section_title(all_visual_edges, hypergraph, block_index, section_title)
                filter_details.append(f"section={section_title}")
                print(f"     ✓ Section filter: {len(filtered_edges)} edges")
            elif page_range:
                # Convert 1-based user pages to 0-based physical indices
                physical_indices = _convert_page_range_to_physical(
                    page_range, reasoner, page_image_dir, clean_doc_id
                )
                print(f"     → Converted user pages {page_range} (1-based) to physical indices {physical_indices}")
                filtered_edges = filter_edges_by_page_range(list(visual_edges), block_index, physical_indices)
                filter_details.append(f"page_range={page_range}->physical={physical_indices}")
                print(f"     ✓ Page filter: {len(filtered_edges)} edges")
        else:
            # Branch 2: is_visual_mode=True BUT no page/section constraints
            # → Check document size, use keyword pre-filter if >40 anchors
            total_anchors = len(visual_edges)
            if total_anchors > 40:
                if keywords:
                    print(f"     → Large doc ({total_anchors} anchors), applying keyword pre-filter")
                    keyword_matched = []
                    for edge in visual_edges:
                        anchor = get_anchor_block_from_edge(edge, block_index)
                        if not anchor:
                            continue
                        caption = get_caption_block_from_edge(edge, block_index, anchor['block_id'])
                        related = get_related_blocks_from_edge(edge, block_index, anchor['block_id'])
                        text = extract_text_from_blocks([caption] + related).lower()
                        if any(fuzzy_keyword_match(kw, text, threshold=0.7) for kw in keywords):
                            keyword_matched.append(edge)
                            if len(keyword_matched) >= 40:
                                break
                    filtered_edges = keyword_matched
                    filter_details.append(f"keywords={keywords}")
                    print(f"     ✓ Keyword pre-filter: {len(filtered_edges)} edges")
                else:
                    # No keywords: pass all VU up to cap instead of returning empty
                    print(f"     → No keywords for pre-filter, using all anchors up to cap=40")
                    filtered_edges = list(visual_edges)[:40]
            else:
                print(f"     → Small doc ({total_anchors} anchors), using all")
                filtered_edges = list(visual_edges)
        
        # Apply type filter if specified
        if filtered_edges and target_type:
            filtered_edges = filter_edges_by_type(filtered_edges, block_index, target_type)
            filter_details.append(f"type={target_type}")
            print(f"     ✓ Type filter: {len(filtered_edges)} edges")
    
    else:
        # Text mode: count pages/sections
        print(f"  → Text mode: counting pages/sections")
        
        if has_constraints:
            # Branch 3: is_visual_mode=False BUT has page/section constraints
            # → Use hypergraph containment to find section ranges → get source pages
            if section_title:
                candidate_pages = get_pages_from_section_ranges(hypergraph, section_title)
                filter_details.append(f"section={section_title}")
                print(f"     ✓ Section ranges: {len(candidate_pages)} pages")
            elif page_range:
                # Convert 1-based user pages to 0-based physical indices
                physical_indices = _convert_page_range_to_physical(
                    page_range, reasoner, page_image_dir, clean_doc_id
                )
                print(f"     → Converted user pages {page_range} (1-based) to physical indices {physical_indices}")
                candidate_pages = physical_indices
                filter_details.append(f"page_range={page_range}->physical={physical_indices}")
                print(f"     ✓ Page range: {len(candidate_pages)} pages")
        else:
            # Branch 4: No matching conditions
            # Return empty so the pipeline can use coarse page evidence.
            print(f"     -> No constraints in text mode, returning empty (coarse page fallback)")
            candidate_pages = []
    
    # ── FAST PATH: SPARSE_GC_BYPASS ─────────────────────────────────────────
    # When GC finds zero candidate edges, returning count_result=0 with
    # is_count_result=True would block the router's vs_cond_fallback
    # (which requires ocr_has_results=False). The VLM then receives count=0
    # as confirmed ground truth → pred=0 → wrong for all GT>0 queries.
    #
    # Fix: return a log-only sentinel with is_count_result=False and
    # fallback_to_coarse_pages=True. The router excludes such results from the
    # ocr_has_results check, so vs_cond_fallback fires → tool_visual_search
    # supplies coarse page candidates so the VLM answers from actual evidence.
    if not filtered_edges and not candidate_pages:
        print(f"  [SPARSE_GC_BYPASS] No candidates found; returning log sentinel, "
              f"router will activate visual_search coarse page fallback")
        return [QueryHyperedge.create_from_components(
            score=0.0,
            match_reason="Count: no candidates (using coarse page evidence)",
            tool='tool_global_count',
            count_result=None,
            extra_meta={
                'is_count_result': False,
                'fallback_to_coarse_pages': True,
                'fast_path': 'SPARSE_GC_BYPASS',
            }
        )]

    # ── FAST PATH: count typed elements directly from hypergraph metadata ────
    # When target_type is specified with no keyword/page/section constraints,
    # the block metadata already tells us exactly how many elements exist —
    # no need to send images to VLM one-by-one.
    if (is_visual_mode and target_type
            and not keywords and not section_title and not (page_range or [])):
        typed_anchors = [
            b for b in block_index.values()
            if b.get('type', '').lower() == target_type.lower()
        ]
        if typed_anchors:
            n_elements = len(typed_anchors)
            n_pages = len(set(b.get('page', 0) for b in typed_anchors))
            pages_1based = sorted(set(b.get('page', 0) + 1 for b in typed_anchors))
            print(f"  ⚡ Fast count: {n_elements} '{target_type}' on {n_pages} page(s) "
                  f"{pages_1based} — skipping VLM")
            # Include matched QHEs so downstream can still access images if needed
            fast_matched = []
            for edge in filtered_edges:
                ab = get_anchor_block_from_edge(edge, block_index)
                if ab and ab.get('type', '').lower() == target_type.lower():
                    fast_matched.append({
                        'query_hyperedge': edge,
                        'anchor_block': ab,
                        'caption_block': get_caption_block_from_edge(
                            edge, block_index, ab['block_id']),
                        'related_blocks': get_related_blocks_from_edge(
                            edge, block_index, ab['block_id']),
                    })
            return [QueryHyperedge.create_from_components(
                score=1.0,
                match_reason=(
                    f"Fast count: {n_elements} '{target_type}' blocks "
                    f"on {n_pages} page(s) {pages_1based}"
                ),
                tool='tool_global_count',
                extra_meta={
                    'is_count_result': True,
                    'count_result': n_elements,
                    'n_elements': n_elements,
                    'n_pages_with_type': n_pages,
                    'pages_with_type_1based': pages_1based,
                    'fast_path': True,
                    'matched_results': fast_matched,
                }
            )]

    # ========================================================================
    # STAGE 1: VLM MATCHING WITH JSON OUTPUT
    # ========================================================================
    
    vlm_matching_json = None
    json_match_count = 0
    json_parse_success = False
    vlm_validation_attempted = False
    per_image = []
    matched_indices = []  # Indices of blocks that matched (match=true)
    
    if reasoner and original_query and clean_doc_id:
        print(f"\n  🎯 COUNT VALIDATION (Stage 1 VLM)")
        
        # Prepare images from filtered edges
        candidate_images = []
        edge_to_index = {}  # Map edge to its index in candidate_images
        
        print(f"     Mode: HYPEREDGE-LEVEL ({len(filtered_edges)} units)")
        for idx, edge in enumerate(filtered_edges):
            anchor_block = get_anchor_block_from_edge(edge, block_index)
            if anchor_block:
                page_idx = anchor_block.get('page')
                if page_idx is not None:
                    img_path = f"{page_image_dir}/{clean_doc_id}_{page_idx}.png"
                    candidate_images.append(img_path)
                    edge_to_index[idx] = len(candidate_images) - 1
        
        if candidate_images:
            # Stage 1: VLM Matching with JSON output
            try:
                matching_prompt = format_count_matching_prompt(original_query)
                print(f"     → Stage 1: VLM matching {len(candidate_images)} images...")
                vlm_validation_attempted = True
                vlm_matching_json, _ = reasoner.predict(matching_prompt, images=candidate_images)

                parsed, json_str, parse_error = _load_vlm_json_object(vlm_matching_json)
                if parsed is not None:
                    json_parse_success = True
                    json_match_count = int(parsed.get('count') or 0)
                    per_image = parsed.get('per_image', []) or []
                    matched_indices = [
                        i for i, item in enumerate(per_image)
                        if isinstance(item, dict) and item.get('match', False)
                    ]
                    print(f"     ✓ JSON parsed: count={json_match_count}, matched_indices={len(matched_indices)}")
                    if len(matched_indices) != json_match_count:
                        print(f"     ⚠️  WARNING: Mismatch! matched_indices({len(matched_indices)}) != count({json_match_count})")
                else:
                    matched_indices, per_image = _recover_per_image_matches(vlm_matching_json)
                    json_match_count = len(matched_indices)
                    if per_image:
                        print(
                            f"     ⚠️  JSON parse failed ({parse_error}); "
                            f"recovered {len(matched_indices)} matched indices from per_image fragments"
                        )
                    else:
                        print(f"     ⚠️  JSON parse failed ({parse_error}); no matches recovered")
                
            except Exception as e:
                 print(f"     ✗ Stage 1 failed: {e}")
                 vlm_matching_json = f"Error: {e}"
    
    # ========================================================================
    # PREPARE RESULTS: Only return blocks where match=true
    # ========================================================================
    
    # If VLM matching was attempted, trust its match flags. A valid zero-match
    # response or unrecoverable parse failure should not expand to all candidates.
    if vlm_validation_attempted:
        print(f"     → Filtering to {len(matched_indices)} matched blocks")
        final_edges = [filtered_edges[i] for i in matched_indices if i < len(filtered_edges)]
    else:
        final_edges = filtered_edges
    count = len(final_edges)
    
    type_counts = Counter()
    matched_results = []
    for edge in final_edges:
        anchor_block = get_anchor_block_from_edge(edge, block_index)
        if anchor_block:
            anchor_type = anchor_block.get('type', 'unknown')
            type_counts[anchor_type] += 1
            
            caption_block = get_caption_block_from_edge(edge, block_index, anchor_block['block_id'])
            related_blocks = get_related_blocks_from_edge(edge, block_index, anchor_block['block_id'])
            
            matched_results.append({
                'query_hyperedge': edge,
                'anchor_block': anchor_block,
                'caption_block': caption_block,
                'related_blocks': related_blocks
            })
    
    filter_str = ", ".join(filter_details) if filter_details else "no additional filters"
    match_reason = f"Count: {count} matched elements (VLM validated, {filter_str})"
    
    return [QueryHyperedge.create_from_components(
        edge=None,
        score=1.0,
        match_reason=match_reason,
        tool='tool_global_count',
        extra_meta={
            'type_distribution': dict(type_counts),
            'matched_results': matched_results,
            'is_count_result': True,
            'count_result': count,
            'vlm_matching_json': vlm_matching_json,
            'json_match_count': json_match_count,
            'json_parse_success': json_parse_success,
            'per_image': per_image,
            'vlm_validation_attempted': vlm_validation_attempted,
            'candidate_count': len(filtered_edges),
            'final_count': count
        }
    )]

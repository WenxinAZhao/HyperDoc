#!/usr/bin/env python3
"""
Retrieval Helpers
=================
Common helper functions for hypergraph retrieval.
"""

import re
import difflib
from typing import List, Dict, Optional, Union
from .graph_structures import Hypergraph

# ============================================================================
# Helper Functions
# ============================================================================

def build_block_index(hypergraph: Union[Dict, Hypergraph]) -> Dict[str, Dict]:
    """"""
    if isinstance(hypergraph, Hypergraph):
        return hypergraph.block_index
    return {b['block_id']: b for b in hypergraph.get('blocks', [])}


def get_visual_contextual_edges(hypergraph: Union[Dict, Hypergraph]) -> List[Dict]:
    """"""
    if isinstance(hypergraph, Hypergraph):
        return hypergraph.get_visual_contextual_edges()
    return [
        e for e in hypergraph.get('hyperedges', [])
        if e.get('edge_type') == 'contextual'
        and e.get('meta', {}).get('relation') == 'contextual'
    ]


def get_anchor_block_from_edge(edge: Dict, block_index: Dict) -> Optional[Dict]:
    """"""
    visual_types = {'figure', 'table', 'chart', 'map', 'diagram', 'graph'}
    for member_id in edge.get('members', []):
        block = block_index.get(member_id)
        # Check both type in visual_types OR member_id contains 'anchor'
        if block:
            is_visual = block.get('type') in visual_types
            is_anchor_id = 'anchor' in member_id
            if is_visual or is_anchor_id:
                return block
    return None


def get_caption_block_from_edge(edge: Dict, block_index: Dict, anchor_block_id: str) -> Optional[Dict]:
    """"""
    for member_id in edge.get('members', []):
        if member_id == anchor_block_id:
            continue
        block = block_index.get(member_id)
        # Check both type contains 'caption' OR block_id contains 'caption'
        if block and ('caption' in block.get('type', '') or 'caption' in member_id):
            return block
    return None


def get_related_blocks_from_edge(edge: Dict, block_index: Dict, anchor_block_id: str) -> List[Dict]:
    """"""
    related = []
    
    #
    allowed_types = {
        'text',
        'title',
        'figure_caption',
        'table_caption',
        'chart_caption',
        'map_caption',
        'diagram_caption',
        'graph_caption'
    }
    
    for member_id in edge.get('members', []):
        if member_id == anchor_block_id:
            continue
        block = block_index.get(member_id)
        if block:
            block_type = block.get('type', 'unknown')
            if block_type in allowed_types:
                related.append(block)
            #
    
    return related


def extract_text_from_blocks(blocks: List[Dict]) -> str:
    """"""
    texts = [b.get('text', '') for b in blocks if b and b.get('text')]
    return ' '.join(texts)


VISUAL_TYPES = {"figure", "table", "chart", "map", "diagram", "graph"}


def is_visual_type(type_str: Optional[str]) -> bool:
    if not type_str:
        return False
    return type_str.lower() in VISUAL_TYPES




def filter_edges_by_type(
    edges: List[Dict],
    block_index: Dict,
    target_type: Optional[str]
) -> List[Dict]:
    """Filter edges by visual type, fallback to original if no matches."""
    if not is_visual_type(target_type):
        print(f"    [filter_edges_by_type] No valid type constraint, returning {len(edges)} original edges")
        return edges
    filtered = []
    target = target_type.lower()
    for edge in edges:
        anchor = get_anchor_block_from_edge(edge, block_index)
        if anchor and anchor.get('type', '').lower() == target:
            filtered.append(edge)
    
    if filtered:
        print(f"    [filter_edges_by_type] Filtered to {len(filtered)} edges (type={target_type})")
        return filtered
    else:
        print(f"    [filter_edges_by_type] No matches for type={target_type}, fallback to {len(edges)} original edges")
        return edges


def filter_edges_by_page_range(
    edges: List[Dict],
    block_index: Dict,
    page_range: Union[List, int, str, None]
) -> List[Dict]:
    """Filter edges by page_range (0-based physical indices). Fallback to original if empty.
    
    Note: As of the new page handling system, page_range should contain 0-based physical indices
          that have already been converted from user's 1-based page numbers using VLM-based mapping.
          For tools that receive 1-based user pages, they should convert them first using
          convert_user_pages_to_physical() before calling this function.
    """
    target_pages = page_range
    if not target_pages:
        print(f"    [filter_edges_by_page_range] No page constraint, returning {len(edges)} original edges")
        return edges
    filtered = []
    for edge in edges:
        anchor = get_anchor_block_from_edge(edge, block_index)
        if anchor and anchor.get('page') in target_pages:
            filtered.append(edge)
    
    if filtered:
        print(f"    [filter_edges_by_page_range] Filtered to {len(filtered)} edges (physical_indices={target_pages})")
        return filtered
    else:
        print(f"    [filter_edges_by_page_range] No matches for physical_indices={target_pages}, fallback to {len(edges)} original edges")
        return edges


def filter_edges_by_section_title(
    edges: List[Dict],
    hypergraph: Union[Dict, Hypergraph],
    block_index: Dict,
    target_section: Optional[str]
) -> List[Dict]:
    """
    Filter edges by section title using containment edges.
    Fallback to original edges if no matches found.
    """
    if not target_section:
        print(f"    [filter_edges_by_section_title] No section constraint, returning {len(edges)} original edges")
        return edges

    print(f"    [filter_edges_by_section_title] Searching for section: '{target_section}'")
    matched_edges = []
    seen_edge_ids = set()
    seen_anchor_ids = set()
    total_section_edges = 0

    for edge in hypergraph.get('hyperedges', []):
        edge_relation = edge.get('meta', {}).get('relation', '')
        if (edge.get('edge_type') == 'containment'
            and edge_relation in ['containment', 'section_contains_visual', 'section_contains_visuals']):
            total_section_edges += 1

            title_block = None
            anchor_block = None
            for member_id in edge.get('members', []):
                block = block_index.get(member_id)
                if not block:
                    continue
                if block.get('type') in ['section_header', 'title']:
                    title_block = block
                elif block.get('type') in VISUAL_TYPES:
                    anchor_block = block

            if title_block and anchor_block:
                title_text = title_block.get('text', '')
                if match_section_title(title_text, target_section):
                    print(f"    [filter_edges_by_section_title] ✓ Found matching title: '{title_text[:60]}...'")
                    anchor_id = anchor_block['block_id']
                    if anchor_id in seen_anchor_ids:
                        continue
                    
                    # Try to find existing contextual edge
                    found_ctx = False
                    for ctx_edge in get_visual_contextual_edges(hypergraph):
                        if anchor_id in ctx_edge.get('members', []):
                            edge_id = ctx_edge.get('edge_id')
                            if edge_id not in seen_edge_ids:
                                matched_edges.append(ctx_edge)
                                seen_edge_ids.add(edge_id)
                                seen_anchor_ids.add(anchor_id)
                                print(f"    [filter_edges_by_section_title] ✓ Found existing contextual edge: {edge_id[-40:]}")
                            found_ctx = True
                            break
                    
                    # If no contextual edge exists (figure without caption/text),
                    # create a minimal edge with just the anchor
                    if not found_ctx:
                        synthetic_edge = {
                            'edge_id': f"{anchor_id}_section_synthetic",
                            'edge_type': 'contextual',
                            'members': [anchor_id],
                            'meta': {
                                'relation': 'contextual',
                                'source': 'section_containment_fallback'
                            }
                        }
                        matched_edges.append(synthetic_edge)
                        seen_anchor_ids.add(anchor_id)
                        print(f"    [filter_edges_by_section_title] ✓ Created synthetic edge (no caption/text)")

    print(f"    [filter_edges_by_section_title] Scanned {total_section_edges} section containment edges")
    if matched_edges:
        print(f"    [filter_edges_by_section_title] Found {len(matched_edges)} matching edges")
        return matched_edges
    else:
        print(f"    [filter_edges_by_section_title] No matches found, fallback to {len(edges)} original edges")
        return edges


def get_pages_from_section_ranges(
    hypergraph: Union[Dict, Hypergraph],
    target_section: Optional[str]
) -> List[int]:
    """Return 0-based pages from section_ranges by fuzzy title match."""
    if not target_section:
        return []

    pages = []
    for item in hypergraph.get('section_ranges', []) or []:
        title_text = item.get('title_text', '')
        if match_section_title(title_text, target_section):
            pages.extend(item.get('pages', []) or [])

    return sorted(list(set(p for p in pages if isinstance(p, int) and p >= 0)))


def get_title_block_for_anchor(
    hypergraph: Union[Dict, Hypergraph],
    block_index: Dict,
    anchor_block_id: str
) -> Optional[Dict]:
    """Find title block for a given anchor via section containment edges."""
    for edge in hypergraph.get('hyperedges', []):
        edge_relation = edge.get('meta', {}).get('relation', '')
        if (edge.get('edge_type') == 'containment'
            and edge_relation in ['containment', 'section_contains_visual', 'section_contains_visuals']
            and anchor_block_id in edge.get('members', [])):
            for member_id in edge.get('members', []):
                block = block_index.get(member_id)
                if block and block.get('type') in ['section_header', 'title']:
                    return block
    return None

# ============================================================================
# Matching Functions
# ============================================================================

def match_unit_id(text: str, target_id: str, unit_type: str = None) -> bool:
    """
    Robust OCR-tolerant unit ID matching.
    Matches: "Figure 1", "Table 1", "Chart 1", etc.
    """
    if not text or not target_id:
        return False
    text = text.lower()
    target_id = target_id.lower()
    
    # OCR substitutions map
    ocr_map = {
        '1': '[1lI|i!j]', 'l': '[1lI|i!j]', 'i': '[1lI|i!j]', 
        '0': '[0OoQ]', 'o': '[0OoQ]', 
        '2': '[2Zz]', 'z': '[2Zz]', 
        '5': '[5Ss]', 's': '[5Ss]', 
        '8': '[8B]', 'b': '[8B]'
    }
    
    id_pattern = ""
    for char in target_id:
        if char in ocr_map:
            id_pattern += ocr_map[char]
        elif char == '.':
            id_pattern += r'\.'
        else:
            id_pattern += re.escape(char)
            
    # Determine prefix pattern based on type
    prefixes = {
        "figure": r"(?:f[il1!|][gq9](?:ure)?\.?)",
        "fig": r"(?:f[il1!|][gq9](?:ure)?\.?)",
        "table": r"(?:t[a@]b(?:le)?\.?)",
        "chart": r"(?:c[h]a[r]t\.?)",
        "map": r"(?:m[a@]p\.?)",
        "graph": r"(?:g[r]a[p]h\.?)"
    }
    
    if unit_type and unit_type.lower() in prefixes:
        prefix_pattern = prefixes[unit_type.lower()]
    else:
        # If no type specified, match any common prefix
        prefix_pattern = r"(?:(?:f[il1!|][gq9](?:ure)?\.?)|(?:t[a@]b(?:le)?\.?)|(?:c[h]a[r]t\.?)|(?:m[a@]p\.?)|(?:g[r]a[p]h\.?))"
    
    # Combine: Prefix + Separator + ID + Negative Lookahead (not digit)
    full_pattern = f"{prefix_pattern}[^0-9a-zA-Z]*{id_pattern}(?![0-9])"
    
    return bool(re.search(full_pattern, text))


def match_section_title(text: str, target_title: str, threshold: float = 0.8) -> bool:
    """
    Fuzzy section title matching.
    """
    if not text or not target_title:
        return False
    text = text.lower()
    target_title = target_title.lower()
    
    # Direct substring match
    if target_title in text: 
        return True
        
    # Fuzzy match ratio
    return difflib.SequenceMatcher(None, text, target_title).ratio() >= threshold


def normalize_ocr(text: str) -> str:
    """Normalize common OCR errors before matching."""
    return (
        text.lower()
        .replace('0', 'o')
        .replace('1', 'l')
        .replace('|', 'l')
        .replace('rn', 'm')
    )


def fuzzy_keyword_match_score(keyword: str, text: str, threshold: float = 0.85) -> float:
    """
    Fuzzy keyword matching with similarity score (0-1).
    """
    # Apply OCR normalization
    keyword_norm = normalize_ocr(keyword)
    text_norm = normalize_ocr(text)
    
    # 1. Exact substring match (perfect score)
    if keyword_norm in text_norm:
        return 1.0
    
    # 2. Fuzzy match for OCR errors
    words = re.findall(r'\b\w+\b', text_norm)
    keyword_words = re.findall(r'\b\w+\b', keyword_norm)
    
    if not keyword_words:
        return 0.0
    
    # For multi-word keywords
    if len(keyword_words) > 1:
        k = len(keyword_words)
        keyword_phrase = ' '.join(keyword_words)
        best_score = 0.0
        
        # Sliding window
        for i in range(len(words) - k + 1):
            span = ' '.join(words[i:i+k])
            if span == keyword_phrase:
                return 1.0
            ratio = difflib.SequenceMatcher(None, keyword_phrase, span).ratio()
            best_score = max(best_score, ratio)
        
        # Scattered word matching
        if best_score < threshold:
            matched_scores = []
            for kw_word in keyword_words:
                word_best_score = 0.0
                for text_word in words:
                    if len(kw_word) <= 3:
                        if kw_word == text_word:
                            word_best_score = 1.0
                            break
                    else:
                        ratio = difflib.SequenceMatcher(None, kw_word, text_word).ratio()
                        word_best_score = max(word_best_score, ratio)
                matched_scores.append(word_best_score)
            
            if matched_scores:
                scattered_score = sum(matched_scores) / len(matched_scores)
                best_score = max(best_score, scattered_score * 0.9)
        
        return best_score
    
    # For single-word keywords
    keyword_word = keyword_words[0] if keyword_words else keyword_norm
    
    if len(keyword_word) <= 3:
        return 1.0 if keyword_word in words else 0.0
    
    best_score = 0.0
    for word in words:
        ratio = difflib.SequenceMatcher(None, keyword_word, word).ratio()
        best_score = max(best_score, ratio)
    
    return best_score


def fuzzy_keyword_match(keyword: str, text: str, threshold: float = 0.85) -> bool:
    """Boolean version of fuzzy keyword matching."""
    score = fuzzy_keyword_match_score(keyword, text, threshold)
    return score >= threshold


# ============================================================================
# Pagination Rule Calculation (for double-page PDFs)
# ============================================================================

def probe_page_with_vlm(
    reasoner,
    page_image_dir: str,
    clean_doc_id: str,
    physical_idx: int,
    format_page_extraction_prompt
) -> Optional[Dict]:
    """"""
    import os
    import json
    
    img_path = os.path.join(page_image_dir, f"{clean_doc_id}_{physical_idx}.png")
    
    if not os.path.exists(img_path):
        print(f"     ⚠️ VLM probe: Image not found for physical index {physical_idx}: {img_path}")
        return None
    
    prompt = format_page_extraction_prompt()
    try:
        vlm_out, _ = reasoner.predict(prompt, images=[img_path])
        start = vlm_out.find('{')
        end = vlm_out.rfind('}')
        if start != -1 and end != -1:
            json_str = vlm_out[start:end+1]
            json_str = json_str.replace('\n', ' ').replace('\r', ' ')
            data = json.loads(json_str)
            return {
                "physical_index": physical_idx,
                "layout": data.get("layout", "single"),
                "found_pages": [str(p) for p in data.get("found_pages", []) if str(p).isdigit()]
            }
        else:
            print(f"     ⚠️ VLM probe: No JSON found in VLM output for physical index {physical_idx}.")
            return None
    except Exception as e:
        print(f"     ❌ VLM probe failed for physical index {physical_idx}: {e}")
        return None


def calculate_pagination_rule(vlm_results: List[Dict], physical_indices: List[int]) -> Dict:
    """"""
    from collections import Counter
    
    if not vlm_results or not physical_indices:
        return {
            "layout": "unknown",
            "stride": 1,
            "base_offset": 0,
            "confidence": 0.0
        }
    
    #
    layout_count = Counter()
    for result in vlm_results:
        layout = result.get("layout", "single")
        layout_count[layout] += 1
    
    #
    determined_layout = layout_count.most_common(1)[0][0] if layout_count else "single"
    confidence = layout_count[determined_layout] / len(vlm_results) if vlm_results else 0.0
    
    #
    if determined_layout == "single":
        #
        #
        offsets = []
        for i, result in enumerate(vlm_results):
            if i >= len(physical_indices):
                break
            found_pages = result.get("found_pages", [])
            if found_pages and str(found_pages[0]).isdigit():
                printed_page_1based = int(found_pages[0])
                offset = physical_indices[i] - (printed_page_1based - 1)
                offsets.append(offset)
        
        #
        if offsets:
            offset_counts = Counter(offsets)
            base_offset = offset_counts.most_common(1)[0][0]
        else:
            base_offset = 0
        
        return {
            "layout": "single",
            "stride": 1,
            "base_offset": base_offset,
            "confidence": confidence,
            "samples": list(zip(physical_indices, [r.get("found_pages") for r in vlm_results]))
        }
    
    else:  # double
        #
        # left_page_1based = a + stride * physical_idx
        # right_page_1based = left_page_1based + 1
        left_pages_1based = []
        
        for i, result in enumerate(vlm_results):
            if i >= len(physical_indices):
                break
            found_pages = result.get("found_pages", [])
            if len(found_pages) >= 2:
                #
                pages_int = []
                for p in found_pages[:2]:
                    if str(p).isdigit():
                        pages_int.append(int(p))
                if pages_int:
                    left_page_1based = min(pages_int)
                    left_pages_1based.append((physical_indices[i], left_page_1based))
        
        if len(left_pages_1based) >= 2:
            #
            idx1, left1 = left_pages_1based[0]
            idx2, left2 = left_pages_1based[1]
            
            stride = (left2 - left1) // (idx2 - idx1) if (idx2 - idx1) != 0 else 2
            a = left1 - stride * idx1
            
            return {
                "layout": "double",
                "stride": stride,
                "a": a,
                "confidence": confidence,
                "samples": list(zip(physical_indices, [r.get("found_pages") for r in vlm_results])),
                "formula": f"left_page(1-based) = {a} + {stride} * physical_idx"
            }
        elif len(left_pages_1based) == 1:
            #
            idx, left = left_pages_1based[0]
            stride = 2
            a = left - stride * idx
            
            return {
                "layout": "double",
                "stride": stride,
                "a": a,
                "confidence": confidence * 0.5,
                "samples": list(zip(physical_indices, [r.get("found_pages") for r in vlm_results])),
                "formula": f"left_page(1-based) = {a} + {stride} * physical_idx (single sample)"
            }
        else:
            #
            return {
                "layout": "double",
                "stride": 2,
                "a": 0,
                "confidence": 0.0,
                "error": "Could not extract page numbers from VLM results"
            }


def convert_user_pages_to_physical(
    user_pages_1based: List[int],
    page_mapping: Dict
) -> List[int]:
    """"""
    if not user_pages_1based:
        return []
    
    layout = page_mapping.get("layout", "single")
    all_physical_indices = []
    
    for user_page_1based in user_pages_1based:
        if layout == "single":
            #
            offset = page_mapping.get("base_offset", 0)
            physical_idx = (user_page_1based - 1) + offset
            
            if physical_idx >= 0:
                all_physical_indices.append(physical_idx)
        
        elif layout == "double":
            #
            stride = page_mapping.get("stride", 2)
            a = page_mapping.get("a", 0)
            
            #
            if (user_page_1based - a) % stride == 0:
                idx_if_left = (user_page_1based - a) // stride
                if idx_if_left >= 0:
                    all_physical_indices.append(idx_if_left)
            
            #
            if (user_page_1based - 1 - a) % stride == 0:
                idx_if_right = (user_page_1based - 1 - a) // stride
                if idx_if_right >= 0:
                    all_physical_indices.append(idx_if_right)
        
        else:
            #
            physical_idx = user_page_1based - 1
            if physical_idx >= 0:
                all_physical_indices.append(physical_idx)
    
    #
    return sorted(list(set(all_physical_indices)))

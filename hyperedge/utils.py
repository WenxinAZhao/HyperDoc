#!/usr/bin/env python3
""""""

import re
import torch
import numpy as np
from typing import Dict, List, Optional


def extract_unit_text(unit: Dict, include_caption: bool = True, 
                     include_related: bool = True,
                     include_title: bool = False) -> str:
    """"""
    text_parts = []
    
    # Caption
    if include_caption:
        caption_block = unit.get('caption_block')
        if caption_block:
            caption_text = caption_block.get('text', '').strip()
            if caption_text:
                text_parts.append(caption_text)
    
    # Context title
    if include_title:
        title_block = unit.get('context_title')
        if title_block:
            title_text = title_block.get('text', '').strip()
            if title_text:
                text_parts.append(title_text)
    
    # Related texts
    if include_related:
        related_texts = unit.get('related_texts', [])
        for rt_block in related_texts:
            rt_text = rt_block.get('text', '').strip()
            if rt_text:
                text_parts.append(rt_text)
    
    return " ".join(text_parts)


def encode_text_with_colbert(text: str, colbert_model, device: str = "cuda") -> np.ndarray:
    """"""
    with torch.no_grad():
        #
        result = colbert_model.docFromText([text], bsize=1)
        
        #
        if isinstance(result, tuple):
            embed = result[0]
        else:
            embed = result
        
        #
        if embed.dim() == 3:
            embed = embed.squeeze(0)  # [num_tokens, 128]
    
    return embed.cpu().numpy()


def encode_query_with_colbert(query: str, colbert_model, device: str = "cuda") -> np.ndarray:
    """"""
    with torch.no_grad():
        result = colbert_model.queryFromText([query], bsize=1)
        
        #
        if isinstance(result, tuple):
            embed = result[0]
        else:
            embed = result
        
        #
        if embed.dim() == 3:
            embed = embed.squeeze(0)  # [num_tokens, 128]
    
    return embed.cpu().numpy()


def compute_maxsim_score(
    query_embed: np.ndarray,  # [Q, 128]
    doc_embed: np.ndarray,    # [D, 128]
    device: str = "cuda"
) -> float:
    """"""
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    query_tensor = torch.from_numpy(query_embed).to(device_obj).float()
    doc_tensor = torch.from_numpy(doc_embed).to(device_obj).float()
    
    #
    sim_matrix = torch.matmul(query_tensor, doc_tensor.t())  # [Q, D]
    max_sims = torch.max(sim_matrix, dim=1)[0]  # [Q]
    maxsim_score = torch.mean(max_sims).item()
    
    return maxsim_score


# ============================================================================
#
# ============================================================================

def extract_answer(prediction: str) -> Optional[str]:
    """"""
    if not prediction or not isinstance(prediction, str):
        return None
    
    prediction = prediction.strip()
    
    # Strategy 1: Extract from <answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', prediction, re.DOTALL | re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()
    
    # Strategy 2: Look for explicit answer patterns
    patterns = [
        r'(?:the\s+)?answer\s+is[:\s]+([^\n\.]+)',  # "The answer is: X"
        r'(?:final\s+)?answer[:\s]+([^\n\.]+)',     # "Final answer: X"
        r'^([^\n]+)\.$',                             # First sentence ending with period
        r'^([^\n]{20,100})',                         # First 20-100 chars
    ]
    
    for pattern in patterns:
        match = re.search(pattern, prediction, re.IGNORECASE | re.MULTILINE)
        if match:
            answer = match.group(1).strip()
            answer = re.sub(r'^(is|:|\s)+', '', answer)
            if answer and len(answer) > 2:
                return answer
    
    # Strategy 3: Fallback - first sentence
    first_sentence = re.split(r'[\.!?\n]\s+', prediction)[0]
    if first_sentence and len(first_sentence) < 200:
        return first_sentence.strip()
    
    return prediction[:100].strip() if len(prediction) > 10 else None


def extract_count(prediction: str) -> Optional[int]:
    """"""
    if not prediction:
        return None
    
    # Try <answer> tag first
    answer = extract_answer(prediction)
    if answer:
        number_match = re.search(r'\b(\d+)\b', answer)
        if number_match:
            return int(number_match.group(1))
    
    # Fallback: search patterns
    patterns = [
        r'count[:\s]+(\d+)',
        r'total[:\s]+(\d+)',
        r'(\d+)\s+(?:items?|elements?|units?|images?|charts?|figures?)',
        r'^\s*(\d+)\s*$',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, prediction, re.IGNORECASE | re.MULTILINE)
        if match:
            return int(match.group(1))
    
    return None


def extract_count_answer(prediction: str) -> Optional[str]:
    if not prediction:
        return None

    answer = extract_answer(prediction)
    if answer:
        return answer.strip()

    prediction = prediction.strip()
    first_sentence = re.split(r'[\.!?\n]\s+', prediction)[0].strip()
    if first_sentence and len(first_sentence) < 200:
        return first_sentence

    return prediction[:100].strip() if len(prediction) > 10 else None


def _he_anchor_key(he) -> Optional[str]:
    """Extract anchor key for entity grouping.  Works for both replay dicts and QueryHyperedge objects."""
    if hasattr(he, 'raw_dict'):
        ab = he.raw_dict.get('anchor_block')
    elif isinstance(he, dict):
        ab = he.get('anchor_block')
    else:
        return None
    if ab is None:
        return None
    bid = ab.get('block_id') if isinstance(ab, dict) else getattr(ab, 'block_id', None)
    if bid is not None:
        return f"bid_{bid}"
    page = ab.get('page') if isinstance(ab, dict) else getattr(ab, 'page', None)
    if page is not None:
        return f"page_{page}"
    return None


def _he_anchor_key_page(he) -> Optional[str]:
    """Strategy A: group by anchor page.
    Multiple blocks on the same page belong to one entity group.
    Coarser than block_id; enables round-robin to diversify entity coverage.
    """
    rd = he.raw_dict if hasattr(he, 'raw_dict') else (he if isinstance(he, dict) else None)
    if rd is None:
        return None
    ab = rd.get('anchor_block')
    if ab is None:
        return None
    page = ab.get('page') if isinstance(ab, dict) else getattr(ab, 'page', None)
    return f"page_{page}" if page is not None else None


def _he_anchor_key_entity(he) -> Optional[str]:
    """Strategy B: group by entity ID extracted from match_reason.
    For tool_find_by_id HEs: extracts ID from 'Matched ID X in caption' as "entity_X".
    Falls back to page-based grouping for keyword_search / other tools.
    Entity label becomes the raw ID, such as "Evidence [1]:".
    """
    import re as _re
    rd = he.raw_dict if hasattr(he, 'raw_dict') else (he if isinstance(he, dict) else None)
    if rd is None:
        return None
    match_reason = rd.get('match_reason', '') or ''
    m = _re.search(r'Matched ID\s+(\S+)\s+in', match_reason)
    if m:
        return f"entity_{m.group(1)}"
    # Fallback: page-based
    return _he_anchor_key_page(he)


def convert_units_to_vlm_input(
    query_hyperedges: List[Dict],
    page_image_dir: str,
    clean_doc_id: str,
    max_candidates: Optional[int] = 3,
    visual_strategy: str = "mixed",
    debug_info: Optional[Dict] = None,
    intent_state: Optional[Dict] = None,
    group_mode: str = "block_id",
    question_text: Optional[str] = None,
) -> tuple[List[str], List[str]]:
    """"""
    from PIL import Image
    import tempfile
    import os

    images = []
    context_texts = []
    all_added_pages = set()
    query_text = " ".join(
        str(getattr(res, 'match_reason', '') or getattr(res, 'tool', '') or '')
        for res in query_hyperedges[:8]
    )
    supplement_signal_text = (question_text or "").strip() or query_text

    def _is_page_node(res) -> bool:
        if hasattr(res, 'raw_dict'):
            return bool(res.raw_dict.get('is_page_node'))
        return bool(res.get('is_page_node'))

    def _source_page_of(res) -> Optional[int]:
        if hasattr(res, 'raw_dict'):
            return res.raw_dict.get('source_page')
        return res.get('source_page')

    def _to_debug_entry(res) -> Dict:
        if hasattr(res, 'raw_dict'):
            rd = res.raw_dict
            return {
                "tool": getattr(res, 'tool', rd.get('tool')),
                "score": round(float(getattr(res, 'score', rd.get('score', 0.0))), 6),
                "source_page": rd.get('source_page'),
                "is_page_node": bool(rd.get('is_page_node')),
            }
        return {
            "tool": res.get('tool'),
            "score": round(float(res.get('score', 0.0)), 6),
            "source_page": res.get('source_page'),
            "is_page_node": bool(res.get('is_page_node')),
        }

    def _needs_wider_page_supplement() -> bool:
        intent_type = (intent_state or {}).get("intent_type")
        arithmetic_cues = re.compile(
            r'\b(?:calculate|sum|combined|difference|increase|decrease|total|amount|revenue|'
            r'expenditure|income|assets|liabilities|equity|gains|loss|quantity|capacity|'
            r'percentage|percent|rate|value|days|shares|responses|skipped)\b',
            re.IGNORECASE,
        )
        return (
            intent_type in {"count", "compare", "cross_reasoning"}
            and arithmetic_cues.search(supplement_signal_text) is not None
        )

    page_supplement_limit = 5 if _needs_wider_page_supplement() else 3

    def _append_candidate(res, idx: int):
        if hasattr(res, 'to_vlm_input'):
            unit_images, unit_text = res.to_vlm_input(clean_doc_id, page_image_dir, candidate_idx=idx)
            images.extend(unit_images)
            if unit_text:
                context_texts.append(unit_text)
            for img in unit_images:
                if f"{clean_doc_id}_" in img:
                    try:
                        p = int(img.split(f"{clean_doc_id}_")[1].split(".")[0])
                        all_added_pages.add(p)
                    except Exception:
                        pass
            return

        anchor_block = res.get('anchor_block')
        caption_block = res.get('caption_block')
        related_blocks = res.get('related_blocks', []) or []

        if anchor_block:
            page_idx = anchor_block.get('page')
            bbox = anchor_block.get('bbox')
            if page_idx is not None and bbox:
                page_img_name = f"{clean_doc_id}_{page_idx}.png"
                page_img_path = os.path.join(page_image_dir, page_img_name)
                if os.path.exists(page_img_path):
                    try:
                        page_img = Image.open(page_img_path)
                        x1, y1, x2, y2 = bbox
                        padding = 10
                        x1 = max(0, x1 - padding)
                        y1 = max(0, y1 - padding)
                        x2 = min(page_img.width, x2 + padding)
                        y2 = min(page_img.height, y2 + padding)
                        cropped_img = page_img.crop((x1, y1, x2, y2))
                        temp_file = tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix='.png',
                            prefix=f'candidate_{idx}_'
                        )
                        cropped_img.save(temp_file.name)
                        temp_file.close()
                        images.append(temp_file.name)
                    except Exception as e:
                        print(f"  Failed to crop candidate {idx} from page {page_idx}: {e}")

        current_pages = set()
        if anchor_block and anchor_block.get('page') is not None:
            current_pages.add(anchor_block['page'])
        for blk in related_blocks:
            page_idx = blk.get('page')
            if page_idx is not None:
                current_pages.add(page_idx)

        for page_idx in current_pages:
            if page_idx in all_added_pages:
                continue
            all_added_pages.add(page_idx)
            page_img_name = f"{clean_doc_id}_{page_idx}.png"
            page_img_path = os.path.join(page_image_dir, page_img_name)
            if os.path.exists(page_img_path):
                images.append(page_img_path)

        texts = []
        if caption_block and caption_block.get('text'):
            texts.append(caption_block['text'])
        for blk in related_blocks:
            t = blk.get('text')
            if t:
                texts.append(t)
        if texts:
            unit_text = "\n".join(texts)
            context_texts.append(f"Unit {idx}: {unit_text}")

    if visual_strategy == "fixed_topk":
        # Strategy A: OCR edges first, then page nodes as supplement.
        #
        ocr_edges  = [r for r in query_hyperedges
                      if not (hasattr(r, 'raw_dict') and
                              r.raw_dict.get('is_page_node'))]
        page_nodes = [r for r in query_hyperedges
                      if hasattr(r, 'raw_dict') and
                      r.raw_dict.get('is_page_node')]

        #
        primary_candidates = ocr_edges[:max_candidates]
        for idx, res in enumerate(primary_candidates, 1):
            if hasattr(res, 'to_vlm_input'):
                unit_images, unit_text = res.to_vlm_input(clean_doc_id, page_image_dir, candidate_idx=idx)
                images.extend(unit_images)
                if unit_text:
                    context_texts.append(unit_text)
                # track pages added
                for img in unit_images:
                    if f"{clean_doc_id}_" in img:
                        try:
                            p = int(img.split(f"{clean_doc_id}_")[1].split(".")[0])
                            all_added_pages.add(p)
                        except Exception:
                            pass

        #
        for res in page_nodes:
            src_page = _source_page_of(res)
            if src_page is None or src_page in all_added_pages:
                continue
            page_img = os.path.join(page_image_dir, f"{clean_doc_id}_{src_page}.png")
            if os.path.exists(page_img):
                images.append(page_img)
                all_added_pages.add(src_page)

        if debug_info is not None:
            debug_info.update({
                "focused_top3_sources": [_to_debug_entry(r) for r in primary_candidates],
                "supplement_triggered": bool(page_nodes),
                "supplement_pages_added": [],
                "visual_page_nodes_count": len(page_nodes),
            })

        return images, context_texts

    # Intent-aware MULTI_UNIT: entity-balanced round-robin with entity labels.
    # Only activates when intent_state is provided and evidence_level == MULTI_UNIT.
    # Groups HEs by anchor_key (entity groups, e.g. entity A vs entity B in a compare task).
    # Round-robin selects up to max_candidates from groups to ensure entity balance.
    # Relabels context text "Unit N:" as "Evidence [A]:" for clearer VLM comparison context.
    if intent_state is not None:
        _ev = (intent_state.get("target") or {}).get("evidence_level")
        _it = (intent_state or {}).get("intent_type", "")
        if _ev == "MULTI_UNIT" and _it not in ("count", "enumerate"):
            _all_regular = [r for r in query_hyperedges if not _is_page_node(r)]
            _all_pnodes  = [r for r in query_hyperedges if _is_page_node(r)]

            # Select grouping function based on group_mode
            if group_mode == "page":
                _anchor_fn = _he_anchor_key_page
            elif group_mode == "entity":
                _anchor_fn = _he_anchor_key_entity
            else:
                _anchor_fn = _he_anchor_key

            # Group by anchor key (each group = one entity/comparison unit)
            _groups: Dict[Optional[str], list] = {}
            _group_order: list = []
            for _he in _all_regular:
                _k = _anchor_fn(_he)
                if _k not in _groups:
                    _groups[_k] = []
                    _group_order.append(_k)
                _groups[_k].append(_he)

            # Entity labels:
            #   entity mode: use extracted ID as label (e.g., "1", "10") for ID-matched HEs
            #   page / block_id mode: A, B, C, ...
            _entity_labels = {}
            for i, _k in enumerate(_group_order):
                if group_mode == "entity" and _k and _k.startswith("entity_"):
                    _entity_labels[_k] = _k[len("entity_"):]   # raw ID, e.g. "1" or "1,10"
                else:
                    _entity_labels[_k] = chr(65 + i) if i < 26 else str(i + 1)

            # Round-robin select up to max_candidates; None means no convert-stage cap.
            _round_robin_limit = (
                max_candidates if max_candidates is not None else len(_all_regular)
            )
            _sel_hes: list = []
            _sel_labels: list = []
            _used: dict = {_k: 0 for _k in _group_order}
            while len(_sel_hes) < _round_robin_limit:
                _added = False
                for _k in _group_order:
                    if len(_sel_hes) >= _round_robin_limit:
                        break
                    _i = _used[_k]
                    if _i < len(_groups[_k]):
                        _sel_hes.append(_groups[_k][_i])
                        _sel_labels.append(_entity_labels[_k])
                        _used[_k] += 1
                        _added = True
                if not _added:
                    break

            # Append each HE; relabel context_text to "Evidence [X]:"
            for _unit_idx, (_he, _lbl) in enumerate(zip(_sel_hes, _sel_labels), 1):
                _pre_len = len(context_texts)
                _append_candidate(_he, _unit_idx)
                if len(context_texts) > _pre_len:
                    _t = context_texts[-1]
                    _prefix = f"Unit {_unit_idx}:"
                    if _t.startswith(_prefix):
                        context_texts[-1] = f"Evidence [{_lbl}]:" + _t[len(_prefix):]

            # Page-node supplement (same limit as mixed: up to 3 pages)
            _supp: list = []
            for _res in _all_pnodes:
                if len(_supp) >= page_supplement_limit:
                    break
                _sp = _source_page_of(_res)
                if _sp is None or _sp in all_added_pages:
                    continue
                _pi = os.path.join(page_image_dir, f"{clean_doc_id}_{_sp}.png")
                if os.path.exists(_pi):
                    images.append(_pi)
                    all_added_pages.add(_sp)
                    _supp.append(_sp)

            if debug_info is not None:
                debug_info.update({
                    "focused_top3_sources": [_to_debug_entry(r) for r in _sel_hes],
                    "supplement_triggered": bool(_all_pnodes),
                    "supplement_pages_added": _supp,
                    "visual_page_nodes_count": len(_all_pnodes),
                    "intent_aware": True,
                    "n_entity_groups": len(_group_order),
                })
            return images, context_texts

    #
    regular_candidates = [r for r in query_hyperedges if not _is_page_node(r)]
    page_nodes = [r for r in query_hyperedges if _is_page_node(r)]

    primary_candidates = (
        regular_candidates
        if max_candidates is None
        else regular_candidates[:max_candidates]
    )
    for idx, res in enumerate(primary_candidates, 1):
        _append_candidate(res, idx)

    supplement_pages_added = []
    supplement_triggered = bool(page_nodes)
    if page_nodes:
        for res in page_nodes:
            if len(supplement_pages_added) >= page_supplement_limit:
                break
            src_page = _source_page_of(res)
            if src_page is None or src_page in all_added_pages:
                continue
            page_img = os.path.join(page_image_dir, f"{clean_doc_id}_{src_page}.png")
            if os.path.exists(page_img):
                images.append(page_img)
                all_added_pages.add(src_page)
                supplement_pages_added.append(src_page)

    if debug_info is not None:
        debug_info.update({
            "focused_top3_sources": [_to_debug_entry(r) for r in primary_candidates],
            "supplement_triggered": supplement_triggered,
            "supplement_pages_added": supplement_pages_added,
            "visual_page_nodes_count": len(page_nodes),
        })
    
    return images, context_texts


# ============================================================================
#
# ============================================================================

__all__ = [
    "extract_unit_text",
    "encode_query_with_colbert",
    "encode_text_with_colbert",
    "compute_maxsim_score",
    "extract_answer",
    "extract_count",
    "convert_units_to_vlm_input",
]

#!/usr/bin/env python3
""""""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np
import torch
from PIL import Image


def load_ocr_result(ocr_path: Path) -> Dict:
    """"""
    with open(ocr_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_page_image_paths(ocr_data: Dict, image_dir: Optional[str] = None) -> List[str]:
    """"""
    doc_id = ocr_data["document_id"]
    total_pages = ocr_data["total_pages"]

    script_dir = Path(__file__).parent

    if image_dir is not None:
        candidate_dirs = [Path(image_dir)]
    else:
        # Auto-discover: try every subdir of tmp/ then fall back to MMLongBench
        tmp_root = script_dir / "tmp"
        candidate_dirs = sorted(tmp_root.iterdir()) if tmp_root.is_dir() else []
        fallback = script_dir / "tmp" / "MMLongBench"
        if fallback not in candidate_dirs:
            candidate_dirs.append(fallback)

    # Pick the first directory that contains at least one page image for this doc
    chosen_dir = None
    for d in candidate_dirs:
        if not d.is_dir():
            continue
        probe = d / f"{doc_id}_0.png"
        if probe.exists():
            chosen_dir = d
            break

    if chosen_dir is None:
        # No images found in any candidate directory
        print(f"Warning: no page images found for {doc_id} in any tmp/ subdir")
        return []

    image_paths = []
    for page_idx in range(total_pages):
        img_path = chosen_dir / f"{doc_id}_{page_idx}.png"
        if img_path.exists():
            image_paths.append(str(img_path))
        else:
            print(f"Warning: image not found: {img_path}")

    return image_paths


def compute_bbox_distance(bbox1: List[float], bbox2: List[float]) -> float:
    """"""
    center1_x = (bbox1[0] + bbox1[2]) / 2
    center1_y = (bbox1[1] + bbox1[3]) / 2
    center2_x = (bbox2[0] + bbox2[2]) / 2
    center2_y = (bbox2[1] + bbox2[3]) / 2
    
    return np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)


def is_vertically_close(bbox1: List[float], bbox2: List[float], threshold: float = 200) -> bool:
    """"""
    #
    #
    
    #
    vertical_gap = min(
        abs(bbox1[1] - bbox2[3]),  #
        abs(bbox1[3] - bbox2[1])   #
    )
    
    return vertical_gap < threshold


def find_caption_for_figure(
    figure_block: Dict,
    all_blocks: List[Dict],
    max_distance: float = 300
) -> Optional[Dict]:
    """"""
    figure_page = figure_block["page"]
    figure_bbox = figure_block["bbox"]
    figure_type = figure_block["type"]
    
    #
    caption_candidates = []
    text_candidates = []
    
    for block in all_blocks:
        #
        if block["block_id"] == figure_block["block_id"]:
            continue
        
        block_page = block["page"]
        block_type = block.get("type", "")
        block_bbox = block["bbox"]
        
        #
        if block_page != figure_page:
            continue
        
        #
        is_caption_type = False
        is_text_type = False
        
        if figure_type == "figure" and block_type == "figure_caption":
            is_caption_type = True
        elif figure_type == "table" and block_type == "table_caption":
            is_caption_type = True
        elif block_type == "text":
            is_text_type = True
        
        if not (is_caption_type or is_text_type):
            continue
        
        #
        #
        fig_x_left, fig_x_right = figure_bbox[0], figure_bbox[2]
        block_x_left, block_x_right = block_bbox[0], block_bbox[2]
        
        x_overlap = min(fig_x_right, block_x_right) - max(fig_x_left, block_x_left)
        fig_width = fig_x_right - fig_x_left
        
        #
        if x_overlap < fig_width * 0.3:
            continue
        
        #
        is_above = block_bbox[3] < figure_bbox[1]  #
        is_below = block_bbox[1] > figure_bbox[3]  #
        
        if not (is_above or is_below):
            continue
        
        #
        if is_below:
            distance = block_bbox[1] - figure_bbox[3]  #
        else:  # is_above
            distance = figure_bbox[1] - block_bbox[3]  #
        
        if distance < max_distance:
            candidate = {
                "block": block,
                "distance": distance,
                "is_below": is_below
            }
            
            if is_caption_type:
                caption_candidates.append(candidate)
            else:  # is_text_type
                text_candidates.append(candidate)
    
    #
    if caption_candidates:
        #
        caption_candidates.sort(key=lambda x: (0 if x["is_below"] else 1, x["distance"]))
        return caption_candidates[0]["block"]
    
    if text_candidates:
        #
        #
        below_texts = [c for c in text_candidates if c["is_below"]]
        if below_texts:
            below_texts.sort(key=lambda x: x["distance"])
            return below_texts[0]["block"]
        #
        above_texts = [c for c in text_candidates if not c["is_below"]]
        if above_texts:
            above_texts.sort(key=lambda x: x["distance"])
            return above_texts[0]["block"]
    
    return None


def find_nearest_title(
    figure_block: Dict,
    all_blocks: List[Dict],
    max_pages_back: int = 3
) -> Optional[Dict]:
    """"""
    figure_page = figure_block["page"]
    figure_bbox = figure_block["bbox"]  # [x1, y1, x2, y2]
    figure_y_top = figure_bbox[1]
    figure_x_range = (figure_bbox[0], figure_bbox[2])
    
    #
    candidates = []
    
    for block in all_blocks:
        if block.get("type") != "title":
            continue
        
        block_page = block["page"]
        block_bbox = block["bbox"]
        block_y_top = block_bbox[1]
        block_x_range = (block_bbox[0], block_bbox[2])
        
        #
        if block_page > figure_page:
            continue
        
        #
        if block_page == figure_page:
            #
            is_above = block_y_top < figure_y_top
            
            #
            x_overlap = min(block_x_range[1], figure_x_range[1]) - max(block_x_range[0], figure_x_range[0])
            is_left_or_right = x_overlap < 50  #
            
            #
            if not (is_above or is_left_or_right):
                continue
        
        #
        if figure_page - block_page > max_pages_back:
            continue
        
        #
        if block_page == figure_page:
            #
            vertical_distance = abs(figure_y_top - block_y_top)
            distance_score = vertical_distance / 1000  #
        else:
            #
            distance_score = figure_page - block_page
        
        candidates.append({
            "block": block,
            "distance_score": distance_score,
            "page": block_page
        })
    
    if not candidates:
        return None
    
    #
    candidates.sort(key=lambda x: (x["distance_score"], -x["page"]))
    return candidates[0]["block"]


def load_clip_model(
    clip_model_path: str = os.environ.get('HYPERDOC_OPENCLIP_MODEL', ''),
    device: str = "cuda"
):
    """"""
    if not clip_model_path:
        raise RuntimeError("HYPERDOC_OPENCLIP_MODEL is not set")
    import open_clip
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32',
        pretrained=clip_model_path,
        device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
    return model, preprocess, tokenizer


def find_related_text_blocks(
    figure_block: Dict,
    caption_block: Optional[Dict],
    title_block: Optional[Dict],
    all_blocks: List[Dict],
    image_paths: List[str],
    clip_model = None,
    clip_preprocess = None,
    clip_tokenizer = None,
    device: str = "cuda",
    page_range: int = 2,
    similarity_threshold: float = 0.25,
    max_blocks: int = 3,
    min_text_length: int = 5,
    min_confidence: float = 0.9,
    margin_ratio: float = 0.1
) -> List[Dict]:
    """"""
    if not clip_model or not clip_preprocess or not clip_tokenizer:
        return []
    
    figure_page = figure_block["page"]
    figure_bbox = figure_block["bbox"]
    
    #
    if figure_page >= len(image_paths):
        return []
    
    try:
        #
        img_path = image_paths[figure_page]
        img_pil = Image.open(img_path).convert('RGB')
        img_width, img_height = img_pil.size
        
        #
        x1, y1, x2, y2 = figure_bbox
        width = x2 - x1
        height = y2 - y1
        margin_w = width * margin_ratio
        margin_h = height * margin_ratio
        
        crop_x1 = max(0, x1 - margin_w)
        crop_y1 = max(0, y1 - margin_h)
        crop_x2 = min(img_width, x2 + margin_w)
        crop_y2 = min(img_height, y2 + margin_h)
        
        #
        figure_crop = img_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        
        #
        with torch.no_grad():
            image_input = clip_preprocess(figure_crop).unsqueeze(0).to(device)
            image_features = clip_model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
    except Exception as e:
        print(f"Warning: Failed to encode figure image: {e}")
        return []
    
    #
    all_caption_ids = set()
    for b in all_blocks:
        if b.get("type") in ["figure_caption", "table_caption"]:
            all_caption_ids.add(b["block_id"])
    
    #
    candidates = []
    
    for block in all_blocks:
        #
        block_type = block.get("type", "")
        if block_type != "text":
            continue
        
        #
        block_page = block["page"]
        if abs(block_page - figure_page) > page_range:
            continue
        
        #
        text = block.get("text", "").strip()
        if len(text) <= min_text_length:
            continue
        
        #
        confidence = block.get("rec_conf", 0.0)
        if confidence < min_confidence:
            continue
        
        #
        if caption_block and block["block_id"] == caption_block["block_id"]:
            continue
        if title_block and block["block_id"] == title_block["block_id"]:
            continue
        
        #
        if block["block_id"] in all_caption_ids:
            continue
        
        try:
            #
            with torch.no_grad():
                text_tokens = clip_tokenizer([text]).to(device)
                text_features = clip_model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            #
            similarity = (image_features @ text_features.T).item()
            
            candidates.append({
                "block": block,
                "similarity": similarity,
                "page": block_page,
                "confidence": confidence
            })
        
        except Exception as e:
            print(f"Warning: Failed to encode text block {block['block_id']}: {e}")
            continue
    
    if not candidates:
        return []
    
    #
    candidates = [c for c in candidates if c["similarity"] > similarity_threshold]
    
    if not candidates:
        return []
    
    #
    candidates.sort(key=lambda x: -x["similarity"])
    return [c["block"] for c in candidates[:max_blocks]]


def merge_adjacent_blocks(blocks: List[Dict]) -> Tuple[List[Dict], Dict[int, List[int]]]:
    """"""
    fig_table_blocks = [b for b in blocks if b.get("type") in ["figure", "table"]]
    
    #
    by_page = {}
    for b in fig_table_blocks:
        page = b["page"]
        if page not in by_page:
            by_page[page] = []
        by_page[page].append(b)
    
    merged_blocks = []
    merge_map = {}  #
    processed = set()  #
    
    for page, page_blocks in by_page.items():
        #
        page_blocks.sort(key=lambda b: b["bbox"][1])
        
        i = 0
        while i < len(page_blocks):
            if page_blocks[i]["block_id"] in processed:
                i += 1
                continue
            
            #
            current = page_blocks[i]
            merge_group = [current]
            processed.add(current["block_id"])
            
            #
            j = i + 1
            while j < len(page_blocks):
                if page_blocks[j]["block_id"] in processed:
                    j += 1
                    continue
                
                prev = merge_group[-1]
                next_block = page_blocks[j]
                
                #
                #
                if prev["type"] != next_block["type"]:
                    break
                
                #
                gap = next_block["bbox"][1] - prev["bbox"][3]
                if gap >= 20:
                    break
                
                #
                x1_left, x1_right = prev["bbox"][0], prev["bbox"][2]
                x2_left, x2_right = next_block["bbox"][0], next_block["bbox"][2]
                x_overlap = min(x1_right, x2_right) - max(x1_left, x2_left)
                min_width = min(x1_right - x1_left, x2_right - x2_left)
                overlap_ratio = x_overlap / min_width if min_width > 0 else 0
                
                if overlap_ratio <= 0.8:
                    break
                
                #
                merge_group.append(next_block)
                processed.add(next_block["block_id"])
                j += 1
            
            #
            if len(merge_group) == 1:
                #
                merged_blocks.append(current)
                merge_map[current["block_id"]] = [current["block_id"]]
            else:
                #
                #
                merged_id = merge_group[0]["block_id"]
                
                #
                all_x = []
                all_y = []
                for b in merge_group:
                    all_x.extend([b["bbox"][0], b["bbox"][2]])
                    all_y.extend([b["bbox"][1], b["bbox"][3]])
                
                merged_bbox = [min(all_x), min(all_y), max(all_x), max(all_y)]
                
                #
                merged_block = {
                    "block_id": merged_id,
                    "type": current["type"],
                    "page": current["page"],
                    "bbox": merged_bbox,
                    "text": "",  #
                    "is_merged": True,
                    "merged_from": [b["block_id"] for b in merge_group]
                }
                
                merged_blocks.append(merged_block)
                merge_map[merged_id] = [b["block_id"] for b in merge_group]
                
                print(f"   Merged {len(merge_group)} adjacent {current['type']} blocks on page {page}: "
                      f"{[b['block_id'] for b in merge_group]} -> {merged_id}")
            
            i = j if j > i + 1 else i + 1
    
    return merged_blocks, merge_map


def build_visual_groups(ocr_data: Dict, use_semantic_matching: bool = True, image_dir: Optional[str] = None) -> List[Dict]:
    """"""
    blocks = ocr_data.get("blocks", [])
    
    print("\nPreprocessing: merging adjacent figure/table blocks...")
    figure_blocks, merge_map = merge_adjacent_blocks(blocks)
    
    print("\nBuilding visual groups")
    print(f"   Total blocks: {len(blocks)}")
    print(f"   Figure/Table blocks: {len(figure_blocks)}")
    
    image_paths = get_page_image_paths(ocr_data, image_dir=image_dir)
    print(f"   Page images found: {len(image_paths)}")
    
    clip_model = None
    clip_preprocess = None
    clip_tokenizer = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if use_semantic_matching and figure_blocks:
        print("   Loading CLIP model for image-text matching...")
        try:
            clip_model, clip_preprocess, clip_tokenizer = load_clip_model(device=device)
            print(f"   CLIP model loaded on {device}")
        except Exception as e:
            print(f"   Failed to load CLIP model: {e}")
            print("   Will skip related text matching")
    
    visual_groups = []
    
    for idx, figure_block in enumerate(figure_blocks):
        merged_block_ids = merge_map.get(figure_block["block_id"], [figure_block["block_id"]])
        
        unit = {
            "unit_id": idx,
            "unit_type": f"{figure_block['type']}_unit",
            "anchor_block": figure_block,
            "caption_block": None,
            "context_title": None,
            "related_texts": [],
            "all_blocks": merged_block_ids.copy(),
            "page_span": [figure_block["page"]],
            "is_merged": figure_block.get("is_merged", False),
            "merged_from": figure_block.get("merged_from", None)
        }
        
        caption = find_caption_for_figure(figure_block, blocks)
        if caption:
            unit["caption_block"] = caption
            unit["all_blocks"].append(caption["block_id"])
            if caption["page"] not in unit["page_span"]:
                unit["page_span"].append(caption["page"])
        
        title = find_nearest_title(figure_block, blocks)
        if title:
            unit["context_title"] = title
            unit["all_blocks"].append(title["block_id"])
            if title["page"] not in unit["page_span"]:
                unit["page_span"].append(title["page"])
        
        related_texts = find_related_text_blocks(
            figure_block, caption, title, blocks, image_paths,
            clip_model, clip_preprocess, clip_tokenizer, device
        )
        if related_texts:
            unit["related_texts"] = related_texts
            for text_block in related_texts:
                unit["all_blocks"].append(text_block["block_id"])
                if text_block["page"] not in unit["page_span"]:
                    unit["page_span"].append(text_block["page"])
        
        unit["statistics"] = {
            "has_caption": caption is not None,
            "has_title": title is not None,
            "num_related_texts": len(related_texts),
            "total_blocks": len(unit["all_blocks"]),
            "page_span_size": len(unit["page_span"]),
            "is_cross_page": len(unit["page_span"]) > 1,
            "is_merged_block": figure_block.get("is_merged", False),
            "num_merged": len(merged_block_ids) if figure_block.get("is_merged", False) else 1
        }
        
        visual_groups.append(unit)
    
    print(f"\nBuilt {len(visual_groups)} visual groups")
    
    has_caption = sum(1 for u in visual_groups if u["statistics"]["has_caption"])
    has_title = sum(1 for u in visual_groups if u["statistics"]["has_title"])
    cross_page = sum(1 for u in visual_groups if u["statistics"]["is_cross_page"])
    merged_groups = sum(1 for u in visual_groups if u["statistics"]["is_merged_block"])

    print("\nStatistics:")
    if visual_groups:
        avg_texts = np.mean([u["statistics"]["num_related_texts"] for u in visual_groups])
        avg_blocks = np.mean([u["statistics"]["total_blocks"] for u in visual_groups])
        n = len(visual_groups)
        print(f"   Groups with caption: {has_caption}/{n} ({has_caption/n*100:.1f}%)")
        print(f"   Groups with title context: {has_title}/{n} ({has_title/n*100:.1f}%)")
        print(f"   Cross-page groups: {cross_page}/{n} ({cross_page/n*100:.1f}%)")
        print(f"   Merged groups (adjacent blocks): {merged_groups}/{n} ({merged_groups/n*100:.1f}%)")
        print(f"   Avg related texts per group: {avg_texts:.1f}")
        print(f"   Avg blocks per group: {avg_blocks:.1f}")
    else:
        print("   (no figure/table groups found in this document)")
    
    return visual_groups

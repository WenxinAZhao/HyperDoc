#!/usr/bin/env python3
"""Budget-aware optimizer K derivation utilities.

These helpers run before prediction. They profile offline hypergraph candidates
against the active VLM processor, then translate the backbone context window
into an optimizer-side k_max.
"""

import json
import math
import os
import re
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

# Budget probing runs on CPU and may import scipy/BLAS through transformers.
# Keep launch/probe jobs from spawning many threads on shared/login nodes.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from PIL import Image


VISUAL_BLOCK_TYPES = {"figure", "table", "chart", "map", "diagram", "graph"}
REPO_ROOT = Path(__file__).resolve().parents[1]


def _deep_get(mapping: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    cur = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def load_model_context_window(model_path: Optional[str]) -> Optional[int]:
    """Read the local backbone's maximum text-position window."""
    if not model_path:
        return None
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = (
        _deep_get(data, ["text_config", "max_position_embeddings"])
        or data.get("max_position_embeddings")
        or _deep_get(data, ["llm_config", "max_position_embeddings"])
    )
    try:
        return int(value) if value else None
    except Exception:
        return None


def load_budget_tokenizer(model_path: Optional[str]):
    if not model_path:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception:
        return None


def load_budget_processor(model_path: Optional[str]):
    if not model_path:
        return None
    try:
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception:
        return None


def estimate_text_tokens(text: Any, tokenizer=None) -> int:
    """Estimate prompt text tokens, preferring the active backbone tokenizer."""
    if not text:
        return 0
    if tokenizer is not None:
        try:
            return int(len(tokenizer.encode(str(text), add_special_tokens=False)))
        except Exception:
            pass

    text = str(text)
    chars = len(text)
    words = len(re.findall(r"\w+", text))
    return int(math.ceil(max(chars / 4.0, words * 1.3)))


def estimate_visual_tokens_per_image(
    model_path: Optional[str],
    qwen_vl_max_pixels: Optional[int] = None,
) -> Tuple[int, Dict[str, Any]]:
    """Estimate a worst-case image token budget from preprocessor_config."""
    preproc_path = Path(model_path or "") / "preprocessor_config.json"
    try:
        preproc = json.loads(preproc_path.read_text(encoding="utf-8"))
    except Exception:
        preproc = {}

    patch = int(preproc.get("patch_size") or 14)
    merge = int(preproc.get("merge_size") or 2)
    min_pixels = preproc.get("min_pixels") or _deep_get(preproc, ["size", "shortest_edge"])
    max_pixels = (
        int(qwen_vl_max_pixels)
        if qwen_vl_max_pixels is not None
        else preproc.get("max_pixels")
        or _deep_get(preproc, ["size", "longest_edge"])
    )
    if not max_pixels:
        return 4000, {"source": "fallback", "patch_size": patch, "merge_size": merge}

    visual_tokens = int(math.ceil(float(max_pixels) / float((patch * merge) ** 2)))
    return max(1, visual_tokens), {
        "source": "preprocessor_config",
        "max_pixels": int(max_pixels),
        "min_pixels": int(min_pixels) if min_pixels else None,
        "patch_size": patch,
        "merge_size": merge,
    }


def _resize_to_max_pixels(image: Image.Image, max_pixels: Optional[int]) -> Image.Image:
    if not max_pixels:
        return image
    pixels = image.width * image.height
    if pixels <= int(max_pixels):
        return image
    scale = (float(max_pixels) / float(max(pixels, 1))) ** 0.5
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def visual_tokens_from_processor(processor, image: Image.Image) -> Tuple[Optional[int], Optional[Any]]:
    """Use AutoProcessor image_grid_thw as the authoritative visual token count."""
    if processor is None:
        return None, None
    try:
        inputs = processor(text=[""], images=[image], return_tensors="pt")
        grid = inputs.get("image_grid_thw")
        if grid is None:
            return None, None
        grid_cpu = grid.detach().cpu()
        merge_size = int(getattr(getattr(processor, "image_processor", None), "merge_size", 1) or 1)
        merge_length = max(1, merge_size**2)
        visual_tokens = int(sum(int(row.prod().item()) // merge_length for row in grid_cpu))
        return max(1, visual_tokens), grid_cpu.tolist()
    except Exception:
        return None, None


def estimate_visual_tokens_for_bbox(
    bbox: Any,
    visual_meta: Dict[str, Any],
    fallback_visual_tokens: int,
    image_size: Optional[Tuple[int, int]] = None,
    crop_padding: int = 10,
) -> int:
    """Patch/merge fallback for one crop, matching convert's padded bbox."""
    if not bbox or len(bbox) < 4:
        return int(fallback_visual_tokens)
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        x1 -= float(crop_padding)
        y1 -= float(crop_padding)
        x2 += float(crop_padding)
        y2 += float(crop_padding)
        if image_size:
            width_px, height_px = image_size
            x1 = max(0.0, x1)
            y1 = max(0.0, y1)
            x2 = min(float(width_px), x2)
            y2 = min(float(height_px), y2)
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
    except Exception:
        return int(fallback_visual_tokens)

    area = width * height
    max_pixels = visual_meta.get("max_pixels")
    min_pixels = visual_meta.get("min_pixels")
    if max_pixels and area > float(max_pixels):
        scale = (float(max_pixels) / area) ** 0.5
        width *= scale
        height *= scale
    if min_pixels and area < float(min_pixels):
        scale = (float(min_pixels) / max(area, 1.0)) ** 0.5
        width *= scale
        height *= scale

    patch = int(visual_meta.get("patch_size") or 14)
    merge = int(visual_meta.get("merge_size") or 2)
    grid_w = int(math.ceil(width / float(patch)))
    grid_h = int(math.ceil(height / float(patch)))
    return max(1, int(math.ceil((grid_w * grid_h) / float(merge**2))))


def _is_visual_block(block: Dict[str, Any], member_id: Any) -> bool:
    block_type = str(block.get("type") or "").lower()
    return block_type in VISUAL_BLOCK_TYPES or "anchor" in str(member_id)


@lru_cache(maxsize=20000)
def _resolve_page_image_path(path: str, doc_id: str, page_num: str) -> Optional[str]:
    if path and Path(path).exists():
        return path
    if not doc_id or page_num == "":
        return path or None
    fallback_name = f"{doc_id}_{page_num}.png"
    tmp_dir = REPO_ROOT / "tmp"
    if tmp_dir.exists():
        for candidate in tmp_dir.glob(f"*/{fallback_name}"):
            if candidate.exists():
                return str(candidate)
    return path or None


def _page_image_path_for_block(
    block: Dict[str, Any],
    pages_by_num: Dict[int, Dict[str, Any]],
) -> Optional[str]:
    page_idx = block.get("page")
    try:
        page = pages_by_num.get(int(page_idx))
    except Exception:
        page = None
    if not page:
        return None
    path = page.get("page_image_path")
    doc_id = page.get("doc_id")
    page_num = page.get("page_num")
    return _resolve_page_image_path(str(path or ""), str(doc_id or ""), str(page_num if page_num is not None else ""))


def _crop_block_image(
    block: Dict[str, Any],
    pages_by_num: Dict[int, Dict[str, Any]],
    crop_padding: int,
    qwen_vl_max_pixels: Optional[int],
) -> Tuple[Optional[Image.Image], Optional[Tuple[int, int]]]:
    image_path = _page_image_path_for_block(block, pages_by_num)
    bbox = block.get("bbox")
    if not image_path or not bbox or len(bbox) < 4 or not Path(image_path).exists():
        return None, None
    try:
        with Image.open(image_path) as page_img:
            page_img = page_img.convert("RGB")
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
            x1 = max(0, x1 - crop_padding)
            y1 = max(0, y1 - crop_padding)
            x2 = min(page_img.width, x2 + crop_padding)
            y2 = min(page_img.height, y2 + crop_padding)
            if x2 <= x1 or y2 <= y1:
                return None, page_img.size
            crop = page_img.crop((x1, y1, x2, y2))
            return _resize_to_max_pixels(crop, qwen_vl_max_pixels), page_img.size
    except Exception:
        return None, None


def _visual_cost_for_block(
    block: Dict[str, Any],
    member_id: Any,
    pages_by_num: Dict[int, Dict[str, Any]],
    processor,
    visual_meta: Dict[str, Any],
    fallback_visual_tokens: int,
    crop_padding: int,
    qwen_vl_max_pixels: Optional[int],
    visual_cache: Dict[Any, Dict[str, Any]],
    disk_visual_cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if processor is None:
        return {
            "tokens": estimate_visual_tokens_for_bbox(
                block.get("bbox"),
                visual_meta,
                fallback_visual_tokens,
                image_size=None,
                crop_padding=crop_padding,
            ),
            "source": "bbox_estimate",
            "image_grid_thw": None,
            "member_id": member_id,
        }

    bbox = tuple(block.get("bbox") or ())
    image_path = _page_image_path_for_block(block, pages_by_num)
    image_mtime = None
    if image_path and Path(image_path).exists():
        try:
            image_mtime = int(Path(image_path).stat().st_mtime)
        except Exception:
            image_mtime = None
    cache_key = (image_path, image_mtime, bbox, crop_padding, qwen_vl_max_pixels)
    disk_key = json.dumps(cache_key, sort_keys=True, default=str)
    if cache_key in visual_cache:
        cached = visual_cache[cache_key]
        if processor is None or cached.get("source") == "processor":
            return cached
    if disk_visual_cache is not None and disk_key in disk_visual_cache:
        result = dict(disk_visual_cache[disk_key])
        if processor is None or result.get("source") == "processor":
            visual_cache[cache_key] = result
            return result

    crop, image_size = _crop_block_image(block, pages_by_num, crop_padding, qwen_vl_max_pixels)
    tokens = None
    grid = None
    source = "bbox_estimate"
    if crop is not None and processor is not None:
        tokens, grid = visual_tokens_from_processor(processor, crop)
        if tokens is not None:
            source = "processor"

    if tokens is None:
        tokens = estimate_visual_tokens_for_bbox(
            block.get("bbox"),
            visual_meta,
            fallback_visual_tokens,
            image_size=image_size,
            crop_padding=crop_padding,
        )

    result = {
        "tokens": int(tokens),
        "source": source,
        "image_grid_thw": grid,
        "member_id": member_id,
    }
    visual_cache[cache_key] = result
    if disk_visual_cache is not None:
        disk_visual_cache[disk_key] = result
    return result


def _visual_cache_keys(
    block: Dict[str, Any],
    pages_by_num: Dict[int, Dict[str, Any]],
    crop_padding: int,
    qwen_vl_max_pixels: Optional[int],
) -> Tuple[Any, str]:
    bbox = tuple(block.get("bbox") or ())
    image_path = _page_image_path_for_block(block, pages_by_num)
    image_mtime = None
    if image_path and Path(image_path).exists():
        try:
            image_mtime = int(Path(image_path).stat().st_mtime)
        except Exception:
            image_mtime = None
    cache_key = (image_path, image_mtime, bbox, crop_padding, qwen_vl_max_pixels)
    disk_key = json.dumps(cache_key, sort_keys=True, default=str)
    return cache_key, disk_key


def _prefill_visual_costs_for_graph(
    visual_blocks: Iterable[Tuple[Any, Dict[str, Any]]],
    pages_by_num: Dict[int, Dict[str, Any]],
    processor,
    visual_meta: Dict[str, Any],
    fallback_visual_tokens: int,
    crop_padding: int,
    qwen_vl_max_pixels: Optional[int],
    visual_cache: Dict[Any, Dict[str, Any]],
    disk_visual_cache: Optional[Dict[str, Dict[str, Any]]],
    batch_size: int,
) -> None:
    if processor is None:
        return

    pending = []
    for member_id, block in visual_blocks:
        cache_key, disk_key = _visual_cache_keys(block, pages_by_num, crop_padding, qwen_vl_max_pixels)
        if cache_key in visual_cache:
            continue
        if disk_visual_cache is not None and disk_key in disk_visual_cache:
            visual_cache[cache_key] = dict(disk_visual_cache[disk_key])
            continue
        crop, image_size = _crop_block_image(block, pages_by_num, crop_padding, qwen_vl_max_pixels)
        if crop is None:
            continue
        pending.append((member_id, block, crop, image_size, cache_key, disk_key))
        if len(pending) >= batch_size:
            _process_visual_batch(
                pending,
                processor,
                visual_meta,
                fallback_visual_tokens,
                crop_padding,
                visual_cache,
                disk_visual_cache,
            )
            pending = []

    if pending:
        _process_visual_batch(
            pending,
            processor,
            visual_meta,
            fallback_visual_tokens,
            crop_padding,
            visual_cache,
            disk_visual_cache,
        )


def _process_visual_batch(
    pending,
    processor,
    visual_meta: Dict[str, Any],
    fallback_visual_tokens: int,
    crop_padding: int,
    visual_cache: Dict[Any, Dict[str, Any]],
    disk_visual_cache: Optional[Dict[str, Dict[str, Any]]],
) -> None:
    images = [item[2] for item in pending]
    try:
        inputs = processor(text=[""] * len(images), images=images, return_tensors="pt", padding=True)
        grid = inputs.get("image_grid_thw")
        if grid is None:
            raise ValueError("processor did not return image_grid_thw")
        grid_cpu = grid.detach().cpu()
        merge_size = int(getattr(getattr(processor, "image_processor", None), "merge_size", 1) or 1)
        merge_length = max(1, merge_size**2)
        for item, row in zip(pending, grid_cpu):
            member_id, block, _crop, image_size, cache_key, disk_key = item
            tokens = max(1, int(row.prod().item()) // merge_length)
            result = {
                "tokens": int(tokens),
                "source": "processor",
                "image_grid_thw": [row.tolist()],
                "member_id": member_id,
            }
            visual_cache[cache_key] = result
            if disk_visual_cache is not None:
                disk_visual_cache[disk_key] = result
    except Exception:
        for member_id, block, _crop, image_size, cache_key, disk_key in pending:
            tokens = estimate_visual_tokens_for_bbox(
                block.get("bbox"),
                visual_meta,
                fallback_visual_tokens,
                image_size=image_size,
                crop_padding=crop_padding,
            )
            result = {
                "tokens": int(tokens),
                "source": "bbox_estimate",
                "image_grid_thw": None,
                "member_id": member_id,
            }
            visual_cache[cache_key] = result
            if disk_visual_cache is not None:
                disk_visual_cache[disk_key] = result


def _default_profile_cache_path(
    model_path: Optional[str],
    hypergraph_dirs: Any,
    crop_padding: int,
    qwen_vl_max_pixels: Optional[int],
) -> Path:
    if isinstance(hypergraph_dirs, (str, Path)):
        dirs = [str(hypergraph_dirs)]
    else:
        dirs = [str(p) for p in (hypergraph_dirs or [])]
    payload = {
        "model_path": str(model_path or ""),
        "hypergraph_dirs": dirs,
        "crop_padding": crop_padding,
        "qwen_vl_max_pixels": qwen_vl_max_pixels,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return Path("results") / "budget_profiles" / f"visual_tokens_{digest}.json"


def _load_json_cache(cache_path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if not cache_path or not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_json_cache(cache_path: Optional[Path], data: Dict[str, Dict[str, Any]]) -> None:
    if not cache_path:
        return
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def profile_offline_hyperedge_cost(
    hypergraph_dirs: Any,
    visual_tokens_per_image: int,
    visual_meta: Optional[Dict[str, Any]] = None,
    tokenizer=None,
    processor=None,
    stat: str = "max",
    crop_padding: int = 10,
    qwen_vl_max_pixels: Optional[int] = None,
    profile_cache_path: Optional[Path] = None,
    processor_batch_size: int = 32,
) -> Optional[Dict[str, Any]]:
    """Profile contextual hyperedge costs from offline hypergraph JSON files."""
    if isinstance(hypergraph_dirs, (str, Path)):
        hypergraph_dirs = [hypergraph_dirs]

    costs = []
    max_item = None
    scanned_dirs = []
    visual_cache: Dict[Any, Dict[str, Any]] = {}
    disk_visual_cache = _load_json_cache(profile_cache_path)
    loaded_disk_cache_entries = len(disk_visual_cache)
    text_cache: Dict[Any, int] = {}
    processor_costs = 0
    fallback_visual_costs = 0

    for hypergraph_dir in hypergraph_dirs or []:
        hg_dir = Path(hypergraph_dir)
        if not hg_dir.exists():
            continue
        scanned_dirs.append(str(hg_dir))
        for graph_path in hg_dir.glob("*_hypergraph.json"):
            try:
                graph = json.loads(graph_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            blocks = {b.get("block_id"): b for b in graph.get("blocks", []) if b.get("block_id")}
            pages_by_num = {}
            for page in graph.get("pages", []) or []:
                try:
                    pages_by_num[int(page.get("page_num"))] = page
                except Exception:
                    continue

            visual_member_ids = []
            seen_visual_member_ids = set()
            contextual_edges = []
            for edge in graph.get("hyperedges", []) or []:
                if edge.get("edge_type") != "contextual":
                    continue
                if (edge.get("meta") or {}).get("relation") != "contextual":
                    continue
                contextual_edges.append(edge)
                for member_id in edge.get("members", []) or []:
                    block = blocks.get(member_id) or {}
                    if member_id in seen_visual_member_ids or not _is_visual_block(block, member_id):
                        continue
                    seen_visual_member_ids.add(member_id)
                    visual_member_ids.append((member_id, block))

            _prefill_visual_costs_for_graph(
                visual_member_ids,
                pages_by_num,
                processor,
                visual_meta or {},
                visual_tokens_per_image,
                crop_padding,
                qwen_vl_max_pixels,
                visual_cache,
                disk_visual_cache,
                max(1, int(processor_batch_size)),
            )

            for edge in contextual_edges:

                visual_count = 0
                visual_tokens = 0
                text_tokens = 0
                visual_sources = {"processor": 0, "bbox_estimate": 0}
                max_visual_grid = None
                members = edge.get("members", []) or []

                for member_id in members:
                    block = blocks.get(member_id) or {}
                    if _is_visual_block(block, member_id):
                        visual_count += 1
                        vinfo = _visual_cost_for_block(
                            block,
                            member_id,
                            pages_by_num,
                            processor,
                            visual_meta or {},
                            visual_tokens_per_image,
                            crop_padding,
                            qwen_vl_max_pixels,
                            visual_cache,
                            disk_visual_cache,
                        )
                        visual_tokens += int(vinfo["tokens"])
                        visual_sources[vinfo["source"]] = visual_sources.get(vinfo["source"], 0) + 1
                        if vinfo["source"] == "processor":
                            processor_costs += 1
                        else:
                            fallback_visual_costs += 1
                        if vinfo.get("image_grid_thw"):
                            max_visual_grid = vinfo["image_grid_thw"]
                    else:
                        cache_key = block.get("block_id") or (graph_path.name, member_id)
                        if cache_key not in text_cache:
                            text_cache[cache_key] = estimate_text_tokens(block.get("text") or "", tokenizer=tokenizer)
                        text_tokens += text_cache[cache_key]

                cost = int(visual_tokens + text_tokens)
                if cost <= 0:
                    continue
                costs.append(cost)
                if max_item is None or cost > max_item["cost_tokens"]:
                    max_item = {
                        "cost_tokens": cost,
                        "hypergraph_dir": str(hg_dir),
                        "doc_file": graph_path.name,
                        "edge_id": edge.get("edge_id"),
                        "visual_count": visual_count,
                        "visual_tokens": visual_tokens,
                        "text_tokens": text_tokens,
                        "member_count": len(members),
                        "visual_sources": visual_sources,
                        "sample_image_grid_thw": max_visual_grid,
                    }

    if not costs:
        _save_json_cache(profile_cache_path, disk_visual_cache)
        return None

    costs.sort()
    _save_json_cache(profile_cache_path, disk_visual_cache)

    def percentile(q: float) -> int:
        idx = min(len(costs) - 1, max(0, int(math.ceil((q / 100.0) * len(costs))) - 1))
        return costs[idx]

    stat_norm = str(stat or "max").lower()
    if stat_norm in {"avg", "mean"}:
        proxy = sum(costs) / float(len(costs))
        stat_norm = "avg"
    elif stat_norm in {"p95", "95"}:
        proxy = percentile(95)
        stat_norm = "p95"
    elif stat_norm in {"p99", "99"}:
        proxy = percentile(99)
        stat_norm = "p99"
    else:
        proxy = costs[-1]
        stat_norm = "max"

    return {
        "stat": stat_norm,
        "cost_proxy_tokens": int(math.ceil(proxy)),
        "hypergraph_dirs": scanned_dirs,
        "num_dirs": len(scanned_dirs),
        "count": len(costs),
        "avg": round(sum(costs) / float(len(costs)), 2),
        "p50": percentile(50),
        "p90": percentile(90),
        "p95": percentile(95),
        "p99": percentile(99),
        "max": costs[-1],
        "max_item": max_item,
        "processor_visual_costs": processor_costs,
        "fallback_visual_costs": fallback_visual_costs,
        "unique_visual_crops": len(visual_cache),
        "unique_text_blocks": len(text_cache),
        "loaded_visual_cache_entries": loaded_disk_cache_entries,
        "profile_cache_path": str(profile_cache_path) if profile_cache_path else None,
    }


def remeasure_max_item_with_processor(
    max_item: Dict[str, Any],
    visual_tokens_per_image: int,
    visual_meta: Optional[Dict[str, Any]] = None,
    tokenizer=None,
    processor=None,
    crop_padding: int = 10,
    qwen_vl_max_pixels: Optional[int] = None,
    profile_cache_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Recompute only the max-cost hyperedge with the active AutoProcessor."""
    if not max_item or processor is None:
        return None

    graph_path = Path(max_item.get("hypergraph_dir", "")) / str(max_item.get("doc_file", ""))
    edge_id = max_item.get("edge_id")
    if not graph_path.exists() or not edge_id:
        return None

    try:
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    edge = next((e for e in graph.get("hyperedges", []) or [] if e.get("edge_id") == edge_id), None)
    if edge is None:
        return None

    blocks = {b.get("block_id"): b for b in graph.get("blocks", []) if b.get("block_id")}
    pages_by_num = {}
    for page in graph.get("pages", []) or []:
        try:
            pages_by_num[int(page.get("page_num"))] = page
        except Exception:
            continue

    visual_cache: Dict[Any, Dict[str, Any]] = {}
    disk_visual_cache = _load_json_cache(profile_cache_path)
    visual_count = 0
    visual_tokens = 0
    text_tokens = 0
    visual_sources = {"processor": 0, "bbox_estimate": 0}
    missing_visual_images = 0
    max_visual_grid = None
    members = edge.get("members", []) or []

    for member_id in members:
        block = blocks.get(member_id) or {}
        if _is_visual_block(block, member_id):
            visual_count += 1
            image_path = _page_image_path_for_block(block, pages_by_num)
            if not image_path or not Path(image_path).exists():
                missing_visual_images += 1
            vinfo = _visual_cost_for_block(
                block,
                member_id,
                pages_by_num,
                processor,
                visual_meta or {},
                visual_tokens_per_image,
                crop_padding,
                qwen_vl_max_pixels,
                visual_cache,
                disk_visual_cache,
            )
            visual_tokens += int(vinfo["tokens"])
            visual_sources[vinfo["source"]] = visual_sources.get(vinfo["source"], 0) + 1
            if vinfo.get("image_grid_thw"):
                max_visual_grid = vinfo["image_grid_thw"]
        else:
            text_tokens += estimate_text_tokens(block.get("text") or "", tokenizer=tokenizer)

    _save_json_cache(profile_cache_path, disk_visual_cache)
    status = "processor_exact"
    if missing_visual_images:
        status = "missing_image_fallback"
    elif visual_sources.get("processor", 0) == 0 and visual_count:
        status = "processor_failed_fallback"
    return {
        **max_item,
        "cost_tokens": int(visual_tokens + text_tokens),
        "visual_count": visual_count,
        "visual_tokens": int(visual_tokens),
        "text_tokens": int(text_tokens),
        "member_count": len(members),
        "visual_sources": visual_sources,
        "missing_visual_images": missing_visual_images,
        "sample_image_grid_thw": max_visual_grid,
        "processor_remeasured": True,
        "processor_remeasure_status": status,
    }


def derive_dynamic_optimizer_k(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Derive k from backbone context window and measured offline hyperedge cost."""
    retrieval = config.get("retrieval", {})
    if not retrieval.get("dynamic_vlm_budget", False):
        return None

    model_cfg = config.get("model", {})
    model_path = model_cfg.get("path")
    hypergraph_dirs = retrieval.get("hypergraph_budget_dirs") or [retrieval.get("hypergraph_dir")]
    context_window = load_model_context_window(model_path)
    if not context_window or not hypergraph_dirs:
        return None

    qwen_vl_max_pixels = model_cfg.get("qwen_vl_max_pixels")
    visual_tokens, visual_meta = estimate_visual_tokens_per_image(model_path, qwen_vl_max_pixels)
    processor_max_pixels = (
        int(qwen_vl_max_pixels)
        if qwen_vl_max_pixels is not None
        else int(visual_meta["max_pixels"])
        if visual_meta.get("max_pixels") is not None
        else None
    )
    tokenizer = (
        load_budget_tokenizer(model_path)
        if retrieval.get("vlm_budget_use_tokenizer", True)
        else None
    )
    processor_scope = str(retrieval.get("vlm_budget_processor_scope", "disabled")).lower()
    use_processor = bool(retrieval.get("vlm_budget_use_processor", False))
    processor = load_budget_processor(model_path) if use_processor else None
    profile_cache_path = retrieval.get("vlm_budget_profile_cache")
    if profile_cache_path:
        profile_cache_path = Path(profile_cache_path)
    elif retrieval.get("vlm_budget_cache_profile", True):
        profile_cache_path = _default_profile_cache_path(
            model_path,
            hypergraph_dirs,
            int(retrieval.get("vlm_budget_crop_padding", 10)),
            int(qwen_vl_max_pixels) if qwen_vl_max_pixels is not None else None,
        )
    profile = profile_offline_hyperedge_cost(
        hypergraph_dirs,
        visual_tokens,
        visual_meta=visual_meta,
        tokenizer=tokenizer,
        processor=processor if processor_scope == "all" else None,
        stat=retrieval.get("vlm_budget_cost_stat", "max"),
        crop_padding=int(retrieval.get("vlm_budget_crop_padding", 10)),
        qwen_vl_max_pixels=int(qwen_vl_max_pixels) if qwen_vl_max_pixels is not None else None,
        profile_cache_path=profile_cache_path,
        processor_batch_size=int(retrieval.get("vlm_budget_processor_batch_size", 32)),
    )
    if not profile:
        return None

    if processor is not None and processor_scope != "all" and profile.get("max_item"):
        exact_max_item = remeasure_max_item_with_processor(
            profile["max_item"],
            visual_tokens,
            visual_meta=visual_meta,
            tokenizer=tokenizer,
            processor=processor,
            crop_padding=int(retrieval.get("vlm_budget_crop_padding", 10)),
            qwen_vl_max_pixels=processor_max_pixels,
            profile_cache_path=profile_cache_path,
        )
        if exact_max_item:
            profile["max_item"] = exact_max_item
            if profile["stat"] == "max":
                profile["cost_proxy_tokens"] = int(exact_max_item["cost_tokens"])
                profile["max"] = int(exact_max_item["cost_tokens"])
            profile["processor_visual_costs"] = int(exact_max_item["visual_sources"].get("processor", 0))
            profile["fallback_visual_costs"] = int(exact_max_item["visual_sources"].get("bbox_estimate", 0))

    ratio = float(retrieval.get("vlm_context_budget_ratio", 0.90))
    prompt_reserve = int(retrieval.get("vlm_prompt_reserve_tokens", 4096))
    max_new_tokens = int(model_cfg.get("max_new_tokens", 2048))
    usable_tokens = max(1, int(float(context_window) * ratio) - prompt_reserve - max_new_tokens)
    k = max(1, int(usable_tokens // max(1, int(profile["cost_proxy_tokens"]))))
    return {
        "k": k,
        "context_window_tokens": int(context_window),
        "context_budget_ratio": ratio,
        "prompt_reserve_tokens": prompt_reserve,
        "max_new_tokens": max_new_tokens,
        "usable_input_tokens": usable_tokens,
        "visual_tokens_per_image": visual_tokens,
        "visual_meta": visual_meta,
        "text_tokenizer_used": tokenizer is not None,
        "processor_used": processor is not None,
        "processor_scope": processor_scope if processor is not None else "disabled",
        "crop_padding": int(retrieval.get("vlm_budget_crop_padding", 10)),
        "hyperedge_cost_profile": profile,
    }

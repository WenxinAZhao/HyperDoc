#!/usr/bin/env python3
""""""

import math
import os
import cv2
import torch
import numpy as np
import re
from typing import List, Tuple, Optional, Union, Dict, Any
from PIL import Image
import networkx as nx
from paddleocr import PPStructure, PaddleOCR
import open_clip
import time
import paddle


class BlockDetector:
    """"""
    
    def __init__(
        self,
        clip_model_path: str = os.environ.get('HYPERDOC_OPENCLIP_MODEL', ''),
        #
        ocr_use_trt: bool = True,
        ocr_precision: str = 'fp16',
        ocr_rec_batch_num: int = 128,
        det_limit_side_len: int = 1280,
        det_db_box_thresh: float = 0.3,
        ocr_drop_score: float = 0.2,
        layout_use_gpu: bool = True,
        ocr_verbose: bool = False,
        ocr_prewarm: bool = False,
        #
        ocr_gpu_id: int = 0,
        clip_gpu_id: int = 0,
        layout_gpu_id: int = 0,
    ):
        """"""
        #
        try:
            if layout_use_gpu and torch.cuda.is_available():
                paddle.set_device(f'gpu:{layout_gpu_id}')
            else:
                paddle.set_device('cpu')
        except Exception:
            pass

        #
        #
        #
        self.engine_layout = PPStructure(
            show_log=False,
            layout=True,
            ocr=False,
            table=False,
            image_orientation=False,
            lang='en',
            use_gpu=layout_use_gpu
        )

        #
        #
        try:
            paddle.set_device(f'gpu:{ocr_gpu_id}')
        except Exception:
            pass
        self.ocr_engine = PaddleOCR(
            use_angle_cls=False,
            lang='en',
            use_gpu=True,
            use_tensorrt=ocr_use_trt,
            precision=ocr_precision,
            rec_batch_num=ocr_rec_batch_num,
            det_limit_side_len=det_limit_side_len,
            det_db_box_thresh=det_db_box_thresh,
            drop_score=ocr_drop_score,
            #
            gpu_mem=8000,  #
            enable_mkldnn=False,  #
        )

        #
        self.ocr_verbose = ocr_verbose
        self.ocr_prewarm = ocr_prewarm
        self.layout_use_gpu = layout_use_gpu
        self.layout_gpu_id = layout_gpu_id
        self.ocr_gpu_id = ocr_gpu_id
        self.clip_gpu_id = clip_gpu_id

        #
        if self.ocr_verbose and ocr_use_trt:
            print(f"[BlockDetector] TensorRT config: precision={ocr_precision}, batch_num={ocr_rec_batch_num}, gpu_mem=8000MB")
            #
            import os
            cache_dirs = [
                os.path.expanduser("~/.paddleocr"),
                os.path.expanduser("~/.cache/paddle"),
                "/tmp/.tensorrt_llm_cache"
            ]
            for d in cache_dirs:
                if os.path.exists(d):
                    print(f"[BlockDetector] Found cache directory: {d}")

        #
        if self.ocr_prewarm:
            try:
                self._prewarm_ocr()
            except Exception as e:
                if self.ocr_verbose:
                    print(f"[BlockDetector] OCR prewarm failed: {e}")
        
        self.clip_model_path = clip_model_path
        self.device = f"cuda:{clip_gpu_id}" if torch.cuda.is_available() else "cpu"
        self.clip_model = None
        self.preprocess = None
        self.tokenizer = None

    def _ensure_clip_model(self) -> bool:
        if self.clip_model is not None and self.preprocess is not None and self.tokenizer is not None:
            return True
        if not self.clip_model_path:
            return False
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32',
            pretrained=self.clip_model_path,
            device=self.device,
        )
        self.clip_model.eval()
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        return True
    
    def detect_blocks(self, image_path: str) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        #
        try:
            mtime = os.path.getmtime(image_path)
        except Exception:
            mtime = None
        cache_key = (image_path, mtime)
        global _DETECT_CACHE
        if _DETECT_CACHE is not None and cache_key in _DETECT_CACHE:
            if self.ocr_verbose:
                print(f"[detect_blocks] Cache hit: {image_path}")
            return _DETECT_CACHE[cache_key]
        elif self.ocr_verbose:
            print(f"[detect_blocks] Cache miss; running OCR: {image_path}")

        #
        img = cv2.imread(image_path)

        t0 = time.time()
        #
        results = self.engine_layout(img)
        t1 = time.time()

        #
        results = [r for r in results if r.get('type') not in ['header', 'footer']]

        #
        try:
            results = self.attach_text_by_page_ocr(img, results)
        except Exception as e:
            print(f"Page-level OCR failed; falling back to block-level OCR: {e}")
            results = self.attach_text_by_blockwise_ocr(img, results)
        t2 = time.time()

        if self.ocr_verbose:
            print(f"[BlockDetector] layout={t1-t0:.3f}s, ocr+merge={t2-t1:.3f}s, total={t2-t0:.3f}s")
        #
        try:
            if _DETECT_CACHE is not None:
                _DETECT_CACHE[cache_key] = (results, img)
        except Exception:
            pass
        return results, img
    
    def _extract_text_from_result(self, result: Dict[str, Any]) -> str:
        """"""
        txt = ""
        if 'text' in result and isinstance(result['text'], str):
            txt = result['text']
        elif 'res' in result and isinstance(result['res'], list):
            parts = []
            for item in result['res']:
                if isinstance(item, dict) and 'text' in item:
                    parts.append(item['text'])
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    #
                    t = item[1][0]
                    parts.append(t)
            txt = " ".join(parts).strip()
        return txt
    
    def attach_text_by_blockwise_ocr(self, img_bgr: np.ndarray, results: List[Dict[str, Any]], min_side: int = 4, sort_lines: bool = True) -> List[Dict[str, Any]]:
        """"""
        if img_bgr is None or not isinstance(img_bgr, np.ndarray):
            return results

        H, W = img_bgr.shape[:2]

        def _safe_crop(box: List[float]) -> Optional[np.ndarray]:
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x2)
            y2 = min(H, y2)
            if x2 <= x1 or y2 <= y1:
                return None
            return img_bgr[y1:y2, x1:x2]

        for r in results:
            if r.get('type') not in {'text', 'title', 'list', 'figure_caption', 'table_caption', 'caption'}:
                continue
            box = r.get('bbox')
            if not box:
                continue

            crop = _safe_crop(box)
            if crop is None or min(crop.shape[:2]) < min_side:
                continue

            try:
                ocr_out = self.ocr_engine.ocr(crop, det=True, cls=False)
            except Exception:
                ocr_out = None

            lines: List[Tuple[float, float, str]] = []
            confs: List[float] = []
            if ocr_out and isinstance(ocr_out, list) and len(ocr_out) > 0:
                for item in ocr_out[0]:
                    # item: [ [[x,y],...], (text, score) ]
                    if isinstance(item, list) and len(item) >= 2 and isinstance(item[1], (list, tuple)):
                        txt = str(item[1][0])
                        sc = float(item[1][1])
                        poly = item[0]
                        xs = [p[0] for p in poly]
                        ys = [p[1] for p in poly]
                        cx = sum(xs) / len(xs)
                        cy = sum(ys) / len(ys)
                        lines.append((cy, cx, txt))
                        confs.append(sc)

            if sort_lines and lines:
                lines.sort(key=lambda t: (round(t[0] / 8), t[1]))
            merged = " ".join([t[2] for t in lines]).strip()
            avg_conf = float(np.mean(confs)) if confs else 0.0

            if merged:
                r['text'] = merged
            r['rec_conf'] = avg_conf

        return results

    def attach_text_by_page_ocr(self, img_bgr: np.ndarray, results: List[Dict[str, Any]], sort_lines: bool = True) -> List[Dict[str, Any]]:
        """"""
        if img_bgr is None or not isinstance(img_bgr, np.ndarray):
            return results

        t_ocr0 = time.time()
        ocr_out = self.ocr_engine.ocr(img_bgr, det=True, cls=False)
        t_ocr1 = time.time()
        if not ocr_out or not isinstance(ocr_out, list) or len(ocr_out) == 0:
            return results

        page_items = ocr_out[0]
        if self.ocr_verbose:
            print(f"[attach_text_by_page_ocr] page_ocr_time={t_ocr1-t_ocr0:.3f}s, text_lines={len(page_items)}")

        #
        lines_meta: List[Tuple[float, float, str, float]] = []  # (cy, cx, text, conf)
        for item in page_items:
            if isinstance(item, list) and len(item) >= 2 and isinstance(item[1], (list, tuple)):
                txt = str(item[1][0])
                sc = float(item[1][1])
                poly = item[0]
                if not poly:
                    continue
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                cx = sum(xs) / len(xs)
                cy = sum(ys) / len(ys)
                lines_meta.append((cy, cx, txt, sc))

        if not lines_meta:
            return results

        #
        for r in results:
            if r.get('type') not in {'text', 'title', 'list', 'figure_caption', 'table_caption', 'caption'}:
                continue
            box = r.get('bbox')
            if not box:
                continue
            x1, y1, x2, y2 = box

            assigned: List[Tuple[float, float, str, float]] = []
            for cy, cx, txt, sc in lines_meta:
                if (x1 <= cx <= x2) and (y1 <= cy <= y2):
                    assigned.append((cy, cx, txt, sc))

            if not assigned:
                continue

            if sort_lines:
                assigned.sort(key=lambda t: (round(t[0] / 8), t[1]))
            merged = " ".join([t[2] for t in assigned]).strip()
            avg_conf = float(np.mean([t[3] for t in assigned])) if assigned else 0.0

            if merged:
                r['text'] = merged
            r['rec_conf'] = avg_conf

        if self.ocr_verbose:
            print(f"[BlockDetector] page_ocr_det+rec={t_ocr1-t_ocr0:.3f}s, assign+merge done")
        return results

    def _prewarm_ocr(self):
        dummy = np.zeros((256, 256, 3), dtype=np.uint8)
        t0 = time.time()
        _ = self.ocr_engine.ocr(dummy, det=True, cls=False)
        t1 = time.time()
        if self.ocr_verbose:
            print(f"[BlockDetector] prewarm ocr: {t1-t0:.3f}s")
    
    def get_box_features(self, box: List[float]) -> Dict[str, Any]:
        """"""
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        area = width * height
        return {
            'bbox': box,
            'center': (center_x, center_y),
            'area': area,
            'width': width,
            'height': height,
            'aspect_ratio': width/height if height != 0 else 0
        }

    def compute_spatial_relation(self, box1: List[float], box2: List[float]) -> Dict[str, Any]:
        """"""
        x1a, y1a, x2a, y2a = box1
        x1b, y1b, x2b, y2b = box2
        
        #
        center_a = ((x1a + x2a)/2, (y1a + y2a)/2)
        center_b = ((x1b + x2b)/2, (y1b + y2b)/2)
        
        #
        dist = math.sqrt((center_a[0]-center_b[0])**2 + (center_a[1]-center_b[1])**2)
        
        #
        x_left = max(x1a, x1b)
        y_top = max(y1a, y1b)
        x_right = min(x2a, x2b)
        y_bottom = min(y2a, y2b)
        
        if x_right < x_left or y_bottom < y_top:
            iou = 0
        else:
            intersection = (x_right - x_left) * (y_bottom - y_top)
            area_a = (x2a - x1a) * (y2a - y1a)
            area_b = (x2b - x1b) * (y2b - y1b)
            iou = intersection / float(area_a + area_b - intersection)
        
        #
        dx = center_b[0] - center_a[0]
        dy = center_b[1] - center_a[1]
        
        #
        angle = math.atan2(dy, dx) * 180 / math.pi
        
        #
        if abs(angle) < 30:
            direction = "right"
        elif abs(angle - 180) < 30 or abs(angle + 180) < 30:
            direction = "left"
        elif abs(angle - 90) < 30:
            direction = "below"
        elif abs(angle + 90) < 30:
            direction = "above"
        else:
            direction = "near"
        
        return {
            'direction': direction,
            'distance': dist,
            'iou': iou
        }

    def extract_region_image(self, img: np.ndarray, bbox: List[float], margin_ratio: float = 0.15) -> Optional[Image.Image]:
        """"""
        x1, y1, x2, y2 = [int(x) for x in bbox]
        
        #
        width = x2 - x1
        height = y2 - y1
        margin_x = int(width * margin_ratio)
        margin_y = int(height * margin_ratio)
        
        #
        h, w = img.shape[:2]
        x1_expanded = max(0, x1 - margin_x)
        y1_expanded = max(0, y1 - margin_y)
        x2_expanded = min(w, x2 + margin_x)
        y2_expanded = min(h, y2 + margin_y)
        
        #
        region = img[y1_expanded:y2_expanded, x1_expanded:x2_expanded]
        if region.size == 0:
            return None
        
        return Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))

    @torch.no_grad()
    def get_clip_image_embedding(self, pil_img: Image.Image) -> Optional[np.ndarray]:
        """"""
        if pil_img is None:
            return None
        if not self._ensure_clip_model():
            return None
        image = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        image_features = self.clip_model.encode_image(image)
        return image_features.squeeze().cpu().numpy()

    @torch.no_grad()
    def get_clip_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """"""
        if not text:
            return None
        if not self._ensure_clip_model():
            return None
        text_tokens = self.tokenizer([text]).to(self.device)
        text_features = self.clip_model.encode_text(text_tokens)
        return text_features.squeeze().cpu().numpy()


class BlockReranker:
    """"""
    
    def __init__(self, detector: BlockDetector):
        """"""
        self.detector = detector
    
    def create_spatial_graph(self, blocks: List[Dict[str, Any]], img: np.ndarray) -> nx.DiGraph:
        """"""
        G = nx.DiGraph()
        
        #
        for idx, block in enumerate(blocks):
            box = block.get('bbox')
            if box is None:
                continue
            
            #
            node_attrs = {
                'type': block.get('type', 'unknown'),
                'text': block.get('text', ''),
                'id': idx,
                **self.detector.get_box_features(box)
            }
            
            #
            node_type = block.get('type', 'unknown')
            if node_type in ['figure', 'image', 'table']:
                #
                pil_img = self.detector.extract_region_image(img, box)
                if pil_img is not None:
                    emb = self.detector.get_clip_image_embedding(pil_img)
                    if emb is not None:
                        node_attrs['embedding'] = emb
                        node_attrs['has_visual_embedding'] = True
            elif node_type in ['text', 'title', 'list']:
                #
                text = block.get('text', '')
                if text:
                    emb = self.detector.get_clip_text_embedding(text)
                    if emb is not None:
                        node_attrs['embedding'] = emb
                        node_attrs['has_text_embedding'] = True
            
            G.add_node(idx, **node_attrs)
        
        #
        for i in G.nodes():
            for j in G.nodes():
                if i != j:
                    box1 = G.nodes[i]['bbox']
                    box2 = G.nodes[j]['bbox']
                    
                    spatial_rel = self.detector.compute_spatial_relation(box1, box2)
                    
                    #
                    if spatial_rel['distance'] < 300 or spatial_rel['iou'] > 0.05:
                        G.add_edge(i, j, type='spatial', **spatial_rel)
        
        return G
    
    def rerank_blocks_simple(
        self, 
        blocks: List[Dict[str, Any]], 
        query: str, 
        img: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """"""
        #
        query_embedding = self.detector.get_clip_text_embedding(query)
        if query_embedding is None:
            #
            return sorted(blocks, key=lambda x: self.detector.get_box_features(x['bbox'])['area'], reverse=True)[:top_k]
        
        #
        block_scores = []
        for idx, block in enumerate(blocks):
            score = 0.0
            
            #
            text = block.get('text', '')
            if text:
                text_emb = self.detector.get_clip_text_embedding(text)
                if text_emb is not None:
                    text_sim = np.dot(query_embedding, text_emb) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(text_emb)
                    )
                    score += text_sim * 0.7  #
            
            #
            if block.get('type') in ['figure', 'image', 'table']:
                bbox = block.get('bbox')
                if bbox:
                    pil_img = self.detector.extract_region_image(img, bbox)
                    if pil_img is not None:
                        img_emb = self.detector.get_clip_image_embedding(pil_img)
                        if img_emb is not None:
                            img_sim = np.dot(query_embedding, img_emb) / (
                                np.linalg.norm(query_embedding) * np.linalg.norm(img_emb)
                            )
                            score += img_sim * 0.3  #
            
            block_scores.append((idx, block, score))
        
        #
        block_scores.sort(key=lambda x: x[2], reverse=True)
        
        return [item[1] for item in block_scores[:top_k]]
    
    def rerank_blocks_with_graph(
        self, 
        blocks: List[Dict[str, Any]], 
        query: str, 
        img: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """"""
        #
        G = self.create_spatial_graph(blocks, img)
        
        if G.number_of_nodes() == 0:
            return []
        
        #
        query_embedding = self.detector.get_clip_text_embedding(query)
        if query_embedding is None:
            return self.rerank_blocks_simple(blocks, query, img, top_k)
        
        #
        node_scores = {}
        for node_id in G.nodes():
            node = G.nodes[node_id]
            score = 0.0
            
            #
            if 'embedding' in node:
                emb = node['embedding']
                sim = np.dot(query_embedding, emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(emb)
                )
                score += sim
            
            #
            #
            neighbor_scores = []
            for neighbor_id in G.neighbors(node_id):
                neighbor = G.nodes[neighbor_id]
                if 'embedding' in neighbor:
                    neighbor_emb = neighbor['embedding']
                    neighbor_sim = np.dot(query_embedding, neighbor_emb) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(neighbor_emb)
                    )
                    neighbor_scores.append(neighbor_sim)
            
            if neighbor_scores:
                score += np.mean(neighbor_scores) * 0.2  #
            
            node_scores[node_id] = score
        
        #
        sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        
        selected_blocks = []
        for node_id, score in sorted_nodes[:top_k]:
            #
            if node_id < len(blocks):
                block = blocks[node_id].copy()
                block['rerank_score'] = score
                selected_blocks.append(block)
        
        return selected_blocks

    def rerank_blocks_ppr_hybrid_singlepage(
        self,
        blocks: List[Dict[str, Any]],
        query: str,
        img: np.ndarray,
        top_k: int = 5,
        scope: str = 'global',
        alpha: float = 0.75,
        lambda_mix: Optional[float] = 0.7,
        seed_boost_topm: int = 5,
        seed_boost_each: float = 0.15,
        exclude_types_for_p: Tuple[str, ...] = ("page",),
        weight_key: str = 'weight'
    ) -> List[Dict[str, Any]]:
        """"""
        #
        G = self.create_spatial_graph(blocks, img)
        if G.number_of_nodes() == 0:
            return []

        #
        query_emb = self.detector.get_clip_text_embedding(query)
        #
        personalization: Dict[int, float] = {}
        sims: Dict[int, float] = {}
        for n in G.nodes():
            t = (G.nodes[n].get('type') or 'unknown').lower()
            emb = G.nodes[n].get('embedding')
            if emb is None:
                #
                typ = G.nodes[n].get('type')
                bbox = G.nodes[n].get('bbox')
                if typ in ['figure', 'image', 'table'] and bbox is not None:
                    pil_img = self.detector.extract_region_image(img, bbox)
                    if pil_img is not None:
                        emb = self.detector.get_clip_image_embedding(pil_img)
                elif typ in ['text', 'title', 'list']:
                    txt = G.nodes[n].get('text', '')
                    if txt:
                        emb = self.detector.get_clip_text_embedding(txt)
                if emb is not None:
                    G.nodes[n]['embedding'] = emb
            sim = 0.0
            if query_emb is not None and emb is not None:
                sim = float(np.dot(query_emb, emb) / (max(1e-12, np.linalg.norm(query_emb)) * max(1e-12, np.linalg.norm(emb))))
            sims[n] = sim
            score = max((sim + 1.0)/2.0, 1e-8)
            if t in set(tt.lower() for tt in exclude_types_for_p):
                score = 1e-8
            personalization[n] = score
        s = sum(personalization.values())
        if s > 0:
            personalization = {k: v/s for k, v in personalization.items()}

        scope_norm = (scope or 'global').lower()
        alpha_eff = float(alpha)
        lambda_eff = lambda_mix
        seed_topm_eff = int(seed_boost_topm)
        seed_each_eff = float(seed_boost_each)
        if scope_norm == 'local':
            alpha_eff = min(alpha_eff, 0.7)
            if lambda_mix is not None:
                lambda_eff = min(0.9, max(0.5, lambda_mix + 0.1))
            seed_topm_eff = max(1, seed_boost_topm // 2 or 1)
            seed_each_eff = seed_boost_each * 0.6

        #
        if seed_topm_eff > 0 and seed_each_eff > 0:
            #
            cand = [n for n in G.nodes() if (G.nodes[n].get('type') or '').lower() not in set(tt.lower() for tt in exclude_types_for_p)]
            cand.sort(key=lambda n: sims.get(n, 0.0), reverse=True)
            m = min(seed_topm_eff, len(cand))
            if m > 0:
                for n in cand[:m]:
                    personalization[n] = personalization.get(n, 0.0) + float(seed_each_eff)
                s2 = sum(personalization.values())
                if s2 > 0:
                    personalization = {k: v/s2 for k, v in personalization.items()}

        #
        try:
            pr = nx.pagerank(G, alpha=alpha_eff, personalization=personalization, weight=weight_key, tol=1e-6, max_iter=100)
        except Exception:
            return self.rerank_blocks_simple(blocks, query, img, top_k)

        #
        #
        nodes = list(G.nodes())
        pr_scores = np.array([pr.get(n, 0.0) for n in nodes], dtype=float)
        sim_scores = np.array([sims.get(n, 0.0) for n in nodes], dtype=float)
        if lambda_eff is not None:
            final_scores = float(lambda_eff) * sim_scores + (1.0 - float(lambda_eff)) * pr_scores
            order = np.argsort(-final_scores)
            pick = [(nodes[i], float(final_scores[i])) for i in order[:top_k]]
        else:
            order = np.argsort(-pr_scores)
            pick = [(nodes[i], float(pr_scores[i])) for i in order[:top_k]]

        selected: List[Dict[str, Any]] = []
        for nid, sc in pick:
            if nid < len(blocks):
                b = blocks[nid].copy()
                b['pagerank_score'] = float(pr.get(nid, 0.0))
                b['sim_score'] = float(sim_scores[list(nodes).index(nid)]) if len(nodes) > 0 else 0.0
                b['rerank_score'] = float(sc)
                selected.append(b)
        return selected

    def rerank_blocks_ppr_hybrid_global_ss(
        self,
        pages_blocks: List[List[Dict[str, Any]]],
        imgs: List[np.ndarray],
        query: str,
        top_k: int = 5,
        scope: str = 'global',
        alpha: float = 0.75,
        lambda_mix: float = 0.7,
        seed_boost_topm: int = 5,
        seed_boost_each: float = 0.15,
        exclude_types_for_p: Tuple[str, ...] = ("page",),
        weight_key: str = 'weight',
        min_edge_weight: float = 1e-6,
        #
        modality_weights: Optional[Dict[str, float]] = None,
        min_visual_ratio: Optional[float] = None,
        return_diagnostics: bool = True,
        **ss_kwargs: Any,
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
        """"""
        if not pages_blocks:
            return []
        
        #
        visual_types = {'figure', 'image', 'table'}
        total_blocks = sum(len(pb) for pb in pages_blocks)
        total_visual = sum(1 for pb in pages_blocks for b in pb if b.get('type') in visual_types)
        print("[rerank_ppr_hybrid_global_ss] Input statistics:")
        print(f"   - pages: {len(pages_blocks)}, blocks: {total_blocks}")
        print(f"   - visual blocks: {total_visual} ({100*total_visual/total_blocks if total_blocks > 0 else 0:.1f}%)")
        print(f"   - top_k: {top_k}, min_visual_ratio: {min_visual_ratio}")

        #
        G, id_map = self.build_semantic_structural_graph(
            pages_blocks, imgs, min_edge_weight=min_edge_weight, **ss_kwargs
        )
        if G.number_of_nodes() == 0:
            return []

        #
        query_emb = self.detector.get_clip_text_embedding(query)
        sims: Dict[int, float] = {}
        personalization: Dict[int, float] = {}
        excl = set(t.lower() for t in exclude_types_for_p)
        scope_norm = (scope or 'global').lower()
        alpha_eff = float(alpha)
        lambda_eff = lambda_mix
        seed_topm_eff = int(seed_boost_topm)
        seed_each_eff = float(seed_boost_each)
        if scope_norm == 'local':
            alpha_eff = min(alpha_eff, 0.7)
            lambda_eff = min(0.95, max(0.4, lambda_mix + 0.1))
            seed_topm_eff = max(1, seed_boost_topm // 2 or 1)
            seed_each_eff = seed_boost_each * 0.6

        if query_emb is None:
            #
            for n in G.nodes():
                t = (G.nodes[n].get('type') or 'unknown').lower()
                personalization[n] = 1e-8 if t in excl else 1.0
                sims[n] = 0.0
            s = sum(personalization.values())
            if s > 0:
                personalization = {k: v/s for k, v in personalization.items()}
        else:
            qn = float(np.linalg.norm(query_emb))
            for n in G.nodes():
                t = (G.nodes[n].get('type') or 'unknown').lower()
                emb = G.nodes[n].get('embedding')
                if emb is None:
                    #
                    p_idx = int(G.nodes[n].get('page_id', 1)) - 1
                    emb = self._ensure_node_embedding(G.nodes[n], imgs[p_idx] if (0 <= p_idx < len(imgs)) else None)
                    if emb is not None:
                        G.nodes[n]['embedding'] = emb
                sim = 0.0
                if emb is not None and qn > 0:
                    en = float(np.linalg.norm(emb))
                    if en > 0:
                        sim = float(np.dot(query_emb, emb) / (qn * en))
                sims[n] = sim
                score = max((sim + 1.0)/2.0, 1e-8)
                if t in excl:
                    score = 1e-8
                personalization[n] = score
            s = sum(personalization.values())
            if s > 0:
                personalization = {k: v/s for k, v in personalization.items()}
            #
            if seed_topm_eff > 0 and seed_each_eff > 0:
                cand = [n for n in G.nodes() if (G.nodes[n].get('type') or '').lower() not in excl]
                cand.sort(key=lambda n: sims.get(n, 0.0), reverse=True)
                m = min(seed_topm_eff, len(cand))
                for n in cand[:m]:
                    personalization[n] = personalization.get(n, 0.0) + float(seed_each_eff)
                s2 = sum(personalization.values())
                if s2 > 0:
                    personalization = {k: v/s2 for k, v in personalization.items()}

        # 3) PageRank
        try:
            pr = nx.pagerank(G, alpha=alpha_eff, personalization=personalization, weight=weight_key, tol=1e-6, max_iter=100)
        except Exception:
            #
            flat = []
            for p_idx, blocks in enumerate(pages_blocks):
                for b in blocks:
                    b2 = b.copy(); b2['page_id'] = p_idx + 1
                    flat.append(b2)
            #
            if query_emb is None:
                flat.sort(key=lambda x: 0.0, reverse=True)
            else:
                def _sim_block(b):
                    emb = None
                    t = (b.get('type') or 'unknown').lower()
                    if t in {'text','title','list'}:
                        txt = b.get('text','')
                        emb = self.detector.get_clip_text_embedding(txt) if txt else None
                    elif t in {'figure','image','table'}:
                        pass
                    if emb is None:
                        return 0.0
                    dn = float(np.linalg.norm(emb)); qn2 = float(np.linalg.norm(query_emb))
                    if dn <= 0 or qn2 <= 0: return 0.0
                    return float(np.dot(query_emb, emb) / (qn2 * dn))
                flat.sort(key=_sim_block, reverse=True)
            return flat[:top_k]

        #
        nodes_all = list(G.nodes())
        pr_scores = np.array([pr.get(n, 0.0) for n in nodes_all], dtype=float)
        sim_scores = np.array([sims.get(n, 0.0) for n in nodes_all], dtype=float)
        lam = float(min(max(lambda_eff, 0.0), 1.0))
        final_scores = lam * sim_scores + (1.0 - lam) * pr_scores
        order = np.argsort(-final_scores)

        #
        if modality_weights is not None:
            #
            calibrated_scores = np.zeros_like(final_scores)
            for i, n in enumerate(nodes_all):
                node_type = (G.nodes[n].get('type') or 'unknown').lower()
                weight = modality_weights.get(node_type, 1.0)
                calibrated_scores[i] = final_scores[i] * weight
            final_scores = calibrated_scores
            order = np.argsort(-final_scores)
        
        #
        selected: List[Dict[str, Any]] = []
        for idx in order[:max(0, int(top_k))]:
            n = nodes_all[idx]
            if n not in id_map:
                continue
            p_idx, b_idx = id_map[n]
            if not (0 <= p_idx < len(pages_blocks) and 0 <= b_idx < len(pages_blocks[p_idx])):
                continue
            block = pages_blocks[p_idx][b_idx].copy()
            block['page_id'] = p_idx + 1
            block['pagerank_score'] = float(pr.get(n, 0.0))
            block['sim_score'] = float(sim_scores[idx])
            block['rerank_score'] = float(final_scores[idx])
            selected.append(block)
        
        #
        if min_visual_ratio is not None and min_visual_ratio > 0:
            visual_types = {'figure', 'image', 'table'}
            before_visual = sum(1 for b in selected if b.get('type') in visual_types)
            print(f"[ensure_visual_ratio] before: {len(selected)} blocks, visual={before_visual} ({100*before_visual/len(selected) if selected else 0:.1f}%)")
            
            selected = self._ensure_visual_ratio(selected, order, nodes_all, id_map, pages_blocks, 
                                                  pr, sim_scores, final_scores, top_k, min_visual_ratio)
            
            after_visual = sum(1 for b in selected if b.get('type') in visual_types)
            print(f"[ensure_visual_ratio] after: {len(selected)} blocks, visual={after_visual} ({100*after_visual/len(selected) if selected else 0:.1f}%)")

        
        #
        diagnostics = self._compute_modality_diagnostics(
            selected, G, nodes_all, sim_scores, pr, final_scores, order, top_k
        )
        if return_diagnostics:
            #
            slim_diag = {
                'visual_ratio': diagnostics.get('visual_ratio', 0.0),
                'visual_count': diagnostics.get('visual_count', 0),
                'total_count': diagnostics.get('total_count', 0)
            }
            return selected, slim_diag
        return selected

    def rerank_blocks_pagerank(
        self,
        blocks: List[Dict[str, Any]],
        query: str,
        img: np.ndarray,
        top_k: int = 5,
        alpha: float = 0.85,
        spatial_distance_scale: float = 300.0,
        iou_weight: float = 1.0,
        use_personalization: bool = True,
        min_edge_weight: float = 1e-6,
        max_iter: int = 100,
        tol: float = 1e-06
    ) -> List[Dict[str, Any]]:
        """"""
        #
        G = self.create_spatial_graph(blocks, img)
        if G.number_of_nodes() == 0:
            return []

        #
        for u, v, data in G.edges(data=True):
            distance = float(data.get('distance', 1.0))
            iou = float(data.get('iou', 0.0))
            #
            w_dist = 1.0 / (1.0 + (distance / max(spatial_distance_scale, 1e-6)))
            #
            w_iou = iou_weight * max(iou, 0.0)
            weight = max(w_dist + w_iou, min_edge_weight)
            G[u][v]['weight'] = weight

        #
        personalization = None
        if use_personalization:
            query_emb = self.detector.get_clip_text_embedding(query)
            if query_emb is not None:
                p: Dict[int, float] = {}
                #
                qn = np.linalg.norm(query_emb)
                for node_id in G.nodes():
                    node = G.nodes[node_id]
                    node_score = 0.0
                    emb = node.get('embedding')
                    #
                    if emb is None:
                        node_type = node.get('type')
                        bbox = node.get('bbox')
                        if node_type in ['figure', 'image', 'table'] and bbox is not None:
                            pil_img = self.detector.extract_region_image(img, bbox)
                            if pil_img is not None:
                                emb = self.detector.get_clip_image_embedding(pil_img)
                        elif node_type in ['text', 'title', 'list']:
                            text = node.get('text', '')
                            if text:
                                emb = self.detector.get_clip_text_embedding(text)
                        if emb is not None:
                            G.nodes[node_id]['embedding'] = emb

                    if emb is not None:
                        en = np.linalg.norm(emb)
                        if qn > 0 and en > 0:
                            cos_sim = float(np.dot(query_emb, emb) / (qn * en))
                            #
                            node_score = max((cos_sim + 1.0) / 2.0, 0.0)
                    #
                    if node_score <= 0:
                        node_score = 1e-8
                    p[node_id] = node_score

                #
                s = sum(p.values())
                if s > 0:
                    personalization = {k: v / s for k, v in p.items()}

        #
        try:
            if personalization is not None:
                pr_scores = nx.pagerank(
                    G,
                    alpha=alpha,
                    personalization=personalization,
                    weight='weight',
                    max_iter=max_iter,
                    tol=tol,
                )
            else:
                pr_scores = nx.pagerank(
                    G,
                    alpha=alpha,
                    weight='weight',
                    max_iter=max_iter,
                    tol=tol,
                )
        except Exception:
            #
            return self.rerank_blocks_simple(blocks, query, img, top_k)

        #
        sorted_nodes = sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)
        selected_blocks: List[Dict[str, Any]] = []
        for node_id, score in sorted_nodes[:top_k]:
            if node_id < len(blocks):
                block = blocks[node_id].copy()
                block['pagerank_score'] = float(score)
                #
                block['rerank_score'] = float(score)
                selected_blocks.append(block)

        return selected_blocks

    # -------- Helper: Modality Balance & Diagnostics --------
    def _ensure_visual_ratio(
        self,
        selected: List[Dict[str, Any]],
        order: np.ndarray,
        nodes_all: List[int],
        id_map: Dict[int, Tuple[int, int]],
        pages_blocks: List[List[Dict[str, Any]]],
        pr: Dict[int, float],
        sim_scores: np.ndarray,
        final_scores: np.ndarray,
        top_k: int,
        min_visual_ratio: float
    ) -> List[Dict[str, Any]]:
        """"""
        visual_types = {'figure', 'image', 'table'}
        
        #
        current_visual_count = sum(1 for b in selected if b.get('type') in visual_types)
        current_ratio = current_visual_count / len(selected) if selected else 0.0
        
        print(f"   [_ensure_visual_ratio] current: {current_visual_count}/{len(selected)}={current_ratio:.2%}, target>={min_visual_ratio:.2%}")
        
        if current_ratio >= min_visual_ratio:
            print("   [_ensure_visual_ratio] target already satisfied")
            return selected  #
        
        #
        target_visual_count = max(1, int(top_k * min_visual_ratio))
        needed = target_visual_count - current_visual_count
        
        print(f"   [_ensure_visual_ratio] need {needed} additional visual blocks (target={target_visual_count})")
        
        if needed <= 0:
            return selected
        
        #
        selected_node_ids = set()
        for b in selected:
            #
            p_idx = b.get('page_id', 1) - 1
            for nid, (pi, bi) in id_map.items():
                if pi == p_idx and pages_blocks[pi][bi].get('bbox') == b.get('bbox'):
                    selected_node_ids.add(nid)
                    break
        
        visual_candidates = []
        for idx in order:
            n = nodes_all[idx]
            if n in selected_node_ids:
                continue
            if n not in id_map:
                continue
            p_idx, b_idx = id_map[n]
            if not (0 <= p_idx < len(pages_blocks) and 0 <= b_idx < len(pages_blocks[p_idx])):
                continue
            block = pages_blocks[p_idx][b_idx]
            if block.get('type') in visual_types:
                visual_candidates.append((n, idx, block))
                if len(visual_candidates) >= needed:
                    break
        
        print(f"   [_ensure_visual_ratio] found {len(visual_candidates)} visual candidates")
        
        #
        if visual_candidates:
            #
            non_visual_selected = [(i, b) for i, b in enumerate(selected) 
                                   if b.get('type') not in visual_types]
            non_visual_selected.sort(key=lambda x: x[1].get('rerank_score', 0))
            
            #
            to_remove = min(len(visual_candidates), len(non_visual_selected))
            remove_indices = [idx for idx, _ in non_visual_selected[:to_remove]]
            
            print(f"   [_ensure_visual_ratio] remove {to_remove} low-score non-visual blocks and add {len(visual_candidates[:to_remove])} visual blocks")
            
            #
            new_selected = [b for i, b in enumerate(selected) if i not in remove_indices]
            
            #
            for n, idx, block in visual_candidates[:to_remove]:
                new_block = block.copy()
                p_idx, b_idx = id_map[n]
                new_block['page_id'] = p_idx + 1
                new_block['pagerank_score'] = float(pr.get(n, 0.0))
                new_block['sim_score'] = float(sim_scores[idx])
                new_block['rerank_score'] = float(final_scores[idx])
                new_selected.append(new_block)
            
            #
            new_selected.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
            return new_selected[:top_k]
        
        return selected
    
    def _compute_modality_diagnostics(
        self,
        selected: List[Dict[str, Any]],
        G: nx.DiGraph,
        nodes_all: List[int],
        sim_scores: np.ndarray,
        pr: Dict[int, float],
        final_scores: np.ndarray,
        order: np.ndarray,
        top_k: int
    ) -> Dict[str, Any]:
        """"""
        
        #
        modality_dist = {}
        for b in selected:
            t = b.get('type', 'unknown')
            modality_dist[t] = modality_dist.get(t, 0) + 1
        
        #
        modality_scores = {
            'text': [], 'title': [], 'list': [],
            'figure': [], 'image': [], 'table': []
        }
        
        for i, n in enumerate(nodes_all):
            node_type = (G.nodes[n].get('type') or 'unknown').lower()
            if node_type in modality_scores:
                modality_scores[node_type].append({
                    'sim': float(sim_scores[i]),
                    'pr': float(pr.get(n, 0.0)),
                    'final': float(final_scores[i])
                })
        
        modality_avg = {}
        for mod, scores in modality_scores.items():
            if scores:
                modality_avg[mod] = {
                    'count': len(scores),
                    'avg_sim': float(np.mean([s['sim'] for s in scores])),
                    'avg_final': float(np.mean([s['final'] for s in scores]))
                }
        
        #
        top_candidates_modality = {}
        for idx in order[:50]:
            if idx >= len(nodes_all):
                break
            n = nodes_all[idx]
            t = (G.nodes[n].get('type') or 'unknown').lower()
            top_candidates_modality[t] = top_candidates_modality.get(t, 0) + 1
        
        #
        visual_types = {'figure', 'image', 'table'}
        text_types = {'text', 'title', 'list'}
        
        visual_selected = sum(modality_dist.get(t, 0) for t in visual_types)
        text_selected = sum(modality_dist.get(t, 0) for t in text_types)
        total_selected = sum(modality_dist.values())
        
        diagnostics = {
            'modality_distribution': modality_dist,
            'modality_avg_scores': modality_avg,
            'top50_modality_dist': top_candidates_modality,
            'visual_count': visual_selected,
            'text_count': text_selected,
            'total_count': total_selected,
            'visual_ratio': visual_selected / total_selected if total_selected > 0 else 0.0,
            'text_ratio': text_selected / total_selected if total_selected > 0 else 0.0
        }
        
        return diagnostics
    
    # -------- Multipage (global) PageRank --------
    def _ensure_node_embedding(self, node: Dict[str, Any], img: Optional[np.ndarray]) -> Optional[np.ndarray]:
        emb = node.get('embedding')
        if emb is not None:
            return emb
        node_type = node.get('type')
        bbox = node.get('bbox')
        if node_type in ['figure', 'image', 'table'] and img is not None and bbox is not None:
            pil_img = self.detector.extract_region_image(img, bbox)
            if pil_img is not None:
                emb = self.detector.get_clip_image_embedding(pil_img)
        elif node_type in ['text', 'title', 'list']:
            text = node.get('text', '')
            if text:
                emb = self.detector.get_clip_text_embedding(text)
        if emb is not None:
            node['embedding'] = emb
        return emb

    @staticmethod
    def _cosine_sim(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
        if a is None or b is None:
            return 0.0
        an = np.linalg.norm(a)
        bn = np.linalg.norm(b)
        if an <= 0 or bn <= 0:
            return 0.0
        return float(np.dot(a, b) / (an * bn))

    def build_multipage_graph(
        self,
        pages_blocks: List[List[Dict[str, Any]]],
        imgs: List[np.ndarray],
        connect_adjacent_only: bool = True,
        cross_top_k: int = 3,
        cross_sim_thresh: float = 0.2,
        page_gap_penalty: float = 0.3,
        spatial_distance_scale: float = 300.0,
        iou_weight: float = 1.0,
        min_edge_weight: float = 1e-6,
        # --- graph construction parameters ---
        intra_use_knn: bool = True,
        k_intra_spatial: int = 6,
        intra_weight_boost: float = 0.5,
        cross_candidate_mode: str = "adjacent",  # "adjacent" | "all" | "explicit_only" | "adjacent_or_explicit"
        cross_type_constraints: bool = True,
        cross_min_token_overlap: float = 0.0,
        cross_out_degree_cap: Optional[int] = None,
        explicit_ref_boost: float = 0.2,
        # --- page/title anchor edges ---
        add_page_nodes: bool = False,
        page_anchor_weight: float = 0.25,
        page_anchor_out_degree_cap: Optional[int] = None,
        add_title_affinity_edges: bool = True,
        title_affinity_weight: float = 0.6,
        title_horizontal_overlap_thresh: float = 0.2,
        title_max_vertical_gap_ratio: float = 0.25,
        title_affinity_out_degree_cap: Optional[int] = None,
        title_stop_at_next_title: bool = True,
        # --- caption & list local edges ---
        add_caption_edges: bool = True,
        caption_horizontal_overlap_thresh: float = 0.2,
        caption_max_vertical_gap_ratio: float = 0.2,
        add_list_local_edges: bool = True,
        list_horizontal_overlap_thresh: float = 0.2,
        list_max_vertical_gap_ratio: float = 0.1,
        # --- page chain (weak) ---
        add_page_chain_edges: bool = False,
        page_chain_weight: float = 0.05
    ) -> Tuple[nx.DiGraph, Dict[int, Tuple[int, int]]]:
        """"""
        G = nx.DiGraph()
        id_map: Dict[int, Tuple[int, int]] = {}
        global_id = 0
        page_node_ids: Dict[int, int] = {}

        #
        for p_idx, blocks in enumerate(pages_blocks):
            img = imgs[p_idx] if p_idx < len(imgs) else None
            #
            for b_idx, b in enumerate(blocks):
                box = b.get('bbox')
                if box is None:
                    continue
                node_attrs = {
                    'type': b.get('type', 'unknown'),
                    'text': b.get('text', ''),
                    'page_id': p_idx + 1,
                    'local_id': b_idx,
                    **self.detector.get_box_features(box)
                }
                #
                emb = self._ensure_node_embedding(node_attrs, img)
                if emb is not None:
                    node_attrs['embedding'] = emb
                G.add_node(global_id, **node_attrs)
                id_map[global_id] = (p_idx, b_idx)
                global_id += 1

            #
            page_global_ids = [gid for gid,(pp,_) in id_map.items() if pp == p_idx]
            if intra_use_knn and len(page_global_ids) > 1:
                #
                for i in page_global_ids:
                    candidates: List[Tuple[int, float, Dict[str, Any]]] = []
                    box_i = G.nodes[i]['bbox']
                    for j in page_global_ids:
                        if i == j:
                            continue
                        box_j = G.nodes[j]['bbox']
                        rel = self.detector.compute_spatial_relation(box_i, box_j)
                        dist = float(rel.get('distance', 1.0))
                        iou = float(rel.get('iou', 0.0))
                        w_dist = 1.0 / (1.0 + (dist / max(spatial_distance_scale, 1e-6)))
                        w_iou = iou_weight * max(iou, 0.0)
                        base_w = max(w_dist + w_iou, min_edge_weight)
                        #
                        weight = max(base_w * (1.0 + max(intra_weight_boost, 0.0)), min_edge_weight)
                        candidates.append((j, dist, {**rel, 'weight': weight}))
                    #
                    candidates.sort(key=lambda t: t[1])
                    for j, _, rel_data in candidates[:max(1, k_intra_spatial)]:
                        #
                        if i < j:
                            G.add_edge(i, j, type='intra-page', **rel_data)
            else:
                for i in page_global_ids:
                    for j in page_global_ids:
                        if i == j:
                            continue
                        box1 = G.nodes[i]['bbox']
                        box2 = G.nodes[j]['bbox']
                        rel = self.detector.compute_spatial_relation(box1, box2)
                        dist = float(rel.get('distance', 1.0))
                        iou = float(rel.get('iou', 0.0))
                        w_dist = 1.0 / (1.0 + (dist / max(spatial_distance_scale, 1e-6)))
                        w_iou = iou_weight * max(iou, 0.0)
                        w = max(w_dist + w_iou, min_edge_weight)
                        if dist < 3 * spatial_distance_scale or iou > 0.05:
                            #
                            w = max(w * (1.0 + max(intra_weight_boost, 0.0)), min_edge_weight)
                            if i < j:
                                G.add_edge(i, j, weight=w, type='intra-page', **rel)

            #
            if add_page_nodes:
                H, W = (imgs[p_idx].shape[0], imgs[p_idx].shape[1]) if (imgs and p_idx < len(imgs) and isinstance(imgs[p_idx], np.ndarray)) else (2000, 1500)
                page_node_attr = {
                    'type': 'page',
                    'text': f'page-{p_idx+1}',
                    'page_id': p_idx + 1,
                    'local_id': -1,
                    'bbox': [0.0, 0.0, float(W), float(H)],
                    'center': (float(W)/2.0, float(H)/2.0),
                }
                G.add_node(global_id, **page_node_attr)
                id_map[global_id] = (p_idx, -1)
                page_node_ids[p_idx] = global_id
                #
                cand_ids = [gid for gid,(pp,_) in id_map.items() if pp == p_idx and gid != global_id]
                #
                if isinstance(page_anchor_out_degree_cap, int) and page_anchor_out_degree_cap > 0:
                    cand_ids = cand_ids[:page_anchor_out_degree_cap]
                for j in cand_ids:
                    w = max(page_anchor_weight, min_edge_weight)
                    #
                    G.add_edge(global_id, j, weight=w, type='page-anchor')
                global_id += 1

        #
        #
        node_embs: Dict[int, Optional[np.ndarray]] = {}
        for n in G.nodes():
            p_idx = G.nodes[n]['page_id'] - 1
            node_embs[n] = self._ensure_node_embedding(G.nodes[n], imgs[p_idx] if p_idx < len(imgs) else None)

        #
        for n in G.nodes():
            page_id = G.nodes[n]['page_id']
            this_emb = node_embs[n]
            if this_emb is None:
                continue
            #
            candidate_pages: List[int] = []
            if cross_candidate_mode == "all":
                candidate_pages = [p for p in range(1, len(pages_blocks)+1) if p != page_id]
            elif cross_candidate_mode == "explicit_only":
                candidate_pages = [p for p in range(1, len(pages_blocks)+1) if p != page_id]
            elif cross_candidate_mode == "adjacent_or_explicit":
                candidate_pages = [page_id - 1, page_id + 1]
            else:
                #
                candidate_pages = [page_id - 1, page_id + 1]
            #
            cands: List[Tuple[int, float]] = []  # (node_j, sim)
            src_type = G.nodes[n].get('type')
            src_text = G.nodes[n].get('text', '')
            src_tokens = set(self._tokenize_text(src_text)) if src_text else set()
            ref_flags = self._detect_explicit_reference(src_text)
            #
            if cross_candidate_mode == "adjacent_or_explicit" and any(ref_flags.values()):
                candidate_pages = [p for p in range(1, len(pages_blocks)+1) if p != page_id]
            for p2 in candidate_pages:
                if p2 < 1 or p2 > len(pages_blocks):
                    continue
                #
                for m in G.nodes():
                    if G.nodes[m]['page_id'] != p2:
                        continue
                    #
                    if cross_candidate_mode == "explicit_only" and not any(ref_flags.values()):
                        continue
                    dst_type = G.nodes[m].get('type')
                    if (dst_type or '').lower() == 'page':
                        continue
                    dst_text = G.nodes[m].get('text', '')
                    dst_tokens = set(self._tokenize_text(dst_text)) if dst_text else set()
                    if cross_type_constraints and not self._is_type_compatible(src_type, dst_type, ref_flags):
                        continue
                    token_overlap = self._jaccard_overlap(src_tokens, dst_tokens) if (src_tokens and dst_tokens) else 0.0
                    #
                    if node_embs[m] is None:
                        sim = 0.0
                    else:
                        sim = (self._cosine_sim(this_emb, node_embs[m]) + 1.0) / 2.0  # 0~1
                    #
                    if sim >= cross_sim_thresh and token_overlap >= max(0.0, cross_min_token_overlap):
                        #
                        gap = abs(p2 - page_id)
                        base_w = max(sim - page_gap_penalty * max(gap - 1, 0), 0.0)
                        if base_w <= 0:
                            continue
                        #
                        conf = 1.0
                        if any(ref_flags.values()):
                            conf += max(explicit_ref_boost, 0.0)
                        #
                        conf = max(conf * (1.0 + 0.2 * token_overlap), 0.0)
                        w = max(base_w * conf, 0.0)
                        if w > 0:
                            cands.append((m, w))
            #
            if cands:
                cands.sort(key=lambda x: x[1], reverse=True)
                max_out = cross_out_degree_cap if isinstance(cross_out_degree_cap, int) and cross_out_degree_cap > 0 else cross_top_k
                for m, w in cands[:max(1, max_out)]:
                    #
                    if G.has_edge(m, n):
                        try:
                            rev_w = float(G[m][n].get('weight', 0.0))
                        except Exception:
                            rev_w = 0.0
                        if rev_w >= w:
                            continue
                        else:
                            G.remove_edge(m, n)
                    G.add_edge(n, m, weight=max(w, min_edge_weight), type='cross-page')

        #
        if add_title_affinity_edges:
            for p_idx in range(len(pages_blocks)):
                page_nodes = [n for n in G.nodes() if G.nodes[n].get('page_id') == p_idx + 1 and (G.nodes[n].get('type') or '').lower() != 'page']
                if not page_nodes:
                    continue
                H = (imgs[p_idx].shape[0]) if (imgs and p_idx < len(imgs) and isinstance(imgs[p_idx], np.ndarray)) else 2000
                if title_max_vertical_gap_ratio <= 1.0:
                    max_gap = max(0.0, float(title_max_vertical_gap_ratio)) * float(H)
                else:
                    max_gap = float(title_max_vertical_gap_ratio)
                title_nodes = [n for n in page_nodes if (G.nodes[n].get('type') or '').lower() == 'title']
                if not title_nodes:
                    continue
                #
                title_nodes_sorted = sorted(title_nodes, key=lambda n: float(G.nodes[n]['bbox'][1]) if G.nodes[n].get('bbox') else float('inf'))
                content_nodes = [n for n in page_nodes if (G.nodes[n].get('type') or '').lower() != 'title']
                for m in content_nodes:
                    mb = G.nodes[m].get('bbox')
                    if not mb:
                        continue
                    mx1, my1, mx2, my2 = [float(v) for v in mb]
                    #
                    best_t = None
                    best_vdist = float('inf')
                    for t in title_nodes_sorted:
                        tb = G.nodes[t].get('bbox')
                        if not tb:
                            continue
                        tx1, ty1, tx2, ty2 = [float(v) for v in tb]
                        if ty2 >= my1:
                            break  #
                        hor_ov = self._horizontal_overlap_ratio(tb, mb)
                        if hor_ov < max(0.0, title_horizontal_overlap_thresh):
                            continue
                        vdist = max(0.0, my1 - ty2)
                        if vdist > max_gap:
                            continue
                        #
                        if title_stop_at_next_title:
                            blocked = False
                            for tn in title_nodes_sorted:
                                if tn == t:
                                    continue
                                tnb = G.nodes[tn].get('bbox')
                                if not tnb:
                                    continue
                                _, tny1, _, _ = [float(v) for v in tnb]
                                if ty2 < tny1 < my1:
                                    blocked = True
                                    break
                            if blocked:
                                continue
                        if vdist < best_vdist:
                            best_vdist = vdist
                            best_t = t
                    if best_t is not None:
                        w = max(title_affinity_weight, min_edge_weight)
                        G.add_edge(best_t, m, weight=w, type='title-affinity')

        #
        if add_caption_edges:
            cap_types = {'table_caption', 'figure_caption', 'caption'}
            for p_idx in range(len(pages_blocks)):
                page_nodes = [n for n in G.nodes() if G.nodes[n].get('page_id') == p_idx + 1]
                H = (imgs[p_idx].shape[0]) if (imgs and p_idx < len(imgs) and isinstance(imgs[p_idx], np.ndarray)) else 2000
                if caption_max_vertical_gap_ratio <= 1.0:
                    cap_gap = max(0.0, float(caption_max_vertical_gap_ratio)) * float(H)
                else:
                    cap_gap = float(caption_max_vertical_gap_ratio)
                visual_nodes = [n for n in page_nodes if (G.nodes[n].get('type') or '').lower() in {'figure','image','table'}]
                caption_nodes = [n for n in page_nodes if (G.nodes[n].get('type') or '').lower() in cap_types]
                for v in visual_nodes:
                    vb = G.nodes[v].get('bbox')
                    if not vb:
                        continue
                    vx1, vy1, vx2, vy2 = [float(x) for x in vb]
                    best_c = None; best_vd = float('inf')
                    for c in caption_nodes:
                        cb = G.nodes[c].get('bbox')
                        if not cb:
                            continue
                        cx1, cy1, cx2, cy2 = [float(x) for x in cb]
                        if cy1 <= vy2:
                            continue  #
                        hor_ov = self._horizontal_overlap_ratio(vb, cb)
                        if hor_ov < max(0.0, caption_horizontal_overlap_thresh):
                            continue
                        vdist = max(0.0, cy1 - vy2)
                        if vdist > cap_gap:
                            continue
                        if vdist < best_vd:
                            best_vd = vdist; best_c = c
                    if best_c is not None:
                        G.add_edge(v, best_c, weight=max(title_affinity_weight, min_edge_weight), type='caption-affinity')

        #
        if add_list_local_edges:
            for p_idx in range(len(pages_blocks)):
                page_nodes = [n for n in G.nodes() if G.nodes[n].get('page_id') == p_idx + 1]
                H = (imgs[p_idx].shape[0]) if (imgs and p_idx < len(imgs) and isinstance(imgs[p_idx], np.ndarray)) else 2000
                if list_max_vertical_gap_ratio <= 1.0:
                    list_gap = max(0.0, float(list_max_vertical_gap_ratio)) * float(H)
                else:
                    list_gap = float(list_max_vertical_gap_ratio)
                lists = [n for n in page_nodes if (G.nodes[n].get('type') or '').lower() == 'list']
                lists_sorted = sorted(lists, key=lambda n: float(G.nodes[n]['bbox'][1]) if G.nodes[n].get('bbox') else float('inf'))
                for i, a in enumerate(lists_sorted):
                    ab = G.nodes[a].get('bbox')
                    if not ab:
                        continue
                    ax1, ay1, ax2, ay2 = [float(x) for x in ab]
                    best_b = None; best_vd = float('inf')
                    for b in lists_sorted[i+1:]:
                        bb = G.nodes[b].get('bbox')
                        if not bb:
                            continue
                        bx1, by1, bx2, by2 = [float(x) for x in bb]
                        if by1 <= ay2:
                            continue
                        hor_ov = self._horizontal_overlap_ratio(ab, bb)
                        if hor_ov < max(0.0, list_horizontal_overlap_thresh):
                            continue
                        vdist = max(0.0, by1 - ay2)
                        if vdist > list_gap:
                            break
                        best_b = b; best_vd = vdist
                        break
                    if best_b is not None:
                        G.add_edge(a, best_b, weight=max(title_affinity_weight, min_edge_weight), type='list-next')

        #
        if add_page_chain_edges and len(pages_blocks) >= 2:
            for p_idx in range(len(pages_blocks) - 1):
                this_nodes = [n for n in G.nodes() if G.nodes[n].get('page_id') == p_idx + 1 and (G.nodes[n].get('type') or '').lower() != 'page']
                next_nodes = [n for n in G.nodes() if G.nodes[n].get('page_id') == p_idx + 2 and (G.nodes[n].get('type') or '').lower() != 'page']
                if not this_nodes or not next_nodes:
                    continue
                #
                tail = max(this_nodes, key=lambda n: float(G.nodes[n]['bbox'][3]) if G.nodes[n].get('bbox') else -1e9)
                head = min(next_nodes, key=lambda n: float(G.nodes[n]['bbox'][1]) if G.nodes[n].get('bbox') else 1e9)
                G.add_edge(tail, head, weight=max(page_chain_weight, min_edge_weight), type='page-chain')

        return G, id_map

    # -------- New: Semantic-Structural Graph Builder --------
    def build_semantic_structural_graph(
        self,
        pages_blocks: List[List[Dict[str, Any]]],
        imgs: List[np.ndarray],
        # Intra-modal config
        title_horizontal_overlap_thresh: float = 0.2,
        title_max_vertical_gap_ratio: float = 0.25,
        title_stop_at_next_title: bool = True,
        title_child_weight: float = 0.8,
        before_weight: float = 0.5,
        text_knn_k: int = 2,
        text_knn_sim_thresh: float = 0.3,
        text_knn_weight: float = 0.6,
        # Inter-modal config
        caption_horizontal_overlap_thresh: float = 0.2,
        caption_max_vertical_gap_ratio: float = 0.2,
        inter_caption_weight: float = 0.9,
        caption_search_radius_ratio: float = 0.4,
        caption_candidate_page_gap: int = 0,
        caption_min_score: float = 0.5,
        caption_weight_geo: float = 0.25,
        caption_weight_clip: float = 0.5,
        caption_weight_cue: float = 0.15,
        caption_weight_hier: float = 0.10,
        # Semantic co-occurrence (optional, replaces same-time)
        add_semantic_cooccur_edges: bool = False,
        cooccur_within_same_title: bool = True,
        cooccur_sim_thresh: float = 0.4,
        cooccur_token_overlap_thresh: float = 0.2,
        cooccur_top_k: int = 1,
        cooccur_weight: float = 0.4,
        min_edge_weight: float = 1e-6,
    ) -> Tuple[nx.DiGraph, Dict[int, Tuple[int, int]]]:
        """"""
        G = nx.DiGraph()
        id_map: Dict[int, Tuple[int, int]] = {}
        global_id = 0

        #
        for p_idx, blocks in enumerate(pages_blocks):
            img = imgs[p_idx] if p_idx < len(imgs) else None
            for b_idx, b in enumerate(blocks):
                box = b.get('bbox')
                if box is None:
                    continue
                node_attrs = {
                    'type': b.get('type', 'unknown'),
                    'text': b.get('text', ''),
                    'page_id': p_idx + 1,
                    'local_id': b_idx,
                    **self.detector.get_box_features(box)
                }
                emb = self._ensure_node_embedding(node_attrs, img)
                if emb is not None:
                    node_attrs['embedding'] = emb
                G.add_node(global_id, **node_attrs)
                id_map[global_id] = (p_idx, b_idx)
                global_id += 1

        #
        for p_idx, blocks in enumerate(pages_blocks):
            page_nodes = [n for n in G.nodes() if G.nodes[n].get('page_id') == p_idx + 1]
            if not page_nodes:
                continue
            H = (imgs[p_idx].shape[0]) if (imgs and p_idx < len(imgs) and isinstance(imgs[p_idx], np.ndarray)) else 2000
            max_gap = (max(0.0, float(title_max_vertical_gap_ratio)) * float(H)) if title_max_vertical_gap_ratio <= 1.0 else float(title_max_vertical_gap_ratio)
            titles = [n for n in page_nodes if (G.nodes[n].get('type') or '').lower() == 'title']
            if not titles:
                continue
            titles_sorted = sorted(titles, key=lambda n: float(G.nodes[n]['bbox'][1]) if G.nodes[n].get('bbox') else float('inf'))
            contents = [n for n in page_nodes if (G.nodes[n].get('type') or '').lower() != 'title']
            for m in contents:
                mb = G.nodes[m].get('bbox')
                if not mb:
                    continue
                mx1, my1, mx2, my2 = [float(v) for v in mb]
                best_t = None; best_vd = float('inf')
                for t in titles_sorted:
                    tb = G.nodes[t].get('bbox')
                    if not tb:
                        continue
                    tx1, ty1, tx2, ty2 = [float(v) for v in tb]
                    if ty2 >= my1:
                        break
                    hov = self._horizontal_overlap_ratio(tb, mb)
                    if hov < max(0.0, title_horizontal_overlap_thresh):
                        continue
                    vdist = max(0.0, my1 - ty2)
                    if vdist > max_gap:
                        continue
                    if title_stop_at_next_title:
                        blocked = False
                        for tn in titles_sorted:
                            if tn == t:
                                continue
                            tnb = G.nodes[tn].get('bbox')
                            if not tnb:
                                continue
                            _, tny1, _, _ = [float(v) for v in tnb]
                            if ty2 < tny1 < my1:
                                blocked = True
                                break
                        if blocked:
                            continue
                    if vdist < best_vd:
                        best_vd = vdist; best_t = t
                if best_t is not None:
                    G.add_edge(best_t, m, weight=max(title_child_weight, min_edge_weight), type='has-child', rel='has-child')

        #
        for p_idx, blocks in enumerate(pages_blocks):
            page_nodes = [n for n in G.nodes() if G.nodes[n].get('page_id') == p_idx + 1]
            texts = [n for n in page_nodes if (G.nodes[n].get('type') or '').lower() in {'text','list','table_caption','figure_caption','caption'}]
            if len(texts) >= 2:
                #
                def keyfn(n):
                    x1, y1, x2, y2 = [float(v) for v in G.nodes[n]['bbox']]
                    return (round(y1/8.0), x1)
                texts_sorted = sorted(texts, key=keyfn)
                for i in range(len(texts_sorted)-1):
                    a, b = texts_sorted[i], texts_sorted[i+1]
                    G.add_edge(a, b, weight=max(before_weight, min_edge_weight), type='before', rel='before')

        #
        for p_idx, blocks in enumerate(pages_blocks):
            page_nodes = [n for n in G.nodes() if G.nodes[n].get('page_id') == p_idx + 1]
            texts = [n for n in page_nodes if (G.nodes[n].get('type') or '').lower() in {'text','list'}]
            if len(texts) >= 2:
                #
                embs = {n: G.nodes[n].get('embedding') for n in texts}
                #
                def keyfn(n):
                    x1, y1, x2, y2 = [float(v) for v in G.nodes[n]['bbox']]
                    return (round(y1/8.0), x1)
                texts_sorted = sorted(texts, key=keyfn)
                for i, a in enumerate(texts_sorted):
                    sims: List[Tuple[int, float]] = []
                    ea = embs[a]
                    if ea is None:
                        continue
                    for j in range(i+1, len(texts_sorted)):
                        b = texts_sorted[j]
                        eb = embs[b]
                        if eb is None:
                            continue
                        s = self._cosine_sim(ea, eb)
                        s01 = (s + 1.0)/2.0
                        if s01 >= max(0.0, text_knn_sim_thresh):
                            sims.append((b, s01))
                    if sims:
                        sims.sort(key=lambda t: t[1], reverse=True)
                        for b, s01 in sims[:max(1, int(text_knn_k))]:
                            w = max(text_knn_weight * s01, min_edge_weight)
                            if not G.has_edge(a, b):
                                G.add_edge(a, b, weight=w, type='semantic-tt', rel='semantic')

        #
        #
        try:
            from networkx.algorithms.matching import max_weight_matching
        except Exception:
            max_weight_matching = None
        #
        parent_title: Dict[int, int] = {}
        for u, v, d in G.edges(data=True):
            if (d.get('type') or '') == 'has-child':
                parent_title[v] = u
        def _vertical_overlap_ratio(box_a: List[float], box_b: List[float]) -> float:
            try:
                ax1, ay1, ax2, ay2 = [float(x) for x in box_a]
                bx1, by1, bx2, by2 = [float(x) for x in box_b]
            except Exception:
                return 0.0
            ha = max(0.0, ay2 - ay1)
            hb = max(0.0, by2 - by1)
            if ha <= 0.0 or hb <= 0.0:
                return 0.0
            overlap = max(0.0, min(ay2, by2) - max(ay1, by1))
            denom = max(1e-6, min(ha, hb))
            return float(overlap / denom)
        def _caption_cue_score(text: str) -> float:
            if not isinstance(text, str):
                return 0.0
            t = text.lower()
            s = 0.0
            if re.search(r"\b(fig(?:\.|ure)?)\b", t):
                s += 0.5
            if re.search(r"\b(table)\b", t):
                s += 0.5
            if re.search(r"\d+", t):
                s += 0.2
            return min(1.0, s)
        for p_idx, blocks in enumerate(pages_blocks):
            #
            pid = p_idx + 1
            candidate_pages = [q for q in range(max(1, pid - caption_candidate_page_gap), min(len(pages_blocks), pid + caption_candidate_page_gap) + 1)]
            page_nodes = [n for n in G.nodes() if int(G.nodes[n].get('page_id', 1)) in candidate_pages]
            if not page_nodes:
                continue
            H = (imgs[p_idx].shape[0]) if (imgs and p_idx < len(imgs) and isinstance(imgs[p_idx], np.ndarray)) else 2000
            W = (imgs[p_idx].shape[1]) if (imgs and p_idx < len(imgs) and isinstance(imgs[p_idx], np.ndarray)) else 1500
            radius = float(caption_search_radius_ratio) * float(H)
            visuals = [n for n in page_nodes if (G.nodes[n].get('type') or '').lower() in {'figure','image','table'}]
            captions = [n for n in page_nodes if (G.nodes[n].get('type') or '').lower() in {'table_caption','figure_caption','caption'}]
            if not visuals or not captions:
                continue
            #
            B = nx.Graph()
            for v in visuals:
                B.add_node(('v', v), bipartite=0)
            for c in captions:
                B.add_node(('c', c), bipartite=1)
            for v in visuals:
                vb = G.nodes[v].get('bbox')
                if not vb:
                    continue
                vcx, vcy = G.nodes[v].get('center', (0.0, 0.0))
                vemb = G.nodes[v].get('embedding')
                for c in captions:
                    cb = G.nodes[c].get('bbox')
                    if not cb:
                        continue
                    ccx, ccy = G.nodes[c].get('center', (0.0, 0.0))
                    #
                    dist = math.sqrt((vcx - ccx)**2 + (vcy - ccy)**2)
                    if dist > radius:
                        continue
                    #
                    geo_dist = max(0.0, 1.0 - dist / max(1e-6, radius))
                    hov = self._horizontal_overlap_ratio(vb, cb)
                    vov = _vertical_overlap_ratio(vb, cb)
                    geo = 0.6 * geo_dist + 0.4 * max(hov, vov)
                    #
                    cue = _caption_cue_score(G.nodes[c].get('text', ''))
                    #
                    cemb = G.nodes[c].get('embedding')
                    sim01 = 0.0
                    if vemb is not None and cemb is not None:
                        sim = self._cosine_sim(vemb, cemb)
                        sim01 = (sim + 1.0) / 2.0
                    #
                    hier = 1.0 if (parent_title.get(v, None) is not None and parent_title.get(v, None) == parent_title.get(c, None)) else 0.0
                    #
                    score = (
                        max(0.0, float(caption_weight_geo)) * geo +
                        max(0.0, float(caption_weight_clip)) * sim01 +
                        max(0.0, float(caption_weight_cue)) * cue +
                        max(0.0, float(caption_weight_hier)) * hier
                    )
                    score = float(score)
                    if score >= float(caption_min_score):
                        B.add_edge(('v', v), ('c', c), weight=score)
            #
            matches = set()
            if max_weight_matching is not None:
                matches = max_weight_matching(B, maxcardinality=False)
            #
            for a, b in matches:
                if a[0] == 'c':
                    a, b = b, a
                v = a[1]; c = b[1]
                w = float(B[a][b].get('weight', 0.0))
                G.add_edge(v, c, weight=max(inter_caption_weight * w, min_edge_weight), type='inter-explain', rel='explain')

        #
        if add_semantic_cooccur_edges:
            #
            parent_title: Dict[int, int] = {}
            for u, v, d in G.edges(data=True):
                if (d.get('type') or '') == 'has-child':
                    parent_title[v] = u
            #
            def _tokens(s: str) -> set:
                return set(self._tokenize_text(s)) if isinstance(s, str) else set()
            for scope in ('title', 'page'):
                #
                if cooccur_within_same_title and scope == 'title':
                    buckets: Dict[int, List[int]] = {}
                    for n in G.nodes():
                        if n in parent_title:
                            buckets.setdefault(parent_title[n], []).append(n)
                    scopes = [(tid, nodes) for tid, nodes in buckets.items() if len(nodes) >= 2]
                elif not cooccur_within_same_title and scope == 'page':
                    buckets: Dict[int, List[int]] = {}
                    for n in G.nodes():
                        pid = int(G.nodes[n].get('page_id', 1))
                        buckets.setdefault(pid, []).append(n)
                    scopes = [(pid, nodes) for pid, nodes in buckets.items() if len(nodes) >= 2]
                else:
                    continue
                for _, nodes in scopes:
                    #
                    cand = [n for n in nodes if (G.nodes[n].get('type') or '').lower() in {'text','list','table_caption','figure_caption','caption'}]
                    if len(cand) < 2:
                        continue
                    #
                    embs = {n: G.nodes[n].get('embedding') for n in cand}
                    toks = {n: _tokens(G.nodes[n].get('text', '')) for n in cand}
                    def keyfn(n):
                        x1, y1, x2, y2 = [float(v) for v in G.nodes[n]['bbox']]
                        return (round(y1/8.0), x1)
                    cand_sorted = sorted(cand, key=keyfn)
                    for i, a in enumerate(cand_sorted):
                        ea = embs[a]
                        if ea is None:
                            continue
                        pairs: List[Tuple[int, float]] = []
                        for j in range(i+1, len(cand_sorted)):
                            b = cand_sorted[j]
                            eb = embs[b]
                            if eb is None:
                                continue
                            sim = self._cosine_sim(ea, eb)
                            sim01 = (sim + 1.0)/2.0
                            if sim01 < max(0.0, cooccur_sim_thresh):
                                continue
                            tover = self._jaccard_overlap(toks[a], toks[b]) if toks[a] and toks[b] else 0.0
                            if tover < max(0.0, cooccur_token_overlap_thresh):
                                continue
                            score = 0.5*sim01 + 0.5*tover
                            pairs.append((b, score))
                        if pairs:
                            pairs.sort(key=lambda t: t[1], reverse=True)
                            for b, sc in pairs[:max(1, int(cooccur_top_k))]:
                                w = max(cooccur_weight * sc, min_edge_weight)
                                if not G.has_edge(a, b):
                                    G.add_edge(a, b, weight=w, type='inter-cooccur', rel='cooccur')

        return G, id_map

    # ---------- helper utilities for graph construction ----------
    @staticmethod
    def _tokenize_text(text: str) -> List[str]:
        if not text:
            return []
        #
        raw = re.split(r"\W+", text.lower(), flags=re.UNICODE)
        tokens = []
        for t in raw:
            if len(t) < 2:
                continue
            #
            if not any(ch.isalnum() for ch in t):
                continue
            tokens.append(t)
        return tokens

    @staticmethod
    def _jaccard_overlap(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return float(inter / union) if union > 0 else 0.0

    @staticmethod
    def _horizontal_overlap_ratio(box_a: Optional[List[float]], box_b: Optional[List[float]]) -> float:
        """"""
        if not box_a or not box_b or len(box_a) < 4 or len(box_b) < 4:
            return 0.0
        try:
            ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
            bx1, by1, bx2, by2 = [float(v) for v in box_b]
        except Exception:
            return 0.0
        wa = max(0.0, ax2 - ax1)
        wb = max(0.0, bx2 - bx1)
        if wa <= 0.0 or wb <= 0.0:
            return 0.0
        overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        denom = max(1e-6, min(wa, wb))
        return float(overlap / denom)

    @staticmethod
    def _detect_explicit_reference(text: str) -> Dict[str, bool]:
        if not text:
            return {"figure_ref": False, "table_ref": False, "section_ref": False, "page_ref": False, "equation_ref": False}
        t = text.lower()
        flags = {
            "figure_ref": bool(re.search(r"\b(fig(?:\.|ure)?)\s*[:#]?-?\s*\d+", t)),
            "table_ref": bool(re.search(r"\b(table)\s*[:#]?-?\s*\d+", t)),
            "section_ref": bool(re.search(r"\b(section|sec\.)\b", t)),
            "page_ref": bool(re.search(r"\b(page|p\.)\s*\d+", t)),
            "equation_ref": bool(re.search(r"\b(eq(?:\.|uation)?)\s*\(?\s*\d+[\w\)]*", t)),
        }
        return flags

    @staticmethod
    def _is_type_compatible(src_type: Optional[str], dst_type: Optional[str], ref_flags: Dict[str, bool]) -> bool:
        s = (src_type or 'unknown').lower()
        d = (dst_type or 'unknown').lower()
        #
        if ref_flags.get('figure_ref', False):
            return d in {'figure', 'image'}
        if ref_flags.get('table_ref', False):
            return d in {'table'}
        if ref_flags.get('equation_ref', False):
            return d in {'text', 'title', 'list'}  #
        if ref_flags.get('section_ref', False):
            return d in {'title', 'text', 'list'}
        #
        if s in {'text', 'list'}:
            return d in {'text', 'list', 'title', 'table', 'figure', 'image'}
        if s in {'title'}:
            return d in {'text', 'list', 'title'}
        if s in {'table'}:
            return d in {'text', 'list', 'title', 'table'}
        if s in {'figure', 'image'}:
            return d in {'text', 'list', 'title', 'figure', 'image'}
        return True

    # ---------- graph inspection & export ----------
    @staticmethod
    def summarize_graph(G: nx.DiGraph) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'edge_types': {},
            'node_types': {},
            'avg_out_degree': float(np.mean([G.out_degree(n) for n in G.nodes()])) if G.number_of_nodes() > 0 else 0.0,
            'avg_in_degree': float(np.mean([G.in_degree(n) for n in G.nodes()])) if G.number_of_nodes() > 0 else 0.0,
        }
        for _, _, data in G.edges(data=True):
            et = data.get('type', 'unknown')
            summary['edge_types'][et] = summary['edge_types'].get(et, 0) + 1
        for n in G.nodes():
            nt = G.nodes[n].get('type', 'unknown')
            summary['node_types'][nt] = summary['node_types'].get(nt, 0) + 1
        return summary

    @staticmethod
    def export_graph(G: nx.DiGraph, out_path: str) -> bool:
        try:
            if out_path.endswith('.gexf'):
                nx.write_gexf(G, out_path)
            elif out_path.endswith('.graphml'):
                nx.write_graphml(G, out_path)
            elif out_path.endswith('.edgelist'):
                nx.write_edgelist(G, out_path, data=['weight', 'type'])
            elif out_path.endswith('.png'):
                try:
                    import matplotlib.pyplot as plt  # type: ignore
                except Exception:
                    return False
                plt.figure(figsize=(10, 8))
                pos = nx.spring_layout(G, k=0.3, seed=42)
                edge_colors = ['#1f77b4' if d.get('type') == 'intra-page' else '#ff7f0e' for _, _, d in G.edges(data=True)]
                nx.draw_networkx_nodes(G, pos, node_size=80, alpha=0.8)
                nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color=edge_colors)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(out_path, dpi=200)
                plt.close()
            else:
                nx.write_gpickle(G, out_path)
            return True
        except Exception:
            return False

    def debug_build_graph_from_pages(
        self,
        pages_blocks: List[List[Dict[str, Any]]],
        imgs: List[np.ndarray],
        export_path: Optional[str] = None,
        **graph_kwargs: Any,
    ) -> Tuple[nx.DiGraph, Dict[int, Tuple[int, int]], Dict[str, Any]]:
        G, id_map = self.build_multipage_graph(pages_blocks, imgs, **graph_kwargs)
        stats = self.summarize_graph(G)
        if export_path:
            self.export_graph(G, export_path)
        return G, id_map, stats

    def rerank_blocks_pagerank_multipage(
        self,
        pages_blocks: List[List[Dict[str, Any]]],
        imgs: List[np.ndarray],
        query: str,
        top_k: int = 5,
        alpha: float = 0.85,
        connect_adjacent_only: bool = True,
        cross_top_k: int = 3,
        cross_sim_thresh: float = 0.2,
        page_gap_penalty: float = 0.3,
        spatial_distance_scale: float = 300.0,
        iou_weight: float = 1.0,
        min_edge_weight: float = 1e-6,
        max_iter: int = 100,
        tol: float = 1e-06,
        #
        intra_use_knn: bool = True,
        k_intra_spatial: int = 6,
        intra_weight_boost: float = 0.5,
        cross_candidate_mode: str = "adjacent",
        cross_type_constraints: bool = True,
        cross_min_token_overlap: float = 0.0,
        cross_out_degree_cap: Optional[int] = None,
        explicit_ref_boost: float = 0.2,
        #
        add_page_nodes: bool = False,
        page_anchor_weight: float = 0.25,
        page_anchor_out_degree_cap: Optional[int] = None,
        add_title_affinity_edges: bool = True,
        title_affinity_weight: float = 0.6,
        title_horizontal_overlap_thresh: float = 0.2,
        title_max_vertical_gap_ratio: float = 0.25,
        title_affinity_out_degree_cap: Optional[int] = None,
        title_stop_at_next_title: bool = True
    ) -> List[Dict[str, Any]]:
        """"""
        if not pages_blocks:
            return []

        #
        G, id_map = self.build_multipage_graph(
            pages_blocks,
            imgs,
            connect_adjacent_only=connect_adjacent_only,
            cross_top_k=cross_top_k,
            cross_sim_thresh=cross_sim_thresh,
            page_gap_penalty=page_gap_penalty,
            spatial_distance_scale=spatial_distance_scale,
            iou_weight=iou_weight,
            min_edge_weight=min_edge_weight,
            intra_use_knn=intra_use_knn,
            k_intra_spatial=k_intra_spatial,
            intra_weight_boost=intra_weight_boost,
            cross_candidate_mode=cross_candidate_mode,
            cross_type_constraints=cross_type_constraints,
            cross_min_token_overlap=cross_min_token_overlap,
            cross_out_degree_cap=cross_out_degree_cap,
            explicit_ref_boost=explicit_ref_boost,
            add_page_nodes=add_page_nodes,
            page_anchor_weight=page_anchor_weight,
            page_anchor_out_degree_cap=page_anchor_out_degree_cap,
            add_title_affinity_edges=add_title_affinity_edges,
            title_affinity_weight=title_affinity_weight,
            title_horizontal_overlap_thresh=title_horizontal_overlap_thresh,
            title_max_vertical_gap_ratio=title_max_vertical_gap_ratio,
            title_affinity_out_degree_cap=title_affinity_out_degree_cap,
            title_stop_at_next_title=title_stop_at_next_title,
        )
        if G.number_of_nodes() == 0:
            return []

        #
        personalization = None
        query_emb = self.detector.get_clip_text_embedding(query)
        if query_emb is not None and G.number_of_nodes() > 0:
            p: Dict[int, float] = {}
            qn = np.linalg.norm(query_emb)
            for n in G.nodes():
                emb = G.nodes[n].get('embedding')
                if emb is None:
                    #
                    p_idx = G.nodes[n]['page_id'] - 1
                    emb = self._ensure_node_embedding(G.nodes[n], imgs[p_idx] if p_idx < len(imgs) else None)
                sim = 0.0
                if emb is not None and qn > 0 and np.linalg.norm(emb) > 0:
                    sim = float(np.dot(query_emb, emb) / (qn * np.linalg.norm(emb)))
                score = max((sim + 1.0) / 2.0, 1e-8)
                p[n] = score
            s = sum(p.values())
            if s > 0:
                personalization = {k: v / s for k, v in p.items()}

        #
        try:
            pr = nx.pagerank(
                G,
                alpha=alpha,
                personalization=personalization,
                weight='weight',
                max_iter=max_iter,
                tol=tol,
            )
        except Exception:
            #
            flat_blocks = []
            for p_idx, blocks in enumerate(pages_blocks):
                for b in blocks:
                    b2 = b.copy()
                    b2['page_id'] = p_idx + 1
                    flat_blocks.append(b2)
            return self.rerank_blocks_simple(flat_blocks, query, imgs[0] if imgs else None, top_k)

        #
        ranked = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:top_k]
        selected: List[Dict[str, Any]] = []
        for n, score in ranked:
            p_idx, b_idx = id_map[n]
            block = pages_blocks[p_idx][b_idx].copy()
            block['pagerank_score'] = float(score)
            block['rerank_score'] = float(score)
            block['page_id'] = p_idx + 1
            selected.append(block)
        return selected


class BlockPromptGenerator:
    """"""

    @staticmethod
    def _shorten(txt: str, max_chars: int = 400) -> str:
        """"""
        if not txt: 
            return ""
        t = " ".join(txt.split())
        return t[:max_chars] + ("..." if len(t) > max_chars else "")
    
    @staticmethod
    def _find_spatially_related_blocks(
        target_idx: int,
        all_blocks: List[Dict[str, Any]],
        max_distance_ratio: float = 0.3
    ) -> List[int]:
        """"""
        target = all_blocks[target_idx]
        target_bbox = target.get("bbox")
        target_page = target.get("page_id")
        
        if not target_bbox:
            return []
        
        tx1, ty1, tx2, ty2 = target_bbox
        target_center_y = (ty1 + ty2) / 2
        
        related = []
        for i, block in enumerate(all_blocks):
            if i == target_idx:
                continue
            
            #
            if block.get("page_id") != target_page:
                continue
            
            bbox = block.get("bbox")
            if not bbox:
                continue
            
            bx1, by1, bx2, by2 = bbox
            block_center_y = (by1 + by2) / 2
            
            #
            vertical_distance = abs(target_center_y - block_center_y)
            
            #
            #
            page_height_estimate = max(ty2, by2)  #
            if vertical_distance < page_height_estimate * max_distance_ratio:
                related.append(i)
        
        return related

    @staticmethod
    def create_structured_prompt(
        selected_blocks: List[Dict[str, Any]],
        query: str,
        page_id: Union[str, int] = 1,
        llm_mode: str = "text",  # "text" or "vlm"
        require_json: bool = True,
        include_scores: bool = True
    ) -> str:
        """"""
        if not selected_blocks:
            return (
                f"Task: Answer the question using ONLY the provided document blocks.\n"
                f"Question: {query}\n\nNo blocks available."
            )


        header = [
            "You are a careful document QA assistant.",
            "Read all provided blocks carefully and identify the relevant ones that help answer the question. ",
            "First select the relevant block ids, then answer.",
        ]
        
        #
        if llm_mode == "vlm":
            header += [
                "",
                "IMPORTANT: This document contains both TEXT and IMAGE blocks.",
                "- TEXT blocks may describe, reference, or provide context for nearby IMAGE blocks.",
                "- IMAGE blocks (figures/tables/charts) contain visual information that complements the text.",
                "- When analyzing images, also consider the surrounding text blocks for context and interpretation.",
                "- The images and text work together to answer the question - integrate information from both modalities.",
            ]

        if require_json:
            header += [
                "Return a single JSON object with fields:",
                '{"selected_block_ids": [int], "answer": str, "confidence": float, "notes": str}',
                "Do not add extra keys or free-form text outside JSON."
            ]
        lines = []
        lines.append("\n".join(header))
        lines.append(f"\nQuestion: {query}\n")
        lines.append("Blocks:")

        #
        spatial_relations = {}
        if llm_mode == "vlm":
            visual_types = {"figure", "image", "table"}
            for i, b in enumerate(selected_blocks):
                if b.get("type") in visual_types:
                    related = BlockPromptGenerator._find_spatially_related_blocks(i, selected_blocks, max_distance_ratio=0.35)
                    if related:
                        spatial_relations[i] = related

        for i, b in enumerate(selected_blocks):
            bid = b.get("id", i)
            btype = b.get("type", "unknown")
            bbox = b.get("bbox", [])
            score = b.get("rerank_score", None)
            text = BlockPromptGenerator._shorten(b.get("text", ""))
            
            #
            img_tag = ""
            context_hint = ""
            if llm_mode == "vlm" and btype in {"figure", "image", "table"}:
                img_tag = f" [IMAGE_REF: <IMAGE:{i}>]"
                
                #
                if i in spatial_relations:
                    related_ids = [selected_blocks[idx].get("id", idx) for idx in spatial_relations[i]]
                    if related_ids:
                        context_hint = f"\n  → Related text blocks: {related_ids} (may provide context/caption for this visual)"

            meta = f'(page={page_id}, bbox={tuple(map(int,bbox)) if bbox else "N/A"})'
            if include_scores and isinstance(score, (int, float)):
                meta += f", rel={score:.3f}"

            content = text if text else f"[{btype} without OCR text]"
            lines.append(f"- id={bid} | type={btype}{img_tag} | {meta}\n  text: {content}{context_hint}")

        lines.append("\nInstructions:")
        
        if llm_mode == "vlm":
            #
            lines.append("1) MULTIMODAL REASONING: When images are present, integrate information from BOTH the visual content AND the related text blocks.")
            lines.append("   - Check if nearby text blocks (marked as 'Related text blocks') provide titles, captions, or explanations for images.")
            lines.append("   - Use text blocks to interpret what the image shows and how it relates to the question.")
            lines.append("   - The answer may require combining visual data (from images) with textual context (from text blocks).")
            lines.append("2) Cite both visual observations AND relevant text blocks in your reasoning.")
        else:
            #
            lines.append("1) Cite the exact quoted block(s) you will use in your reasoning.")
            lines.append("2) Do not include paraphrased or invented content.")

        
        if require_json:
            lines.append('\nOutput JSON only, e.g. {"selected_block_ids":[1,3], "answer":"...", "confidence":0.72, "notes":"..."}')

        return "\n".join(lines)
    
    @staticmethod
    def create_block_prompt(
        selected_blocks: List[Dict[str, Any]], 
        query: str,
        include_spatial_info: bool = True
    ) -> str:
        """"""
        return BlockPromptGenerator.create_structured_prompt(
            selected_blocks, 
            query, 
            llm_mode="text",
            require_json=False,
            include_scores=include_spatial_info
        )
    
    @staticmethod
    def create_multimodal_prompt(
        selected_blocks: List[Dict[str, Any]], 
        query: str,
        image_regions: List[Image.Image] = None
    ) -> Tuple[str, List[Image.Image]]:
        """"""
        #
        text_prompt = BlockPromptGenerator.create_structured_prompt(
            selected_blocks,
            query,
            llm_mode="vlm",
            require_json=False,
            include_scores=True
        )
        
        images = image_regions if image_regions else []
        return text_prompt, images
    
    @staticmethod
    def create_json_prompt(
        selected_blocks: List[Dict[str, Any]], 
        query: str,
        page_id: Union[str, int] = 1,
        llm_mode: str = "text"
    ) -> str:
        """"""
        return BlockPromptGenerator.create_structured_prompt(
            selected_blocks,
            query,
            page_id=page_id,
            llm_mode=llm_mode,
            require_json=True,
            include_scores=True
        )


#
_global_detector = None
_global_reranker = None
_DETECT_CACHE: Optional[Dict[Tuple[str, Optional[float]], Tuple[List[Dict[str, Any]], np.ndarray]]] = {}


def get_global_detector(
    clip_model_path: str = os.environ.get('HYPERDOC_OPENCLIP_MODEL', ''),
    #
    ocr_use_trt: bool = True,
    ocr_precision: str = 'fp16',
    ocr_rec_batch_num: int = 128,
    det_limit_side_len: int = 1280,
    det_db_box_thresh: float = 0.3,
    ocr_drop_score: float = 0.2,
    layout_use_gpu: bool = True,
    ocr_verbose: bool = False,
    ocr_prewarm: bool = False,
    ocr_gpu_id: int = 0,
    clip_gpu_id: int = 0,
    layout_gpu_id: int = 0,
) -> BlockDetector:
    """"""
    global _global_detector
    if _global_detector is None:
        print("Initializing the block detector...")
        _global_detector = BlockDetector(
            clip_model_path=clip_model_path,
            ocr_use_trt=ocr_use_trt,
            ocr_precision=ocr_precision,
            ocr_rec_batch_num=ocr_rec_batch_num,
            det_limit_side_len=det_limit_side_len,
            det_db_box_thresh=det_db_box_thresh,
            ocr_drop_score=ocr_drop_score,
            layout_use_gpu=layout_use_gpu,
            ocr_verbose=ocr_verbose,
            ocr_prewarm=ocr_prewarm,
            ocr_gpu_id=ocr_gpu_id,
            clip_gpu_id=clip_gpu_id,
            layout_gpu_id=layout_gpu_id,
        )
        print("Block detector initialized")
    return _global_detector


def get_global_reranker(clip_model_path: str = os.environ.get('HYPERDOC_OPENCLIP_MODEL', '')) -> BlockReranker:
    """"""
    global _global_reranker
    if _global_reranker is None:
        detector = get_global_detector(clip_model_path)
        _global_reranker = BlockReranker(detector)
    return _global_reranker


def cleanup_global_models():
    """"""
    global _global_detector, _global_reranker
    _global_detector = None
    _global_reranker = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Global model instances cleared")


def process_page_blocks(
    image_path: str,
    query: str,
    use_graph: bool = False,
    top_k: int = 5,
    clip_model_path: str = os.environ.get('HYPERDOC_OPENCLIP_MODEL', ''),
    detector: Optional[BlockDetector] = None,
    reranker: Optional[BlockReranker] = None
) -> Dict[str, Any]:
    """"""
    #
    if detector is None:
        detector = get_global_detector(clip_model_path)
    if reranker is None:
        reranker = get_global_reranker(clip_model_path)
    
    #
    blocks, img = detector.detect_blocks(image_path)
    
    if not blocks:
        return {
            'selected_blocks': [],
            'text_prompt': f"Question: {query}\n\nDocument content: no content blocks were detected.",
            'multimodal_prompt': f"Question: {query}\n\nDocument content: no content blocks were detected.",
            'images': [],
            'block_count': 0,
            'total_detected': 0
        }
    
    #
    if use_graph:
        selected_blocks = reranker.rerank_blocks_ppr_hybrid_singlepage(
            blocks, query, img,
            top_k=top_k,
            alpha=0.75,
            lambda_mix=0.7,
            seed_boost_topm=5,
            seed_boost_each=0.15,
            exclude_types_for_p=('page',),
            weight_key='weight'
        )
    else:
        selected_blocks = reranker.rerank_blocks_simple(blocks, query, img, top_k)
    
    #
    text_prompt = BlockPromptGenerator.create_structured_prompt(
        selected_blocks, 
        query,
        page_id=1,
        llm_mode="text",
        require_json=False,
        include_scores=True
    )
    
    #
    image_regions = []
    for block in selected_blocks:
        if block.get('type') in ['figure', 'image', 'table']:
            bbox = block.get('bbox')
            if bbox:
                region = detector.extract_region_image(img, bbox)
                if region:
                    image_regions.append(region)
    
    multimodal_prompt, _ = BlockPromptGenerator.create_multimodal_prompt(
        selected_blocks, query, image_regions
    )
    
    return {
        'selected_blocks': selected_blocks,
        'text_prompt': text_prompt,
        'multimodal_prompt': multimodal_prompt,
        'images': image_regions,
        'block_count': len(selected_blocks),
        'total_detected': len(blocks)
    }


def process_sample_with_blocks(
    sample_data: Dict[str, Any],
    query: str,
    use_graph: bool = False,
    top_k: int = 5,
    global_top_k: Optional[int] = None,
    visual_top_k: Optional[int] = None,
    max_visual_per_page: Optional[int] = None,
    clip_model_path: str = os.environ.get('HYPERDOC_OPENCLIP_MODEL', ''),
    detector: Optional[BlockDetector] = None,
    reranker: Optional[BlockReranker] = None,
    #
    modality_weights: Optional[Dict[str, float]] = None,
    min_visual_ratio: Optional[float] = None,
    return_diagnostics: bool = True,
    strategy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """"""
    images = sample_data.get('images', [])
    if not images:
        return {
            'enhanced_prompt': query,
            'block_images': [],
            'block_info': {
                'block_count': 0,
                'total_detected': 0,
                'use_graph': use_graph
            }
        }
    
    #
    # page/global K
    if strategy is not None:
        if isinstance(strategy.get('per_page_top_k', None), int):
            top_k = int(strategy['per_page_top_k'])
        if isinstance(strategy.get('final_k', None), int):
            global_top_k = int(strategy['final_k'])
        #
        if strategy.get('modality_weights') is not None:
            modality_weights = strategy.get('modality_weights')
        if strategy.get('min_visual_ratio') is not None:
            min_visual_ratio = strategy.get('min_visual_ratio')

    #
    if use_graph:
        #
        if min_visual_ratio is None:
            min_visual_ratio = 0.3
            print("[process_sample] min_visual_ratio is not set; using default 0.3")
        
        #
        if detector is None:
            detector = get_global_detector(clip_model_path)
        if reranker is None:
            reranker = get_global_reranker(clip_model_path)
        #
        pages_blocks: List[List[Dict[str, Any]]] = []
        imgs: List[np.ndarray] = []
        total_detected = 0
        for image_path in images:
            try:
                blocks, img = detector.detect_blocks(image_path)
                pages_blocks.append(blocks)
                imgs.append(img)
                total_detected += len(blocks)
            except Exception as e:
                print(f"Failed to process image {image_path}: {e}")
                pages_blocks.append([])
                #
                dummy = np.zeros((8, 8, 3), dtype=np.uint8)
                imgs.append(dummy)
        #
        num_pages = max(1, len(images))
        #
        if isinstance(global_top_k, int) and global_top_k > 0:
            final_k = int(global_top_k)
        else:
            if num_pages == 1:
                final_k = 6
            elif num_pages == 2:
                final_k = 12
            else:
                final_k = min(5 * num_pages, 10)

        q_lower = (query or "").lower()
        #
        global_tokens = ["any of", "among all", "overall", "distribution", "proportion across", "throughout", "in total"]
        local_tokens = ["according to figure", "according to table", "in figure", "in table", "in section"]
        is_global_query = any(tok in q_lower for tok in global_tokens)
        is_local_query = any(tok in q_lower for tok in local_tokens)
        #
        pr_alpha = 0.75
        if is_global_query:
            pr_alpha = 0.70
            final_k = min(8 * num_pages, 15)  #
        elif is_local_query:
            pr_alpha = 0.80

        #
        pr_args = {}
        if strategy is not None and isinstance(strategy.get('pagerank'), dict):
            pr = strategy['pagerank']
            pr_args.update({
                'alpha': pr.get('alpha', 0.75),
                'lambda_mix': pr.get('lambda_mix', 0.7),
                'seed_boost_topm': pr.get('seed_boost_topm', 5),
                'seed_boost_each': pr.get('seed_boost_each', 0.15),
            })
        graph_args = {}
        if strategy is not None and isinstance(strategy.get('graph'), dict):
            graph = strategy['graph']
            graph_args.update({
                'caption_search_radius_ratio': graph.get('caption_search_radius_ratio', 0.4),
                'caption_candidate_page_gap': graph.get('caption_candidate_page_gap', 0),
                'caption_min_score': graph.get('caption_min_score', 0.55),
                'caption_weight_geo': graph.get('caption_weight_geo', 0.25),
                'caption_weight_clip': graph.get('caption_weight_clip', 0.5),
                'caption_weight_cue': graph.get('caption_weight_cue', 0.15),
                'caption_weight_hier': graph.get('caption_weight_hier', 0.10),
                'text_knn_k': graph.get('text_knn_k', 2),
                'text_knn_sim_thresh': graph.get('text_knn_sim_thresh', 0.35),
                'text_knn_weight': graph.get('text_knn_weight', 0.6),
                'title_horizontal_overlap_thresh': graph.get('title_horizontal_overlap_thresh', 0.25),
                'title_max_vertical_gap_ratio': graph.get('title_max_vertical_gap_ratio', 0.20),
                'title_stop_at_next_title': graph.get('title_stop_at_next_title', True),
            })

        rerank_result = reranker.rerank_blocks_ppr_hybrid_global_ss(
            pages_blocks, imgs, query,
            top_k=final_k,
            alpha=pr_args.get('alpha', pr_alpha),
            lambda_mix=pr_args.get('lambda_mix', 0.7),
            seed_boost_topm=pr_args.get('seed_boost_topm', 5),
            seed_boost_each=pr_args.get('seed_boost_each', 0.15),
            exclude_types_for_p=('page',),
            #
            caption_search_radius_ratio=graph_args.get('caption_search_radius_ratio', 0.4),
            caption_candidate_page_gap=graph_args.get('caption_candidate_page_gap', 0),
            caption_min_score=graph_args.get('caption_min_score', 0.55),
            caption_weight_geo=graph_args.get('caption_weight_geo', 0.25),
            caption_weight_clip=graph_args.get('caption_weight_clip', 0.5),
            caption_weight_cue=graph_args.get('caption_weight_cue', 0.15),
            caption_weight_hier=graph_args.get('caption_weight_hier', 0.10),
            text_knn_k=graph_args.get('text_knn_k', 2),
            text_knn_sim_thresh=graph_args.get('text_knn_sim_thresh', 0.35),
            text_knn_weight=graph_args.get('text_knn_weight', 0.6),
            title_horizontal_overlap_thresh=graph_args.get('title_horizontal_overlap_thresh', 0.25),
            title_max_vertical_gap_ratio=graph_args.get('title_max_vertical_gap_ratio', 0.20),
            title_stop_at_next_title=graph_args.get('title_stop_at_next_title', True),
            #
            modality_weights=modality_weights,
            min_visual_ratio=min_visual_ratio,
            return_diagnostics=return_diagnostics,
        )
        
        #
        if return_diagnostics and isinstance(rerank_result, tuple):
            selected_blocks, diagnostics = rerank_result
        else:
            selected_blocks = rerank_result
            diagnostics = None
        #
        if selected_blocks:
            enhanced_prompt = BlockPromptGenerator.create_structured_prompt(
                selected_blocks, query, page_id="multi", llm_mode="text", require_json=False, include_scores=True
            )
            multimodal_prompt = BlockPromptGenerator.create_structured_prompt(
                selected_blocks, query, page_id="multi", llm_mode="vlm", require_json=False, include_scores=True
            )
        else:
            enhanced_prompt = f"Question: {query}\n\nDocument content: no relevant content blocks were detected."
            multimodal_prompt = enhanced_prompt
        #
        block_images: List[Image.Image] = []
        visual_limit = int(visual_top_k) if isinstance(visual_top_k, int) and visual_top_k > 0 else int(final_k)
        per_page_cap = int(max_visual_per_page) if isinstance(max_visual_per_page, int) and max_visual_per_page > 0 else None
        visual_types = {'figure','image','table'}
        per_page_counter: Dict[int, int] = {}
        
        #
        visual_blocks_in_selected = [b for b in selected_blocks if b.get('type') in visual_types]
        print("[block_images] Extracting visual block images:")
        print(f"   - selected blocks: {len(selected_blocks)}")
        print(f"   - visual blocks: {total_visual} ({100*total_visual/total_blocks if total_blocks > 0 else 0:.1f}%)")
        print(f"   - visual_limit: {visual_limit}, per_page_cap: {per_page_cap}")
        print(f"   - image array length: {len(imgs)}")
        
        for b in selected_blocks:
            if len(block_images) >= visual_limit:
                print(f"   Reached visual limit {visual_limit}; stop extraction")
                break
            if b.get('type') not in visual_types:
                continue
            
            #
            p = int(b.get('page_id', 1)) - 1
            bbox = b.get('bbox')
            print(f"   Visual block: type={b.get('type')}, page_id={b.get('page_id')}, p={p}, bbox={'present' if bbox else 'missing'}")
            
            if per_page_cap is not None:
                c = per_page_counter.get(p, 0)
                if c >= per_page_cap:
                    print(f"      Per-page cap {per_page_cap} reached; skipping")
                    continue
            
            if not bbox:
                print("      Missing bbox; skipping")
                continue
            
            if p < 0 or p >= len(imgs):
                print(f"      Page index out of range: p={p}, len(imgs)={len(imgs)}; skipping")
                continue
                
            region = detector.extract_region_image(imgs[p], bbox)
            if region is not None:
                block_images.append(region)
                print(f"      Region image extracted (size={region.size})")
                if per_page_cap is not None:
                    per_page_counter[p] = per_page_counter.get(p, 0) + 1
            else:
                print("      extract_region_image returned None")
        try:
            print(f"[blocks] mode=graph pages={len(images)} selected_blocks={len(selected_blocks)} block_images={len(block_images)} final_k={final_k} visual_limit={visual_limit} per_page_cap={per_page_cap}")
        except Exception:
            pass
        result = {
            'enhanced_prompt': enhanced_prompt,
            'multimodal_prompt': multimodal_prompt,
            'block_images': block_images,
            'block_info': {
                'block_count': len(selected_blocks),
                'total_detected': total_detected,
                'use_graph': use_graph,
                'selected_blocks': selected_blocks,
                'processed_pages': len(images)
            }
        }
        
        #
        if diagnostics is not None:
            result['diagnostics'] = diagnostics
        else:
            #
            try:
                visual_types = {'figure','image','table'}
                text_types = {'text','title','list'}
                total_count = len(selected_blocks)
                visual_count = sum(1 for b in selected_blocks if (b.get('type') or '').lower() in visual_types)
                result['diagnostics'] = {
                    'visual_ratio': (visual_count / total_count) if total_count > 0 else 0.0,
                    'visual_count': visual_count,
                    'total_count': total_count
                }
            except Exception:
                pass
        
        return result

    #
    all_selected_blocks = []
    all_image_regions = []
    total_detected = 0
    for page_idx, image_path in enumerate(images):
        try:
            result = process_page_blocks(
                image_path,
                query,
                False,  #
                top_k,
                clip_model_path,
                detector=detector,
                reranker=reranker
            )
            #
            page_blocks = result['selected_blocks']
            for block in page_blocks:
                block['page_id'] = page_idx + 1
                block['source_image'] = image_path
            all_selected_blocks.extend(page_blocks)
            all_image_regions.extend(result['images'])
            total_detected += result['total_detected']
        except Exception as e:
            print(f"Failed to process image {image_path}: {e}")
            continue
    if all_selected_blocks:
        all_selected_blocks.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
        final_k = int(global_top_k) if isinstance(global_top_k, int) and global_top_k > 0 else top_k
        all_selected_blocks = all_selected_blocks[:final_k]
        enhanced_prompt = BlockPromptGenerator.create_structured_prompt(
            all_selected_blocks, query, page_id="multi", llm_mode="text", require_json=False, include_scores=True
        )
        multimodal_prompt = BlockPromptGenerator.create_structured_prompt(
            all_selected_blocks, query, page_id="multi", llm_mode="vlm", require_json=False, include_scores=True
        )
    else:
        enhanced_prompt = f"Question: {query}\n\nDocument content: no relevant content blocks were detected."
        multimodal_prompt = enhanced_prompt
    #
    simple_diag: Optional[Dict[str, Any]] = None
    if return_diagnostics:
        modality_dist: Dict[str, int] = {}
        for b in all_selected_blocks:
            t = (b.get('type') or 'unknown')
            modality_dist[t] = modality_dist.get(t, 0) + 1
        visual_types = {'figure','image','table'}
        text_types = {'text','title','list'}
        visual_count = sum(modality_dist.get(t, 0) for t in visual_types)
        text_count = sum(modality_dist.get(t, 0) for t in text_types)
        total_count = sum(modality_dist.values()) or 1
        simple_diag = {
            'modality_distribution': modality_dist,
            'visual_count': visual_count,
            'text_count': text_count,
            'total_count': total_count,
            'visual_ratio': visual_count/total_count,
            'text_ratio': text_count/total_count,
        }
    result_non_graph = {
        'enhanced_prompt': enhanced_prompt,
        'multimodal_prompt': multimodal_prompt,
        'block_images': all_image_regions,
        'block_info': {
            'block_count': len(all_selected_blocks),
            'total_detected': total_detected,
            'use_graph': use_graph,
            'selected_blocks': all_selected_blocks,
            'processed_pages': len(images)
        }
    }
    if simple_diag is not None:
        result_non_graph['diagnostics'] = simple_diag
    return result_non_graph


def expand_blocks_by_neighbors(
    current_blocks: List[Dict[str, Any]],
    pages_blocks: List[List[Dict[str, Any]]],
    imgs: List[np.ndarray],
    query: str,
    detector: Optional[BlockDetector] = None,
    reranker: Optional[BlockReranker] = None,
    expand_k: int = 5,
    mode: str = 'text',  # 'text' | 'visual' | 'mixed'
    use_graph: bool = True,
    clip_model_path: str = os.environ.get('HYPERDOC_OPENCLIP_MODEL', ''),
) -> List[Dict[str, Any]]:
    """"""
    if detector is None:
        detector = get_global_detector(clip_model_path)
    if reranker is None:
        reranker = get_global_reranker(clip_model_path)
    
    #
    current_block_ids = set()
    for b in current_blocks:
        page_id = b.get('page_id', 1)
        bbox = tuple(b.get('bbox', []))
        current_block_ids.add((page_id, bbox))
    
    #
    candidates: List[Tuple[Dict[str, Any], float]] = []  # (block, score)
    
    if use_graph:
        #
        G, id_map = reranker.build_semantic_structural_graph(pages_blocks, imgs)
        
        #
        current_node_ids = []
        for b in current_blocks:
            page_id = b.get('page_id', 1)
            bbox = b.get('bbox')
            if not bbox:
                continue
            #
            for nid, (p_idx, b_idx) in id_map.items():
                if p_idx + 1 == page_id:
                    pb = pages_blocks[p_idx][b_idx]
                    if pb.get('bbox') == bbox:
                        current_node_ids.append(nid)
                        break
        
        #
        neighbor_nodes = set()
        for nid in current_node_ids:
            # 1-hop
            neighbor_nodes.update(G.successors(nid))
            neighbor_nodes.update(G.predecessors(nid))
            #
            for n1 in list(G.successors(nid)) + list(G.predecessors(nid)):
                neighbor_nodes.update(G.successors(n1))
                neighbor_nodes.update(G.predecessors(n1))
        
        #
        neighbor_nodes -= set(current_node_ids)
        
        #
        query_emb = detector.get_clip_text_embedding(query)
        for nid in neighbor_nodes:
            if nid not in id_map:
                continue
            p_idx, b_idx = id_map[nid]
            block = pages_blocks[p_idx][b_idx]
            
            #
            btype = (block.get('type') or 'unknown').lower()
            if mode == 'text' and btype not in {'text', 'title', 'list'}:
                continue
            elif mode == 'visual' and btype not in {'figure', 'image', 'table'}:
                continue
            
            #
            emb = G.nodes[nid].get('embedding')
            if emb is None:
                emb = reranker._ensure_node_embedding(G.nodes[nid], imgs[p_idx] if p_idx < len(imgs) else None)
            
            score = 0.0
            if query_emb is not None and emb is not None:
                score = float(np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb)))
            
            block_copy = block.copy()
            block_copy['page_id'] = p_idx + 1
            candidates.append((block_copy, score))
    
    else:
        #
        query_emb = detector.get_clip_text_embedding(query)
        
        #
        current_pages = set(b.get('page_id', 1) for b in current_blocks)
        search_pages = set()
        for p in current_pages:
            search_pages.add(p)
            if p > 1:
                search_pages.add(p - 1)
            if p < len(pages_blocks):
                search_pages.add(p + 1)
        
        #
        for p_idx in range(len(pages_blocks)):
            page_id = p_idx + 1
            if page_id not in search_pages:
                continue
            
            for b_idx, block in enumerate(pages_blocks[p_idx]):
                bbox = tuple(block.get('bbox', []))
                if (page_id, bbox) in current_block_ids:
                    continue
                
                #
                btype = (block.get('type') or 'unknown').lower()
                if mode == 'text' and btype not in {'text', 'title', 'list'}:
                    continue
                elif mode == 'visual' and btype not in {'figure', 'image', 'table'}:
                    continue
                
                #
                emb = None
                if btype in {'text', 'title', 'list'}:
                    emb = detector.get_clip_text_embedding(block.get('text', ''))
                elif btype in {'figure', 'image', 'table'}:
                    img = imgs[p_idx] if p_idx < len(imgs) else None
                    if img is not None:
                        region = detector.extract_region_image(img, block.get('bbox'))
                        if region is not None:
                            emb = detector.get_clip_image_embedding(region)
                
                score = 0.0
                if query_emb is not None and emb is not None:
                    score = float(np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb)))
                
                block_copy = block.copy()
                block_copy['page_id'] = page_id
                candidates.append((block_copy, score))
    
    #
    candidates.sort(key=lambda x: x[1], reverse=True)
    expanded_blocks = [b for b, _ in candidates[:expand_k]]
    
    #
    result = current_blocks.copy()
    for new_block in expanded_blocks:
        page_id = new_block.get('page_id', 1)
        bbox = tuple(new_block.get('bbox', []))
        if (page_id, bbox) not in current_block_ids:
            result.append(new_block)
            current_block_ids.add((page_id, bbox))
    
    return result


# ========= New: Supporting Subgraph (InfoNCE-based Expansion & Pruning) =========
class SupportingSubgraph:
    """"""

    def __init__(
        self,
        reranker: BlockReranker,
        pages_blocks: List[List[Dict[str, Any]]],
        imgs: List[np.ndarray],
        G: nx.DiGraph,
        id_map: Dict[int, Tuple[int, int]],
        seed_node_id: int,
        tau: float = 0.07,
    ):
        self.reranker = reranker
        self.pages_blocks = pages_blocks
        self.imgs = imgs
        self.G = G
        self.id_map = id_map
        self.selected_node_ids: List[int] = [seed_node_id]
        self.node_scores: Dict[int, float] = {seed_node_id: 1.0}
        self.tau = max(1e-6, float(tau))
        self._prev_selected_snapshot: List[int] = []
        self._prev_avg_score: float = 0.0

    def _ensure_node_embedding(self, nid: int) -> Optional[np.ndarray]:
        emb = self.G.nodes[nid].get('embedding')
        if emb is not None:
            return emb
        #
        p_idx = int(self.G.nodes[nid].get('page_id', 1)) - 1
        return self.reranker._ensure_node_embedding(self.G.nodes[nid], self.imgs[p_idx] if 0 <= p_idx < len(self.imgs) else None)

    def get_current_blocks(self) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        for nid in self.selected_node_ids:
            if nid not in self.id_map:
                continue
            p_idx, b_idx = self.id_map[nid]
            if 0 <= p_idx < len(self.pages_blocks) and 0 <= b_idx < len(self.pages_blocks[p_idx]):
                b = self.pages_blocks[p_idx][b_idx].copy()
                b['page_id'] = p_idx + 1
                blocks.append(b)
        return blocks

    def _neighbors_of_subgraph(self) -> List[int]:
        cand: set = set()
        for nid in self.selected_node_ids:
            for v in self.G.successors(nid):
                if v not in self.selected_node_ids:
                    cand.add(v)
            for v in self.G.predecessors(nid):
                if v not in self.selected_node_ids:
                    cand.add(v)
        return list(cand)

    def _sample_negatives(self, k: int, exclude: set) -> List[int]:
        #
        import random
        all_nodes = list(self.G.nodes())
        pool = [n for n in all_nodes if n not in exclude]
        if not pool:
            return []
        if k >= len(pool):
            return pool
        return random.sample(pool, k)

    def score_candidates_with_infonce(
        self,
        clue_text: str,
        negatives_per_pos: int = 8,
    ) -> Dict[int, float]:
        """"""
        #
        clue_emb = self.reranker.detector.get_clip_text_embedding(clue_text or "")
        if clue_emb is None:
            return {}

        #
        candidates = self._neighbors_of_subgraph()
        if not candidates:
            try:
                all_nodes = list(self.G.nodes())
                selected = set(self.selected_node_ids)
                candidates = [n for n in all_nodes if n not in selected]
            except Exception:
                candidates = []
        if not candidates:
            return {}

        #
        cand_embs: Dict[int, Optional[np.ndarray]] = {}
        for nid in candidates:
            cand_embs[nid] = self._ensure_node_embedding(nid)

        #
        scores: Dict[int, float] = {}
        exclude = set(self.selected_node_ids) | set(candidates)
        negs = self._sample_negatives(max(negatives_per_pos * max(1, len(candidates)), 8), exclude)

        #
        neg_embs: List[np.ndarray] = []
        for nid in negs:
            e = self._ensure_node_embedding(nid)
            if e is not None:
                neg_embs.append(e)

        #
        def _cos(a: np.ndarray, b: np.ndarray) -> float:
            an = float(np.linalg.norm(a)); bn = float(np.linalg.norm(b))
            if an <= 0.0 or bn <= 0.0:
                return 0.0
            return float(np.dot(a, b) / (an * bn))

        for nid in candidates:
            e = cand_embs[nid]
            if e is None:
                continue
            pos = _cos(e, clue_emb)
            num = np.exp(pos / self.tau)
            den = num
            #
            if neg_embs:
                neg_sims = [max(-1.0, min(1.0, _cos(ne, clue_emb))) for ne in neg_embs]
                den += float(np.sum(np.exp(np.array(neg_sims, dtype=float) / self.tau)))
            score = float(np.log(max(den and num, 1e-12) and (num / max(den, 1e-12))))
            scores[nid] = score

        return scores

    def score_nodes_with_infonce(
        self,
        node_ids: List[int],
        clue_text: str,
        negatives_per_pos: int = 8,
    ) -> Dict[int, float]:
        """"""
        clue_emb = self.reranker.detector.get_clip_text_embedding(clue_text or "")
        if clue_emb is None or not node_ids:
            return {}
        #
        exclude = set(node_ids) | set(self.selected_node_ids)
        import random
        all_nodes = list(self.G.nodes())
        pool = [n for n in all_nodes if n not in exclude]
        negs = []
        if pool:
            k = max(negatives_per_pos * max(1, len(node_ids)), 8)
            if k >= len(pool):
                negs = pool
            else:
                negs = random.sample(pool, k)
        neg_embs: List[np.ndarray] = []
        for nid in negs:
            e = self._ensure_node_embedding(nid)
            if e is not None:
                neg_embs.append(e)
        def _cos(a: np.ndarray, b: np.ndarray) -> float:
            an = float(np.linalg.norm(a)); bn = float(np.linalg.norm(b))
            if an <= 0.0 or bn <= 0.0:
                return 0.0
            return float(np.dot(a, b) / (an * bn))
        scores: Dict[int, float] = {}
        for nid in node_ids:
            e = self._ensure_node_embedding(nid)
            if e is None:
                continue
            pos = _cos(e, clue_emb)
            num = np.exp(pos / self.tau)
            den = num
            if neg_embs:
                neg_sims = [max(-1.0, min(1.0, _cos(ne, clue_emb))) for ne in neg_embs]
                den += float(np.sum(np.exp(np.array(neg_sims, dtype=float) / self.tau)))
            scores[nid] = float(np.log(max(den and num, 1e-12) and (num / max(den, 1e-12))))
        return scores

    def score_candidates_hybrid(
        self,
        query_text: str,
        clue_text: str,
        alpha: float = 0.7,
        negatives_per_pos: int = 8,
        struct_mix: float = 0.5,
    ) -> Dict[int, float]:
        """"""
        alpha = float(min(max(alpha, 0.0), 1.0))
        struct_mix = float(min(max(struct_mix, 0.0), 1.0))

        inf = self.score_candidates_with_infonce(clue_text, negatives_per_pos=negatives_per_pos)

        #
        q_emb = self.reranker.detector.get_clip_text_embedding(query_text or "")
        if q_emb is None:
            q_emb = np.zeros_like(next((e for e in [self.G.nodes[n].get('embedding') for n in self.G.nodes()] if e is not None), np.zeros(512)))

        candidates = list(inf.keys()) if inf else self._neighbors_of_subgraph()
        if not candidates:
            return inf

        #
        cand_emb: Dict[int, Optional[np.ndarray]] = {}
        for nid in candidates:
            cand_emb[nid] = self._ensure_node_embedding(nid)

        def _cos(a: np.ndarray, b: np.ndarray) -> float:
            an = float(np.linalg.norm(a)); bn = float(np.linalg.norm(b))
            if an <= 0.0 or bn <= 0.0:
                return 0.0
            return float(np.dot(a, b) / (an * bn))

        #
        conn: Dict[int, float] = {}
        for nid in candidates:
            csum = 0.0
            for u in self.selected_node_ids:
                if self.G.has_edge(u, nid):
                    try:
                        csum += float(self.G[u][nid].get('weight', 0.0))
                    except Exception:
                        pass
                if self.G.has_edge(nid, u):
                    try:
                        csum += float(self.G[nid][u].get('weight', 0.0))
                    except Exception:
                        pass
            conn[nid] = csum

        #
        struct: Dict[int, float] = {}
        for nid in candidates:
            e = cand_emb[nid]
            s_sim = _cos(e, q_emb) if e is not None else 0.0
            s_conn = conn.get(nid, 0.0)
            struct[nid] = struct_mix * s_sim + (1.0 - struct_mix) * s_conn

        def _minmax_norm(d: Dict[int, float]) -> Dict[int, float]:
            if not d:
                return {}
            vs = list(d.values())
            lo, hi = float(min(vs)), float(max(vs))
            if hi - lo <= 1e-12:
                return {k: 0.0 for k in d}
            return {k: (float(v) - lo) / (hi - lo) for k, v in d.items()}

        inf_n = _minmax_norm(inf)
        struct_n = _minmax_norm(struct)

        hybrid: Dict[int, float] = {}
        keys = set(candidates)
        for nid in keys:
            s1 = inf_n.get(nid, 0.0)
            s2 = struct_n.get(nid, 0.0)
            hybrid[nid] = float(alpha * s1 + (1.0 - alpha) * s2)
        return hybrid

    def expand_topk(self, cand_scores: Dict[int, float], k: int) -> List[int]:
        if not cand_scores:
            return []
        order = sorted(cand_scores.items(), key=lambda t: t[1], reverse=True)
        picked = [nid for nid, _ in order[:max(0, int(k))]]
        added: List[int] = []
        for nid in picked:
            if nid not in self.selected_node_ids:
                self.selected_node_ids.append(nid)
                self.node_scores[nid] = float(cand_scores.get(nid, 0.0))
                added.append(nid)
        return added

    def prune_below(self, cand_scores: Dict[int, float], epsilon: float = -2.0) -> List[int]:
        #
        removed: List[int] = []
        keep = []
        for nid in self.selected_node_ids:
            sc = float(self.node_scores.get(nid, 0.0))
            #
            if nid in cand_scores:
                sc = float(cand_scores[nid])
                self.node_scores[nid] = sc
            if sc < float(epsilon) and len(self.selected_node_ids) - len(removed) > 1:
                removed.append(nid)
            else:
                keep.append(nid)
        self.selected_node_ids = keep
        return removed

    def has_converged(self, min_delta: float = 1e-3) -> bool:
        #
        same_set = set(self._prev_selected_snapshot) == set(self.selected_node_ids)
        avg_score = float(np.mean([self.node_scores.get(n, 0.0) for n in self.selected_node_ids])) if self.selected_node_ids else 0.0
        delta = abs(avg_score - self._prev_avg_score)
        self._prev_selected_snapshot = list(self.selected_node_ids)
        self._prev_avg_score = avg_score
        return same_set and (delta < float(min_delta))


def construct_global_graph_for_sample(
    reranker: BlockReranker,
    pages_blocks: List[List[Dict[str, Any]]],
    imgs: List[np.ndarray],
) -> Tuple[nx.DiGraph, Dict[int, Tuple[int, int]]]:
    """"""
    G, id_map = reranker.build_semantic_structural_graph(
        pages_blocks, imgs,
        #
        title_horizontal_overlap_thresh=0.25,
        title_max_vertical_gap_ratio=0.20,
        title_stop_at_next_title=True,
        text_knn_k=2,
        text_knn_sim_thresh=0.35,
        text_knn_weight=0.6,
        caption_search_radius_ratio=0.4,
        caption_candidate_page_gap=0,
        caption_min_score=0.55,
        caption_weight_geo=0.25,
        caption_weight_clip=0.5,
        caption_weight_cue=0.15,
        caption_weight_hier=0.10,
        min_edge_weight=1e-6,
    )
    return G, id_map


def pick_seed_node_from_page(
    reranker: BlockReranker,
    page_blocks: List[Dict[str, Any]],
    page_img: np.ndarray,
    query: str,
    id_map: Dict[int, Tuple[int, int]],
    page_index_0_based: int,
) -> Optional[int]:
    """"""
    if not page_blocks:
        return None
    top1 = reranker.rerank_blocks_simple(page_blocks, query, page_img, top_k=1)
    if not top1:
        return None
    seed_bbox = top1[0].get('bbox')
    if seed_bbox is None:
        return None
    for nid, (p_idx, b_idx) in id_map.items():
        if p_idx == page_index_0_based and 0 <= b_idx < len(page_blocks):
            try:
                if page_blocks[b_idx].get('bbox') == seed_bbox:
                    return nid
            except Exception:
                continue
    return None



def export_graph_from_sample(
    sample_data: Dict[str, Any],
    query: str,
    export_path: str,
    clip_model_path: str = os.environ.get('HYPERDOC_OPENCLIP_MODEL', ''),
    detector: Optional[BlockDetector] = None,
    reranker: Optional[BlockReranker] = None,
    return_graph: bool = False,
    return_images: bool = False,
    use_semantic_structural_builder: bool = False,
    #
    intra_use_knn: bool = True,
    k_intra_spatial: int = 6,
    intra_weight_boost: float = 0.5,
    cross_candidate_mode: str = 'adjacent_or_explicit',
    cross_type_constraints: bool = True,
    cross_min_token_overlap: float = 0.0,
    cross_out_degree_cap: Optional[int] = 6,
    explicit_ref_boost: float = 0.2,
    connect_adjacent_only: bool = True,
    cross_top_k: int = 3,
    cross_sim_thresh: float = 0.2,
    page_gap_penalty: float = 0.3,
    spatial_distance_scale: float = 300.0,
    iou_weight: float = 1.0,
    min_edge_weight: float = 1e-6,
    #
    add_page_nodes: bool = False,
    page_anchor_weight: float = 0.25,
    page_anchor_out_degree_cap: Optional[int] = None,
    add_title_affinity_edges: bool = True,
    title_affinity_weight: float = 0.6,
    title_horizontal_overlap_thresh: float = 0.2,
    title_max_vertical_gap_ratio: float = 0.25,
    title_affinity_out_degree_cap: Optional[int] = None,
    title_stop_at_next_title: bool = True,
    #
    ss_text_knn_k: int = 2,
    ss_text_knn_sim_thresh: float = 0.3,
    ss_text_knn_weight: float = 0.6,
    ss_title_child_weight: float = 0.8,
    ss_before_weight: float = 0.5,
    ss_inter_caption_weight: float = 0.9,
    ss_caption_horizontal_overlap_thresh: float = 0.2,
    ss_caption_max_vertical_gap_ratio: float = 0.2,
    #
    ss_caption_search_radius_ratio: float = 0.4,
    ss_caption_candidate_page_gap: int = 0,
    ss_caption_min_score: float = 0.5,
    ss_caption_weight_geo: float = 0.25,
    ss_caption_weight_clip: float = 0.5,
    ss_caption_weight_cue: float = 0.15,
    ss_caption_weight_hier: float = 0.10,
    #
    add_semantic_cooccur_edges: bool = False,
    cooccur_within_same_title: bool = True,
    cooccur_sim_thresh: float = 0.4,
    cooccur_token_overlap_thresh: float = 0.2,
    cooccur_top_k: int = 1,
    cooccur_weight: float = 0.4,
) -> Dict[str, Any]:
    """"""
    if detector is None:
        detector = get_global_detector(clip_model_path)
    if reranker is None:
        reranker = get_global_reranker(clip_model_path)

    images = sample_data.get('images', [])
    if not images:
        return {'ok': False, 'reason': 'no images', 'summary': {}}

    pages_blocks: List[List[Dict[str, Any]]] = []
    imgs: List[np.ndarray] = []
    for image_path in images:
        try:
            blocks, img = detector.detect_blocks(image_path)
            pages_blocks.append(blocks)
            imgs.append(img)
        except Exception as e:
            return {'ok': False, 'reason': f'failed on {image_path}: {e}', 'summary': {}}

    if use_semantic_structural_builder:
        G, id_map = reranker.build_semantic_structural_graph(
            pages_blocks, imgs,
            title_horizontal_overlap_thresh=title_horizontal_overlap_thresh,
            title_max_vertical_gap_ratio=title_max_vertical_gap_ratio,
            title_stop_at_next_title=title_stop_at_next_title,
            title_child_weight=ss_title_child_weight,
            before_weight=ss_before_weight,
            text_knn_k=ss_text_knn_k,
            text_knn_sim_thresh=ss_text_knn_sim_thresh,
            text_knn_weight=ss_text_knn_weight,
            caption_horizontal_overlap_thresh=ss_caption_horizontal_overlap_thresh,
            caption_max_vertical_gap_ratio=ss_caption_max_vertical_gap_ratio,
            inter_caption_weight=ss_inter_caption_weight,
            caption_search_radius_ratio=ss_caption_search_radius_ratio,
            caption_candidate_page_gap=ss_caption_candidate_page_gap,
            caption_min_score=ss_caption_min_score,
            caption_weight_geo=ss_caption_weight_geo,
            caption_weight_clip=ss_caption_weight_clip,
            caption_weight_cue=ss_caption_weight_cue,
            caption_weight_hier=ss_caption_weight_hier,
            add_semantic_cooccur_edges=add_semantic_cooccur_edges,
            cooccur_within_same_title=cooccur_within_same_title,
            cooccur_sim_thresh=cooccur_sim_thresh,
            cooccur_token_overlap_thresh=cooccur_token_overlap_thresh,
            cooccur_top_k=cooccur_top_k,
            cooccur_weight=cooccur_weight,
            min_edge_weight=min_edge_weight,
        )
        stats = reranker.summarize_graph(G)
        if export_path:
            reranker.export_graph(G, export_path)
    else:
        G, id_map, stats = reranker.debug_build_graph_from_pages(
            pages_blocks,
            imgs,
            export_path=export_path,
            intra_use_knn=intra_use_knn,
            k_intra_spatial=k_intra_spatial,
            intra_weight_boost=intra_weight_boost,
            cross_candidate_mode=cross_candidate_mode,
            cross_type_constraints=cross_type_constraints,
            cross_min_token_overlap=cross_min_token_overlap,
            cross_out_degree_cap=cross_out_degree_cap,
            explicit_ref_boost=explicit_ref_boost,
            connect_adjacent_only=connect_adjacent_only,
            cross_top_k=cross_top_k,
            cross_sim_thresh=cross_sim_thresh,
            page_gap_penalty=page_gap_penalty,
            spatial_distance_scale=spatial_distance_scale,
            iou_weight=iou_weight,
            min_edge_weight=min_edge_weight,
            add_page_nodes=add_page_nodes,
            page_anchor_weight=page_anchor_weight,
            page_anchor_out_degree_cap=page_anchor_out_degree_cap,
            add_title_affinity_edges=add_title_affinity_edges,
            title_affinity_weight=title_affinity_weight,
            title_horizontal_overlap_thresh=title_horizontal_overlap_thresh,
            title_max_vertical_gap_ratio=title_max_vertical_gap_ratio,
            title_affinity_out_degree_cap=title_affinity_out_degree_cap,
            title_stop_at_next_title=title_stop_at_next_title,
        )

    result: Dict[str, Any] = {
        'ok': True,
        'summary': stats,
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'export_path': export_path
    }
    if return_graph:
        result['graph'] = G
    if return_images:
        result['images'] = imgs
    return result


# ---------- Visualization Utilities (for quick inspection) ----------
def visualize_graph_on_image(
    G: nx.DiGraph,
    imgs: List[np.ndarray],
    page_id: int = 1,
    draw_edge_types: Optional[List[str]] = None,
    node_label: bool = True,
    thickness: int = 2,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """"""
    assert 1 <= page_id <= len(imgs), f"page_id out of range: {page_id}"
    vis = imgs[page_id - 1].copy()

    #
    color_node = {
        'title': (0, 165, 255),          #
        'text': (40, 200, 40),           #
        'list': (40, 200, 120),          #
        'table': (0, 0, 255),            #
        'figure': (0, 0, 255),           #
        'image': (0, 0, 255),            #
        'table_caption': (200, 0, 200),  #
        'figure_caption': (200, 0, 200), #
        'caption': (200, 0, 200),        #
        'page': (180, 180, 180),         #
    }
    color_edge = {
        'intra-page': (255, 128, 0),     #
        'spatial': (255, 128, 0),
        'cross-page': (255, 0, 255),     #
        'page-anchor': (0, 165, 255),    #
        'has-child': (0, 255, 255),      #
        'title-affinity': (0, 255, 255),
        'before': (128, 128, 128),       #
        'semantic-tt': (0, 200, 200),    #
        'inter-explain': (0, 0, 255),    #
        'inter-cooccur': (150, 0, 150),  #
        'list-next': (0, 120, 255),
        'caption-affinity': (0, 0, 200),
        'page-chain': (120, 120, 0),
    }

    #
    default_edge_types = ['intra-page','spatial','has-child','title-affinity','before','semantic-tt','inter-explain','inter-cooccur']
    allowed_edge_types = set(draw_edge_types) if draw_edge_types else set(default_edge_types)

    def _center(b):
        x1, y1, x2, y2 = [int(v) for v in b]
        return (x1 + x2) // 2, (y1 + y2) // 2

    #
    page_nodes = [n for n in G.nodes() if int(G.nodes[n].get('page_id', 1)) == page_id]
    for n in page_nodes:
        nd = G.nodes[n]
        b = nd.get('bbox')
        if not b:
            continue
        t = (nd.get('type') or 'unknown').lower()
        col = color_node.get(t, (180, 180, 180))
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, thickness)
        if node_label:
            label = f"{n}:{t}"
            cv2.putText(vis, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

    #
    for u, v, d in G.edges(data=True):
        et = (d.get('type') or 'unknown').lower()
        if et not in allowed_edge_types:
            continue
        if int(G.nodes.get(u, {}).get('page_id', -1)) != page_id or int(G.nodes.get(v, {}).get('page_id', -1)) != page_id:
            continue
        bu = G.nodes[u].get('bbox'); bv = G.nodes[v].get('bbox')
        if not bu or not bv:
            continue
        c1 = _center(bu); c2 = _center(bv)
        col = color_edge.get(et, (100, 100, 255))
        cv2.line(vis, c1, c2, col, max(1, thickness-1), cv2.LINE_AA)
        #
        rel = d.get('rel') or d.get('direction')
        if rel:
            mid = ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2)
            cv2.putText(vis, str(rel), mid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    if save_path:
        try:
            import matplotlib.pyplot as plt  # type: ignore
            plt.figure(figsize=(12, 14))
            plt.imshow(rgb); plt.axis('off'); plt.tight_layout()
            plt.savefig(save_path, dpi=200); plt.close()
        except Exception:
            pass
    if show:
        try:
            import matplotlib.pyplot as plt  # type: ignore
            plt.figure(figsize=(12, 14))
            plt.imshow(rgb); plt.axis('off'); plt.tight_layout()
            plt.show()
        except Exception:
            pass
    return vis

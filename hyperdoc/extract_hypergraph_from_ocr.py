#!/usr/bin/env python3
"""Build HyperDoc hypergraphs directly from OCR JSON files."""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .hypergraph_utils import build_visual_groups


@dataclass
class Block:
    block_id: str
    type: str
    page: int
    bbox: List[int]
    text: Optional[str] = None
    rec_conf: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Page:
    page_id: str
    doc_id: str
    page_num: int
    pdf_path: str
    page_image_path: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Hyperedge:
    edge_id: str
    edge_type: str
    members: List[str]
    meta: Dict

    def to_dict(self) -> Dict:
        return asdict(self)


CAPTION_PATTERNS = [
    (r"^\s*Figure\s*\d+[\.:]", "figure_caption"),
    (r"^\s*FIGURE\s*\d+[\.:]", "figure_caption"),
    (r"^\s*Fig\.\s*\d+[\.:]", "figure_caption"),
    (r"^\s*Table\s*\d+[\.:]", "table_caption"),
    (r"^\s*TABLE\s*\d+[\.:]", "table_caption"),
    (r"^\s*Chart\s*\d+[\.:]", "chart_caption"),
    (r"^\s*Map\s*\d+[\.:]", "map_caption"),
    (r"^\s*Diagram\s*\d+[\.:]", "diagram_caption"),
    (r"^\s*Graph\s*\d+[\.:]", "graph_caption"),
]


def refine_block_type(text: str, current_type: str) -> str:
    if not text:
        return current_type
    for pattern, caption_type in CAPTION_PATTERNS:
        if re.match(pattern, text[:50], re.IGNORECASE):
            return caption_type
    return current_type


class HypergraphExtractorFromOCR:
    """Construct hypergraphs from OCR blocks using the HyperDoc visual grouping path."""

    def __init__(
        self,
        output_dir: str = "hypergraph",
        page_image_dir: str = "tmp/MMLongBench",
        pdf_base_dir: str = "data/MMLongBench/documents",
    ):
        self.output_dir = output_dir
        self.page_image_dir = page_image_dir
        self.pdf_base_dir = pdf_base_dir
        os.makedirs(output_dir, exist_ok=True)

    @staticmethod
    def classify_visual_type(caption_text: str, ocr_type: str) -> str:
        if caption_text:
            patterns = [
                (r"\bTable\s*\d+[\.:]", "table"),
                (r"\bTABLE\s*\d+[\.:]", "table"),
                (r"\bFigure\s*\d+[\.:]", "figure"),
                (r"\bFIGURE\s*\d+[\.:]", "figure"),
                (r"\bFig\.\s*\d+[\.:]", "figure"),
                (r"\bChart\s*\d+[\.:]", "chart"),
                (r"\bGraph\s*\d+[\.:]", "graph"),
                (r"\bMap\s*\d+[\.:]", "map"),
                (r"\bDiagram\s*\d+[\.:]", "diagram"),
            ]
            for pattern, visual_type in patterns:
                if re.search(pattern, caption_text[:150], re.IGNORECASE):
                    return visual_type
        return ocr_type

    def extract_document_hypergraph(self, ocr_file: str, semantic: bool = True) -> Dict:
        with open(ocr_file, "r", encoding="utf-8") as f:
            ocr_data = json.load(f)

        visual_groups = build_visual_groups(
            ocr_data,
            use_semantic_matching=semantic,
            image_dir=self.page_image_dir,
        )
        return self.extract_document_hypergraph_from_groups(ocr_data, visual_groups)

    def extract_document_hypergraph_from_groups(self, ocr_data: Dict, visual_groups: List[Dict]) -> Dict:
        doc_id = ocr_data.get("document_id") or "document"
        clean_doc_id = doc_id.replace(".pdf", "")
        pdf_path = self._resolve_pdf_path(ocr_data, doc_id)
        total_pages = int(ocr_data.get("total_pages", 0))

        blocks = []
        pages = []
        hyperedges = []
        anchor_nodes = []
        page_visual_members = defaultdict(list)
        section_visual_members = defaultdict(list)

        page_nodes = {}
        for page_num in range(total_pages):
            page_id = f"{clean_doc_id}_page_{page_num}"
            page_node = Page(
                page_id=page_id,
                doc_id=doc_id,
                page_num=page_num,
                pdf_path=pdf_path,
                page_image_path=os.path.join(self.page_image_dir, f"{clean_doc_id}_{page_num}.png"),
            )
            pages.append(page_node.to_dict())
            page_nodes[page_num] = page_id

        edge_counter = 0
        for unit in visual_groups:
            unit_id = unit["unit_id"]
            anchor_block = unit.get("anchor_block")
            if not anchor_block:
                continue

            caption_block = unit.get("caption_block")
            caption_text = caption_block.get("text", "") if caption_block else ""
            anchor_type = self.classify_visual_type(caption_text, anchor_block.get("type", "figure"))
            anchor_type = refine_block_type(anchor_block.get("text", ""), anchor_type)
            anchor_block_id = f"{clean_doc_id}_block_{unit_id}_anchor"
            anchor_node = Block(
                block_id=anchor_block_id,
                type=anchor_type,
                page=anchor_block.get("page", 0),
                bbox=anchor_block.get("bbox", []),
                text=anchor_block.get("text"),
                rec_conf=anchor_block.get("rec_conf"),
            )
            blocks.append(anchor_node.to_dict())
            anchor_nodes.append(anchor_node.to_dict())

            caption_block_id = None
            if caption_block and caption_block.get("text"):
                caption_block_id = f"{clean_doc_id}_block_{unit_id}_caption"
                caption_node = Block(
                    block_id=caption_block_id,
                    type=refine_block_type(caption_block.get("text", ""), caption_block.get("type", "figure_caption")),
                    page=caption_block.get("page", anchor_node.page),
                    bbox=caption_block.get("bbox", []),
                    text=caption_block.get("text"),
                    rec_conf=caption_block.get("rec_conf"),
                )
                blocks.append(caption_node.to_dict())

            title_block_id = None
            title_block = unit.get("context_title") or unit.get("title_block")
            if title_block and title_block.get("text"):
                title_block_id = f"{clean_doc_id}_block_{unit_id}_title"
                title_node = Block(
                    block_id=title_block_id,
                    type=title_block.get("type", "section_header"),
                    page=title_block.get("page", anchor_node.page),
                    bbox=title_block.get("bbox", []),
                    text=title_block.get("text"),
                    rec_conf=title_block.get("rec_conf"),
                )
                blocks.append(title_node.to_dict())

            related_block_ids = []
            for rel_idx, rel_text in enumerate(unit.get("related_texts", [])):
                if not rel_text.get("text"):
                    continue
                rel_block_id = f"{clean_doc_id}_block_{unit_id}_related_{rel_idx}"
                rel_node = Block(
                    block_id=rel_block_id,
                    type=refine_block_type(rel_text.get("text", ""), rel_text.get("type", "text")),
                    page=rel_text.get("page", anchor_node.page),
                    bbox=rel_text.get("bbox", []),
                    text=rel_text.get("text"),
                    rec_conf=rel_text.get("rec_conf"),
                )
                blocks.append(rel_node.to_dict())
                related_block_ids.append(rel_block_id)

            anchor_page = anchor_node.page
            if anchor_page in page_nodes:
                page_visual_members[anchor_page].append(anchor_block_id)
                edge_id = f"{clean_doc_id}_edge_{edge_counter}"
                edge_counter += 1
                hyperedges.append(
                    Hyperedge(
                        edge_id=edge_id,
                        edge_type="containment",
                        members=[page_nodes[anchor_page], anchor_block_id],
                        meta={
                            "hyperedge_id": edge_id,
                            "relation": "page_contains_visual",
                            "page_num": anchor_page,
                        },
                    ).to_dict()
                )

            if title_block_id:
                section_visual_members[title_block_id].append(anchor_block_id)
                edge_id = f"{clean_doc_id}_edge_{edge_counter}"
                edge_counter += 1
                hyperedges.append(
                    Hyperedge(
                        edge_id=edge_id,
                        edge_type="containment",
                        members=[title_block_id, anchor_block_id],
                        meta={
                            "hyperedge_id": edge_id,
                            "relation": "section_contains_visual",
                            "page_num": anchor_page,
                        },
                    ).to_dict()
                )

            contextual_members = [anchor_block_id]
            if caption_block_id:
                contextual_members.append(caption_block_id)
            if title_block_id:
                contextual_members.append(title_block_id)
            contextual_members.extend(related_block_ids)

            edge_id = f"{clean_doc_id}_edge_{edge_counter}"
            edge_counter += 1
            stats = unit.get("statistics", {})
            hyperedges.append(
                Hyperedge(
                    edge_id=edge_id,
                    edge_type="contextual",
                    members=contextual_members,
                    meta={
                        "hyperedge_id": edge_id,
                        "relation": "contextual",
                        "statistics": {
                            "has_caption": caption_block_id is not None,
                            "has_title": title_block_id is not None,
                            "num_related_texts": len(related_block_ids),
                            "total_blocks": len(contextual_members),
                            "page_span_size": stats.get("page_span_size", 1),
                            "is_cross_page": stats.get("is_cross_page", False),
                            "is_merged_block": stats.get("is_merged_block", False),
                        },
                    },
                ).to_dict()
            )

        if anchor_nodes:
            edge_id = f"{clean_doc_id}_edge_{edge_counter}"
            edge_counter += 1
            type_counts = {}
            for anchor in anchor_nodes:
                visual_type = anchor["type"]
                type_counts[visual_type] = type_counts.get(visual_type, 0) + 1
            hyperedges.append(
                Hyperedge(
                    edge_id=edge_id,
                    edge_type="containment",
                    members=[doc_id] + [anchor["block_id"] for anchor in anchor_nodes],
                    meta={
                        "hyperedge_id": edge_id,
                        "relation": "document_contains_visuals",
                        "num_visuals": len(anchor_nodes),
                        "visual_types": type_counts,
                    },
                ).to_dict()
            )

        for page_num, anchor_ids in sorted(page_visual_members.items()):
            edge_id = f"{clean_doc_id}_edge_{edge_counter}"
            edge_counter += 1
            hyperedges.append(
                Hyperedge(
                    edge_id=edge_id,
                    edge_type="containment",
                    members=[page_nodes[page_num]] + anchor_ids,
                    meta={
                        "hyperedge_id": edge_id,
                        "relation": "page_contains_visuals",
                        "page_num": page_num,
                        "num_visuals": len(anchor_ids),
                    },
                ).to_dict()
            )

        for title_block_id, anchor_ids in sorted(section_visual_members.items()):
            title_block = next((b for b in blocks if b.get("block_id") == title_block_id), {})
            edge_id = f"{clean_doc_id}_edge_{edge_counter}"
            edge_counter += 1
            hyperedges.append(
                Hyperedge(
                    edge_id=edge_id,
                    edge_type="containment",
                    members=[title_block_id] + anchor_ids,
                    meta={
                        "hyperedge_id": edge_id,
                        "relation": "section_contains_visuals",
                        "page_num": title_block.get("page"),
                        "section_title": title_block.get("text", ""),
                        "num_visuals": len(anchor_ids),
                    },
                ).to_dict()
            )

        return {
            "document_id": doc_id,
            "pdf_path": pdf_path,
            "total_pages": total_pages,
            "extraction_timestamp": datetime.now().isoformat(),
            "blocks": blocks,
            "pages": pages,
            "hyperedges": hyperedges,
            "statistics": self._statistics(blocks, pages, hyperedges, visual_groups),
        }

    def write_document_hypergraph(self, ocr_file: str, semantic: bool = True) -> Path:
        graph = self.extract_document_hypergraph(ocr_file, semantic=semantic)
        doc_id = graph["document_id"].replace(".pdf", "")
        out_path = Path(self.output_dir) / f"{doc_id}_hypergraph.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2, ensure_ascii=False)
        return out_path

    def _resolve_pdf_path(self, ocr_data: Dict, doc_id: str) -> str:
        if ocr_data.get("pdf_path"):
            return ocr_data["pdf_path"]
        pdf_name = doc_id if doc_id.endswith(".pdf") else f"{doc_id}.pdf"
        return os.path.join(self.pdf_base_dir, pdf_name)

    @staticmethod
    def _statistics(blocks: List[Dict], pages: List[Dict], hyperedges: List[Dict], visual_groups: List[Dict]) -> Dict:
        def count(edge_type: str, relation: Optional[str] = None) -> int:
            total = 0
            for edge in hyperedges:
                if edge.get("edge_type") != edge_type:
                    continue
                if relation is not None and edge.get("meta", {}).get("relation") != relation:
                    continue
                total += 1
            return total

        return {
            "num_blocks": len(blocks),
            "num_pages": len(pages),
            "num_hyperedges": len(hyperedges),
            "num_containment_edges": count("containment"),
            "num_contextual_edges": count("contextual"),
            "num_page_containment": count("containment", "page_contains_visual"),
            "num_section_containment": count("containment", "section_contains_visual"),
            "num_page_group_containment": count("containment", "page_contains_visuals"),
            "num_section_group_containment": count("containment", "section_contains_visuals"),
            "num_document_containment": count("containment", "document_contains_visuals"),
            "source_visual_groups": len(visual_groups),
            "source_visual_units": len(visual_groups),
        }

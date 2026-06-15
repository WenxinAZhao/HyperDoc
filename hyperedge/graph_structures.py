#!/usr/bin/env python3
""""""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class Block:
    """"""
    block_id: str
    type: str
    page: Optional[int] = None
    bbox: Optional[List[int]] = None
    text: Optional[str] = None
    rec_conf: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        return f"Block(id={self.block_id}, type={self.type}, page={self.page})"
    
    @staticmethod
    def extract_text_from_blocks(blocks: List['Block']) -> str:
        """"""
        texts = [b.text for b in blocks if b and b.text]
        return ' '.join(texts)
    
    @staticmethod
    def extract_text_from_dicts(blocks: List[Dict]) -> str:
        """"""
        texts = [b.get('text', '') for b in blocks if b and b.get('text')]
        return ' '.join(texts)


@dataclass
class Page:
    """"""
    page_id: str
    page_num: int
    pdf_path: Optional[str] = None
    image_path: Optional[str] = None
    doc_id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        return f"Page(id={self.page_id}, num={self.page_num})"


@dataclass
class QueryHyperedge:
    """"""
    anchor_blocks: List[Block] = field(default_factory=list)
    caption_blocks: List[Block] = field(default_factory=list)
    related_blocks: List[Block] = field(default_factory=list)
    source_pages: List[Page] = field(default_factory=list)
    
    score: float = 0.0
    match_reason: str = ""
    matched_keywords: List[str] = field(default_factory=list)
    tool: str = ""
    
    # For Document-level / Structured tasks
    count_result: Optional[int] = None
    enumerated_items: Optional[List[Dict]] = None
    
    raw_dict: Dict[str, Any] = field(default_factory=dict)
    
    # @classmethod
    # def from_router_result(cls, result: Dict, doc_id: str = "") -> 'QueryHyperedge':
    #     """
    #
        
    #     Args:
    #
    #
            
    #     Returns:
    #
    #     """
    #     anchor_dict = result.get('anchor_block')
    #     caption_dict = result.get('caption_block')
    #     related_dicts = result.get('related_blocks', []) or []
        
    #     # Convert dicts to Block objects
    #     anchor_blocks = [cls._dict_to_block(anchor_dict)] if anchor_dict else []
    #     caption_blocks = [cls._dict_to_block(caption_dict)] if caption_dict else []
    #     related_blocks = [cls._dict_to_block(b) for b in related_dicts if b]
        
    #     # Extract unique pages
    #     page_nums = set()
    #     for b in anchor_blocks + caption_blocks + related_blocks:
    #         if b and b.page is not None:
    #             page_nums.add(b.page)
        
    #     # Convert to Page objects
    #     source_pages = [
    #         Page(page_id=f"page_{num}", page_num=num, doc_id=doc_id)
    #         for num in sorted(page_nums)
    #     ]
        
    #     return cls(
    #         anchor_blocks=anchor_blocks,
    #         caption_blocks=caption_blocks,
    #         related_blocks=related_blocks,
    #         source_pages=source_pages,
    #         score=result.get('score', 0.0),
    #         match_reason=result.get('match_reason', ''),
    #         matched_keywords=result.get('matched_keywords', []),
    #         tool=result.get('tool', ''),
            
    #         # Special fields
    #         count_result=result.get('count'),
    #         enumerated_items=result.get('enumerated_items'),
            
    #
    #         raw_dict=result
    #     )

    @classmethod
    def create_from_components(
        cls,
        edge: Dict = None,
        anchor_block: Optional[Dict] = None,
        caption_block: Optional[Dict] = None,
        related_blocks: List[Dict] = None,
        score: float = 0.0,
        match_reason: str = "",
        matched_keywords: List[str] = None,
        tool: str = "",
        count_result: int = None,
        enumerated_items: List[Dict] = None,
        doc_id: str = "",
        extra_meta: Dict = None
    ) -> 'QueryHyperedge':
        """"""
        related_blocks = related_blocks or []
        extra_meta = extra_meta or {}
        
        # Convert dicts to Block objects
        anchor_blocks = [cls._dict_to_block(anchor_block)] if anchor_block else []
        caption_blocks = [cls._dict_to_block(caption_block)] if caption_block else []
        related_block_objs = [cls._dict_to_block(b) for b in related_blocks if b]
        
        # Extract unique pages
        page_nums = set()
        for b in anchor_blocks + caption_blocks + related_block_objs:
            if b and b.page is not None:
                page_nums.add(b.page)
        
        # Convert to Page objects
        source_pages = [
            Page(page_id=f"page_{num}", page_num=num, doc_id=doc_id)
            for num in sorted(page_nums)
        ]
        
        # Construct raw dict equivalent for compatibility/logging
        raw_dict = {
            'query_hyperedge': edge,
            'anchor_block': anchor_block,
            'caption_block': caption_block,
            'related_blocks': related_blocks,
            'score': score,
            'match_reason': match_reason,
            'matched_keywords': matched_keywords,
            'tool': tool,
            'count': count_result,
            'enumerated_items': enumerated_items,
            **extra_meta
        }
        
        return cls(
            anchor_blocks=anchor_blocks,
            caption_blocks=caption_blocks,
            related_blocks=related_block_objs,
            source_pages=source_pages,
            score=score,
            match_reason=match_reason,
            matched_keywords=matched_keywords or [],
            tool=tool,
            count_result=count_result,
            enumerated_items=enumerated_items,
            raw_dict=raw_dict
        )
    
    @staticmethod
    def _dict_to_block(d: Dict) -> Optional[Block]:
        """"""
        if not d:
            return None
        return Block(
            block_id=d.get('block_id', ''),
            type=d.get('type', ''),
            page=d.get('page'),
            bbox=d.get('bbox'),
            text=d.get('text'),
            rec_conf=d.get('rec_conf'),
            meta=d.get('meta', {})
        )
    
    def to_vlm_input(
        self,
        doc_id: str,
        page_image_dir: str,
        candidate_idx: int = 1
    ) -> tuple[List[str], str]:
        """"""
        from PIL import Image
        import tempfile
        import os
        
        images = []
        texts = []
        added_pages = set()

        # 0. Handle page-level nodes (created by tool_visual_search)
        #    These carry no anchor/caption/related blocks; the target page is
        #    stored only in raw_dict via extra_meta.
        if self.raw_dict.get('is_page_node'):
            src_page = self.raw_dict.get('source_page')
            if src_page is not None:
                page_img_path = os.path.join(
                    page_image_dir,
                    f"{doc_id}_{src_page}.png"
                )
                if os.path.exists(page_img_path):
                    images.append(page_img_path)
            return images, ""

        #
        for anchor in self.anchor_blocks:
            if anchor and anchor.page is not None and anchor.bbox:
                page_img_path = os.path.join(
                    page_image_dir,
                    f"{doc_id}_{anchor.page}.png"
                )
                
                if os.path.exists(page_img_path):
                    try:
                        page_img = Image.open(page_img_path)
                        x1, y1, x2, y2 = anchor.bbox
                        
                        # Add padding
                        padding = 10
                        x1 = max(0, x1 - padding)
                        y1 = max(0, y1 - padding)
                        x2 = min(page_img.width, x2 + padding)
                        y2 = min(page_img.height, y2 + padding)
                        
                        cropped = page_img.crop((x1, y1, x2, y2))
                        
                        temp_file = tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix='.png',
                            prefix=f'candidate_{candidate_idx}_'
                        )
                        cropped.save(temp_file.name)
                        temp_file.close()
                        
                        images.append(temp_file.name)
                    except Exception as e:
                        print(f"  ⚠️ Failed to crop anchor: {e}")
        
        #
        for page in self.source_pages:
            if page.page_num in added_pages:
                continue
            added_pages.add(page.page_num)
            
            page_img_path = os.path.join(
                page_image_dir,
                f"{doc_id}_{page.page_num}.png"
            )
            if os.path.exists(page_img_path):
                images.append(page_img_path)
        
        # 3. Extract text context (adaptive)
        #
        for caption in self.caption_blocks:
            if caption and caption.text:
                texts.append(caption.text)
        
        for related in self.related_blocks:
            if related and related.text:
                texts.append(related.text)
        
        #
        if texts:
            combined_text = " ".join(texts)  #
            context_text = f"Unit {candidate_idx}: {combined_text}"
        else:
            context_text = ""
        
        return images, context_text
    
    def get_all_page_nums(self) -> List[int]:
        """"""
        page_nums = set()
        for block in self.anchor_blocks + self.caption_blocks + self.related_blocks:
            if block and block.page is not None:
                page_nums.add(block.page)
        return sorted(page_nums)
    
    def __repr__(self):
        return (f"QueryHyperedge("
                f"|anchors|={len(self.anchor_blocks)}, "
                f"|captions|={len(self.caption_blocks)}, "
                f"|related|={len(self.related_blocks)}, "
                f"score={self.score:.4f}, "
                f"tool={self.tool})")


@dataclass
class Hypergraph:
    """
    Hypergraph Container
    Wraps the raw hypergraph dictionary and provides efficient indexing and access methods.
    """
    raw_data: Dict[str, Any]
    block_index: Dict[str, Dict] = field(init=False)
    
    def __post_init__(self):
        self.block_index = {b['block_id']: b for b in self.raw_data.get('blocks', [])}
        
    @property
    def blocks(self) -> List[Dict]:
        return self.raw_data.get('blocks', [])
    
    @property
    def hyperedges(self) -> List[Dict]:
        return self.raw_data.get('hyperedges', [])
        
    def get_block(self, block_id: str) -> Optional[Dict]:
        return self.block_index.get(block_id)
        
    def get_visual_contextual_edges(self) -> List[Dict]:
        """"""
        return [
            e for e in self.hyperedges
            if e.get('edge_type') == 'contextual'
            and e.get('meta', {}).get('relation') == 'contextual'
        ]

    def sub_graph(self, edge_filter_func) -> 'Hypergraph':
        """Create a subgraph based on edge filtering"""
        filtered_edges = [e for e in self.hyperedges if edge_filter_func(e)]
        filtered_member_ids = set()
        for edge in filtered_edges:
            filtered_member_ids.update(edge.get('members', []))
        
        new_data = {
            'blocks': [b for b in self.blocks if b['block_id'] in filtered_member_ids],
            'hyperedges': filtered_edges
        }
        return Hypergraph(raw_data=new_data)
        
    def __getitem__(self, key):
        return self.raw_data[key]
    
    def get(self, key, default=None):
        return self.raw_data.get(key, default)


# ============================================================================
#
# ============================================================================
#
#

Hyperedge = QueryHyperedge


__all__ = [
    "Block",
    "Page",
    "QueryHyperedge",
    "Hypergraph",
    "Hyperedge",  #
]

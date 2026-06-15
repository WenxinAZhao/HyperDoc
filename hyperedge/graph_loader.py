#!/usr/bin/env python3
""""""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from .graph_structures import Block, Page


def load_hypergraph(
    doc_id: str,
    hyperedges_dir: str = "hypergraph"
) -> Optional[Dict]:
    """"""
    clean_doc_id = doc_id.replace('.pdf', '')
    hypergraph_file = os.path.join(hyperedges_dir, f"{clean_doc_id}_hypergraph.json")
    
    if not os.path.exists(hypergraph_file):
        return None
    
    try:
        with open(hypergraph_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        #
        return data
        
    except Exception as e:
        print(f"Error loading hypergraph for {doc_id}: {e}")
        return None




def get_blocks_by_type(hypergraph: Dict, block_type: str) -> List[Dict]:
    """"""
    return [b for b in hypergraph['blocks'] if b['type'] == block_type]


def get_blocks_by_page(hypergraph: Dict, page_num: int) -> List[Dict]:
    """"""
    return [b for b in hypergraph['blocks'] if b['page'] == page_num]


def get_hyperedges_by_type(hypergraph: Dict, edge_type: str) -> List[Dict]:
    """"""
    return [e for e in hypergraph['hyperedges'] if e['edge_type'] == edge_type]


def get_containment_edges(hypergraph: Dict) -> List[Dict]:
    """"""
    return get_hyperedges_by_type(hypergraph, "containment")


def get_contextual_edges(hypergraph: Dict) -> List[Dict]:
    """"""
    return get_hyperedges_by_type(hypergraph, "contextual")


def find_block_by_id(hypergraph: Dict, block_id: str) -> Optional[Dict]:
    """"""
    for block in hypergraph['blocks']:
        if block['block_id'] == block_id:
            return block
    return None


def find_page_by_num(hypergraph: Dict, page_num: int) -> Optional[Dict]:
    """"""
    for page in hypergraph['pages']:
        if page['page_num'] == page_num:
            return page
    return None


def get_page_containment_edges(hypergraph: Dict, page_num: int) -> List[Dict]:
    """"""
    containment_edges = get_containment_edges(hypergraph)
    return [
        e for e in containment_edges 
        if e['meta'].get('page_num') == page_num
    ]


def build_block_index(hypergraph: Dict) -> Dict[str, Dict]:
    """"""
    return {b['block_id']: b for b in hypergraph['blocks']}


def build_page_index(hypergraph: Dict) -> Dict[int, Dict]:
    """"""
    return {p['page_num']: p for p in hypergraph['pages']}


def build_edge_index(hypergraph: Dict) -> Dict[str, List[Dict]]:
    """"""
    index = {}
    
    for edge in hypergraph['hyperedges']:
        for member_id in edge['members']:
            if member_id not in index:
                index[member_id] = []
            index[member_id].append(edge)
    
    return index


def get_related_blocks(
    hypergraph: Dict, 
    block_id: str, 
    edge_type: Optional[str] = None
) -> List[Dict]:
    """"""
    related_block_ids = set()
    
    for edge in hypergraph['hyperedges']:
        #
        if edge_type and edge['edge_type'] != edge_type:
            continue
        
        if block_id in edge['members']:
            #
            for member_id in edge['members']:
                if member_id != block_id:
                    related_block_ids.add(member_id)
    
    #
    related_blocks = []
    for b in hypergraph['blocks']:
        if b['block_id'] in related_block_ids:
            related_blocks.append(b)
    
    return related_blocks


def extract_contextual_text(hypergraph: Dict, edge: Dict) -> str:
    """"""
    texts = []
    
    for member_id in edge['members']:
        block = find_block_by_id(hypergraph, member_id)
        if block and block.get('text'):
            texts.append(block['text'])
    
    return ' '.join(texts)

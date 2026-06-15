""""""

from .prompts import (
    SCHEMA_PROMPT,
    COUNT_QUERY_PROMPT,
    COUNT_MATCHING_PROMPT,
    COUNT_FINAL_ANSWER_PROMPT,
    STANDARD_QA_PROMPT,
    HYPEREDGE_SCHEMA,
    allow_not_answerable_for_dataset,
    get_schema_prompt,
    format_count_matching_prompt,
    format_count_final_prompt,
    format_standard_qa_prompt
)

from .utils import (
    extract_answer,
    extract_count,
    extract_count_answer
)

from .intent_analyzer import (
    analyze_intent,
    parse_intent_response,
    is_count_query,
    is_unit_query,
    is_document_query,
    format_intent_summary
)

from .retrieval_router import (
    execute_hyperedge_retrieval,
    ROUTER_TABLE,
    tool_find_by_id,
    tool_find_by_page,
    tool_find_by_section,
    tool_multi_keyword_fetch,
    tool_keyword_search,
    tool_visual_search,
    tool_global_count,
    tool_enumerate
)

from .hyperedge_optimizer import (
    compute_coverage,
    select_hyperedges,
    _ef_prune_blocks,
    lexicographic_greedy_optimizer,
    maximize_coverage_optimizer,
    PHASE1_TOOLS,
)

from .utils import (
    extract_unit_text,
    encode_query_with_colbert,
    encode_text_with_colbert,
    compute_maxsim_score
)

from .graph_structures import (
    Block,
    Page,
    Hyperedge,
    QueryHyperedge
)

from .graph_loader import (
    load_hypergraph,
    get_blocks_by_type,
    get_blocks_by_page,
    get_hyperedges_by_type,
    get_containment_edges,
    get_contextual_edges,
    find_block_by_id,
    find_page_by_num,
    build_block_index,
    build_page_index,
    extract_contextual_text
)

__version__ = "1.0.0"

__all__ = [
    # Prompts
    "SCHEMA_PROMPT",
    "COUNT_QUERY_PROMPT",
    "STANDARD_QA_PROMPT",
    "HYPEREDGE_SCHEMA",
    "allow_not_answerable_for_dataset",
    "get_schema_prompt",
    "format_standard_qa_prompt",
    
    # Intent Analysis
    "analyze_intent",
    "parse_intent_response",
    "is_count_query",
    "is_unit_query",
    "is_document_query",
    "format_intent_summary",
    
    # Retrieval
    "execute_hyperedge_retrieval",
    "ROUTER_TABLE",
    
    # Utils
    "extract_unit_text",
    "encode_query_with_colbert",
    "encode_text_with_colbert",
    "compute_maxsim_score",
    "extract_answer",
    "extract_count",
    "extract_count_answer",
    
    # Graph Structures
    "Block",
    "Page",
    "Hyperedge",
    "QueryHyperedge",
    
    # Graph Loader
    "load_hypergraph",
    "get_blocks_by_type",
    "get_blocks_by_page",
    "get_hyperedges_by_type",
    "get_containment_edges",
    "get_contextual_edges",
    "find_block_by_id",
    "find_page_by_num",
    "build_block_index",
    "build_page_index",
    "extract_contextual_text"
]

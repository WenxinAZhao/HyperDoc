#!/usr/bin/env python3
""""""

import json


DOCBENCH_INTENT_DISABLED_TYPES = ("count", "enumerate")
FORCE_ANSWER_DATASET_NAMES = {
    "docbench",
    "ldu",
    "longdocurl",
}

# ============================================================================
# 1. Schema Definition
# ============================================================================

HYPEREDGE_SCHEMA = {
    "evidence_levels": [
        "UNIT",        # Single visual unit (anchor + caption + related text + metadata)
        "MULTI_UNIT",  # Multiple specific units to compare or synthesize
        "DOCUMENT"   ,  # global-scan evidence (counting/listing across the document)
        "null",      # Fallback: Similarity-based page retrieval for general Q&A
    ],
    "intent_types": [
        "locate",          # Find specific information/object
        "count",           # Count occurrences
        "enumerate",       # List all items of a type
        "compare",         # Compare two or more items
        "cross_reasoning"  # Cross-document reasoning
    ]
}

# ============================================================================
# 2. Schema Prompt for Intent Analysis
# ============================================================================

def get_intent_disabled_types_for_dataset(dataset_name: str = None):
    if dataset_name and dataset_name.lower() == "docbench":
        return list(DOCBENCH_INTENT_DISABLED_TYPES)
    return []


def allow_not_answerable_for_dataset(dataset_name: str = None) -> bool:
    dataset_key = str(dataset_name or "").strip().lower()
    return dataset_key not in FORCE_ANSWER_DATASET_NAMES


def get_schema_prompt(disabled_types=None) -> str:
    """"""
    disabled_types = list(disabled_types) if disabled_types else []
    schema = dict(HYPEREDGE_SCHEMA)
    if disabled_types:
        schema["intent_types"] = [
            intent_type
            for intent_type in HYPEREDGE_SCHEMA["intent_types"]
            if intent_type not in disabled_types
        ]
    schema_str = json.dumps(schema, indent=2)

    disabled_note = ""
    if disabled_types:
        disabled_note = (
            "\nDataset-specific instruction: the following intent types are not "
            f"used for this dataset and must not be selected: {disabled_types}. "
            "If a question appears to ask for counting or listing content from a "
            "specific table, figure, or section, map it to locate.\n"
        )

    intent_descriptions = [
        ("locate", "- **locate**: Find SPECIFIC info/units. Use this for \"how many\" or \"what are\" questions if they target the content INSIDE a single referenced unit (ID/Page/Spatial). \n  *Evidence Level*: locate type should be UNIT."),
        ("count", "- **count**: Statistical task. Count the occurrences of the UNITS themselves across the document scope.\n  *Evidence Level*: DOCUMENT."),
        ("enumerate", "- **enumerate**: Aggregation task. Extract and list all unique data points of a type found throughout the document.\n  *Evidence Level*: DOCUMENT."),
        ("compare", "- **compare**: Analyze differences/similarities between 2+ specific units."),
        ("cross_reasoning", "- **cross_reasoning**: Synthesize info from multiple separate units to reach a conclusion."),
    ]
    active_intent_descriptions = "\n".join(
        description
        for intent_type, description in intent_descriptions
        if intent_type not in disabled_types
    )
    active_intent_choices = " | ".join(
        intent_type
        for intent_type in HYPEREDGE_SCHEMA["intent_types"]
        if intent_type not in disabled_types
    )
    
    prompt = """
# Role
You are an expert at analyzing user queries for document retrieval. Your task is to extract the user's intent and map it to a structured JSON format to guide visual/textual tool routing.

# 1. Schema Definition
""" + schema_str + disabled_note + """# 2. Critical Rules
1). Evidence Level Logic (Target Selection)
- **UNIT**: Single specific block (Figure 1, the top table, a specific paragraph).
- **MULTI_UNIT**: Comparison or cross-reasoning between 2+ separate blocks that might not be in the same page or sections.
- **DOCUMENT**: Global tasks (counting/listing) across the entire document or large sections.
- **null**: FALLBACK. General text-based Q&A without specific figure/table references.
2). Constraints Extraction Rules
- **id**: Unit identifier (e.g., "3", "Figure 1", "Figure 1, Figure 10"). 
  * Use this field when query EXPLICITLY mentions Figure/Table/Chart numbers
  * For multiple IDs (compare/cross-reasoning), concatenate them
  * IMPORTANT: Do NOT put "Figure X" in keywords if the query explicitly names it - use id field instead
- **keywords**: 2–4 OCR-ready surface words (Entities, Years, Labels). Avoid abstract reasoning words like "trend", "difference", "increase", "compare". Only extract keywords that help IDENTIFY visual evidence, not words describing the reasoning task
- **page_range**: EXPANDED list of 1-based page numbers (always expand ranges). Examples:
  * Single page: "page 14" → [14]
  * Continuous range: "pages 5-7" → [5, 6, 7] (EXPAND the range)
  * Multiple discrete pages: "page 3, page 6, and page 14" → [3, 6, 14]
  * **Relative pages**: "last page" → ["LAST"], "first page" → ["FIRST"]
  * No mention → null
  IMPORTANT: 
  1. If you see "pages X-Y", expand it to the full list [X, X+1, ..., Y]. Do NOT output just [X, Y].
  2. For relative page references (last/first), output the STRING "LAST" or "FIRST" in the list, NOT unquoted words.
- **type**: figure | table | chart | text | mixed.
3). Intent Types
""" + active_intent_descriptions + """

# 3. Output Format
{{
  "intent_type": \"""" + active_intent_choices + """\",
  "target": {{
    "evidence_level": "UNIT | MULTI_UNIT | DOCUMENT | null",
    "constraints": {{
      "id": "Extracted ID (e.g., '3', 'Figure 3, Table 5') or null",
      "type": "figure | table | chart | text | mixed | null",
      "page_range": [list of page numbers] or null,
      "section_title": "Extracted section title or null",
      "keywords": ["list", "of", "keywords"] or null
    }}
  }}
}}

# 4. Examples

1. Query: "How many reasoning steps are involved in Figure 1?"
   Output:
   {{
     "intent_type": "locate",
     "target": {{
       "evidence_level": "UNIT",
       "constraints": {{
         "id": "1",
         "type": "figure",
         "page_range": null,
         "section_title": null,
         "keywords": ["reasoning steps"]
       }}
     }}
   }}

   
2. Query: "What are the words in the first rectangle on top of page 2?"
   Output:
   {{
     "intent_type": "locate",
     "target": {{
       "evidence_level": "UNIT",
       "constraints": {{
         "type": "figure",
         "page_range": [2],
         "section_title": null,
         "keywords": ["rectangle"]
       }}
     }}
   }}
   Reasoning: Spatial positioning ("top of the page") is not a routing constraint; use page_range to locate by page and add a keyword cue.

3. Query: "How many tables include 'F1' as a metric in this report?"
  Result:
  {{
    "intent_type": "count",
    "target": {
      "evidence_level": "DOCUMENT",
      "constraints": {
        "id": null,
        "type": "table",
        "page_range": null,
        "section_title": null,
        "keywords": ["F1", "metric"]
      }}
    }}



4. Query: "I’m at location 'J' shown in the campus map. Tell me the nearest coffee shop."
  Result:
  {{
    "intent_type": "cross_reasoning",
    "target": {
      "evidence_level": "MULTI_UNIT",
      "constraints": {
        "id": null,
        "type": "figure",
        "page_range": null,
        "section_title": null,
        "keywords": ["location J", "coffee shop", "campus map"]
      }}
    }}


5. Query: "According to the methodology section, what sampling method was used?"
   Output:
   {{
     "intent_type": "locate",
     "target": {{
       "evidence_level": "UNIT",
       "constraints": {{
         "section_title": "methodology",
         "keywords": ["sampling method"]
       }}
     }}
   }}

6. Query: "List the number of people in the figure in page 6, the number of buildings in page 14, and the number of legends in figure A in page 3."
   Output:
   {{
     "intent_type": "locate",
     "target": {{
       "evidence_level": "UNIT",
       "constraints": {{
         "id": null,
         "type": "figure",
         "page_range": [3, 6, 14],
         "section_title": null,
         "keywords": ["people", "buildings", "legends"]
       }}
     }}
   }}
   Reasoning: This is "locate" (not "count") because it asks for content INSIDE specific figures on specific pages, not counting figures themselves.

7. Query: "What figures appear in pages 5-7?"
   Output:
   {{
     "intent_type": "locate",
     "target": {{
       "evidence_level": "UNIT",
       "constraints": {{
         "id": null,
         "type": "figure",
         "page_range": [5, 6, 7],
         "section_title": null,
         "keywords": null
       }}
     }}
   }}
   Reasoning: "pages 5-7" is a continuous range, so expand it to [5, 6, 7]. Do NOT output [5, 7].

8. Query: "Which step in Figure 1 maps to the content of Figure 10?"
   Output:
   {{
     "intent_type": "compare",
     "target": {{
       "evidence_level": "MULTI_UNIT",
       "constraints": {{
         "id": "1, 10",
         "type": "figure",
         "page_range": null,
         "section_title": null,
         "keywords": ["step", "content", "maps"]
       }}
     }}
   }}

9. Query: "Who is the commanding officer at the last page?"
   Output:
   {{
     "intent_type": "locate",
     "target": {{
       "evidence_level": "UNIT",
       "constraints": {{
         "id": null,
         "type": "text",
         "page_range": ["LAST"],
         "section_title": null,
         "keywords": ["commanding officer"]
       }}
     }}
   }}
   Reasoning: For relative page references like "last page" or "first page", output the STRING "LAST" or "FIRST" (with quotes) in the page_range list.
   Reasoning: This is compare query with EXPLICIT multiple IDs. Put "Figure 1, Figure 10" in id field, NOT in keywords. Keywords should only contain the reasoning task words like "step", "content".
"""
    return prompt


# Alias for backward compatibility
SCHEMA_PROMPT = get_schema_prompt()

# ============================================================================
# 3. Count Query Prompt Template (Two-stage: Matching + Final Answer)
# ============================================================================

# Stage 1: Image matching with JSON output
COUNT_MATCHING_PROMPT = """You are asked to COUNT visual elements in the provided images.

Task: {question}

Instructions:
1) Examine EACH image.
2) For each image, decide Match = true/false based on the task criteria.
3) Provide a SHORT, evidence-based justification using visible cues only (e.g., title words, legend labels, axes, table headers).
4) Do NOT speculate. If uncertain, set Match=false and mark Uncertain=true.
5) Output MUST be valid JSON.

Output JSON schema:
{{
  "per_image": [
    {{
      "image_index": 1,
      "match": true|false,
      "uncertain": true|false,
      "evidence": "short phrase of what you saw that supports the decision"
    }}
  ],
  "count": <integer>
}}

Example output:
{{
  "per_image": [
    {{"image_index": 1, "match": true, "uncertain": false, "evidence": "title contains 'Latinos' and 'general public'"}},
    {{"image_index": 2, "match": false, "uncertain": false, "evidence": "only shows one demographic group"}},
    {{"image_index": 3, "match": true, "uncertain": false, "evidence": "legend shows both Latinos and general public"}}
  ],
  "count": 2
}}
"""

# Stage 2: Final answer with matched hyperedge
COUNT_FINAL_ANSWER_PROMPT = """Answer the counting question using EXACTLY this format:

<THINK>
[Analyze the matched visual elements and count them]
</THINK>

<answer>
[Final count as integer only]
</answer>

CRITICAL RULES:
1. You MUST use <THINK></THINK> tags for reasoning
2. You MUST use <answer></answer> tags for the final count
3. The answer must be ONLY a number (integer)
4. Do NOT output anything outside these tags

---

Question: {question}
"""


def format_count_matching_prompt(question: str) -> str:
    """"""
    return COUNT_MATCHING_PROMPT.format(question=question)


def format_count_final_prompt(question: str) -> str:
    """"""
    return COUNT_FINAL_ANSWER_PROMPT.format(question=question)


# Backward compatibility: Keep old COUNT_QUERY_PROMPT as alias
COUNT_QUERY_PROMPT = COUNT_MATCHING_PROMPT


# ============================================================================
# 4. Enumerate Extraction Prompt (Extract items across candidates)
# ============================================================================

ENUMERATE_EXTRACTION_PROMPT = """You are asked to EXTRACT a list of items from candidate visual elements.

Task: {{question}}

Instructions:
1) Examine EACH image independently.
2) Decide Match=true if the visual element is relevant to the task (e.g., a chart/graph when asked about charts).
3) If Match=true, extract the requested items EXACTLY as seen (e.g., years, categories). Use strings and preserve order.
4) If uncertain, set Match=false and Uncertain=true; items must be an empty list.
5) Output MUST be valid JSON.

Output JSON schema:
{{
  "per_image": [
    {{
      "image_index": 1,
      "match": true|false,
      "uncertain": true|false,
      "items": ["..."],
      "evidence": "short phrase of what you saw that supports the decision"
    }}
  ],
  "items": ["..."],
  "match_count": <integer>
}}

Example output:
{{
  "per_image": [
    {{"image_index": 1, "match": true, "uncertain": false, "items": ["2010", "2012"], "evidence": "x-axis labels show 2010, 2012"}},
    {{"image_index": 2, "match": false, "uncertain": false, "items": [], "evidence": "table, not a chart"}}
  ],
  "items": ["2010", "2012"],
  "match_count": 1
}}
"""


def format_enumerate_extraction_prompt(question: str) -> str:
    """"""
    return ENUMERATE_EXTRACTION_PROMPT.format(question=question)


# ============================================================================
# ============================================================================
# 5. Standard QA Prompt
# ============================================================================

#
STANDARD_QA_PROMPT_ALLOW_NA = """You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <THINK> </THINK> tags. The final answer MUST BE in <answer> </answer> tags and consistent with the reasoning conclusion. If the answer cannot be determined from the input, output <answer>not answerable</answer>."""
STANDARD_QA_PROMPT_FORCE_ANSWER = """You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <THINK> </THINK> tags. The final answer MUST BE in <answer> </answer> tags and consistent with the reasoning conclusion. Do NOT output "not answerable" (or similar). Provide a best-effort concrete answer based on the given evidence, even if uncertain."""
# Backward-compatible alias
STANDARD_QA_PROMPT = STANDARD_QA_PROMPT_ALLOW_NA


def format_standard_qa_prompt(question: str, context: str = None, allow_not_answerable: bool = True) -> str:
    """"""
    #
    #
    qa_instruction = STANDARD_QA_PROMPT_ALLOW_NA if allow_not_answerable else STANDARD_QA_PROMPT_FORCE_ANSWER
    if context:
        return f"Context from Document:\n{context}\n\nQuestion: {question}\n\nInstructions:\n{qa_instruction}"
    else:
        return f"Question: {question}\n\nInstructions:\n{qa_instruction}"


# ============================================================================
# 5. Intent Analysis Prompt Constructor
# ============================================================================

def build_intent_analysis_prompt(question: str, dataset_name: str = None) -> str:
    """"""
    disabled_types = get_intent_disabled_types_for_dataset(dataset_name)
    schema_prompt = get_schema_prompt(disabled_types=disabled_types)
    return f"{schema_prompt}\n\nUser Query: {question}\n\nAnalyze the query and output the Intent State JSON:"


# ============================================================================
# 6. Page Number Extraction Prompt (For Index Mapping)
# ============================================================================

PAGE_NUMBER_EXTRACTION_PROMPT = """Identify visible printed page numbers on this document page image.

Task: Extract specific page numbers to help map PDF index to printed pages.

Instructions:
1. Look carefully at all corners (top-left, top-right, bottom-left, bottom-right), bottom-center, and page headers/footers.
2. Determine if the image shows:
   - "single": A single page
   - "double": A double-page spread (two pages side-by-side, like in a book/magazine)
3. Extract ALL numeric page numbers visible as strings. If you see two page numbers (e.g., "4" and "5"), include both.
4. If Roman numerals (i, ii, iii, iv) are used, extract them as strings.
5. If no page numbers are found, return empty list [].
6. Important: For double-page spreads, left page number comes first, then right page number.

Output JSON schema:
{{
  "layout": "single" | "double",
  "found_pages": ["4", "5"]
}}

Examples:
- Single page with "10" printed → {{"layout": "single", "found_pages": ["10"]}}
- Double spread with "4" (left) and "5" (right) → {{"layout": "double", "found_pages": ["4", "5"]}}
- No page numbers visible → {{"layout": "single", "found_pages": []}}
"""

def format_page_extraction_prompt() -> str:
    """
    Format prompt for extracting page numbers to cure index mismatch.
    """
    return PAGE_NUMBER_EXTRACTION_PROMPT


# ============================================================================
# Export all prompts
# ============================================================================

__all__ = [
    "HYPEREDGE_SCHEMA",
    "SCHEMA_PROMPT",
    "COUNT_QUERY_PROMPT",
    "COUNT_MATCHING_PROMPT",
    "COUNT_FINAL_ANSWER_PROMPT",
    "ENUMERATE_EXTRACTION_PROMPT",
    "STANDARD_QA_PROMPT",
    "PAGE_NUMBER_EXTRACTION_PROMPT",
    "allow_not_answerable_for_dataset",
    "get_intent_disabled_types_for_dataset",
    "get_schema_prompt",
    "format_count_matching_prompt",
    "format_count_final_prompt",
    "format_enumerate_extraction_prompt",
    "format_standard_qa_prompt",
    "build_intent_analysis_prompt",
    "format_page_extraction_prompt",
]

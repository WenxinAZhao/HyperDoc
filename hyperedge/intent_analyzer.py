#!/usr/bin/env python3
"""Intent parsing for HyperDoc queries."""

import json
import re
from typing import Any, Dict, List, Optional

from .prompts import build_intent_analysis_prompt


def analyze_intent(
    question: str,
    reasoner: Any,
    dataset_name: Optional[str] = None,
    enable_validation: Optional[bool] = None,
    auto_fix: Optional[bool] = None,
    verbose_validation: Optional[bool] = None,
) -> Dict[str, Any]:
    """Parse a question into the intent state consumed by the router."""
    del enable_validation, auto_fix, verbose_validation
    intent_prompt = build_intent_analysis_prompt(question, dataset_name=dataset_name)

    try:
        intent_response, _ = reasoner.predict(intent_prompt, images=None)
        intent_state = parse_intent_response(intent_response)
        intent_state["raw_response"] = intent_response
        intent_state["status"] = "success"
        return intent_state
    except Exception as exc:
        print(f"Intent parsing failed: {exc}")
        return {
            "intent_type": "locate",
            "target": {"evidence_level": None, "constraints": {}},
            "raw_response": str(exc),
            "status": "failed",
        }


def expand_page_constraint(page_input: Optional[List]) -> List:
    """Normalize page_range to a sorted list of 1-based pages or markers."""
    if not page_input or not isinstance(page_input, list):
        return []

    valid_pages = []
    special_markers = []
    for page in page_input:
        if isinstance(page, int) and page > 0:
            valid_pages.append(page)
        elif isinstance(page, str) and page.upper() in {"LAST", "FIRST"}:
            special_markers.append(page.upper())

    result = sorted(set(valid_pages))
    result.extend(sorted(set(special_markers)))
    return result


def _normalize_intent_constraints(intent_state: Dict[str, Any]) -> None:
    target = intent_state.get("target")
    if not isinstance(target, dict):
        return
    constraints = target.get("constraints", {})
    if not isinstance(constraints, dict):
        target["constraints"] = {}
        return
    if "page" in constraints and "page_range" not in constraints:
        constraints["page_range"] = constraints.pop("page")
    if "page_range" in constraints:
        constraints["page_range"] = expand_page_constraint(constraints["page_range"])
    target["constraints"] = constraints


def parse_intent_response(response: str) -> Dict[str, Any]:
    """Extract an intent-state JSON object from a model response."""
    json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if not json_match:
            raise ValueError(f"Cannot extract JSON from response: {response[:200]}")
        json_str = json_match.group(0)

    try:
        intent_state = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON parsing failed: {exc}\nRaw content: {json_str[:200]}") from exc
    if "intent_type" not in intent_state:
        raise ValueError("Intent state is missing required field: intent_type")
    if "target" not in intent_state:
        raise ValueError("Intent state is missing required field: target")
    _normalize_intent_constraints(intent_state)
    return intent_state


def validate_intent_state(intent_state: Dict[str, Any]) -> bool:
    """Return whether an intent state has the expected top-level shape."""
    target = intent_state.get("target")
    if "intent_type" not in intent_state or not isinstance(target, dict):
        return False
    if "evidence_level" not in target or "constraints" not in target:
        return False
    evidence_level = target["evidence_level"]
    if evidence_level is not None and not isinstance(evidence_level, str):
        return False
    return isinstance(target["constraints"], dict)


def extract_constraints(intent_state: Dict[str, Any]) -> Dict[str, Any]:
    """Return the constraints dictionary from an intent state."""
    constraints = intent_state.get("target", {}).get("constraints", {})
    return constraints if isinstance(constraints, dict) else {}


def extract_spatial_hint(intent_state: Dict[str, Any]) -> Optional[str]:
    """Return the spatial hint from an intent state if present."""
    return extract_constraints(intent_state).get("spatial_hint")


def is_count_query(intent_state: Dict[str, Any]) -> bool:
    """Return whether the query is a count query."""
    return intent_state.get("intent_type") == "count"


def is_unit_query(intent_state: Dict[str, Any]) -> bool:
    """Return whether the query asks for unit-level evidence."""
    return intent_state.get("target", {}).get("evidence_level") == "UNIT"


def is_multi_unit_query(intent_state: Dict[str, Any]) -> bool:
    """Return whether the query asks for multiple evidence units."""
    return intent_state.get("target", {}).get("evidence_level") == "MULTI_UNIT"


def is_document_query(intent_state: Dict[str, Any]) -> bool:
    """Return whether the query asks for document-level evidence."""
    return intent_state.get("target", {}).get("evidence_level") == "DOCUMENT"


def format_intent_summary(intent_state: Dict[str, Any]) -> str:
    """Format an intent state for concise logging."""
    intent_type = intent_state.get("intent_type", "unknown")
    target = intent_state.get("target", {})
    evidence_level = target.get("evidence_level", "None")
    constraints = extract_constraints(intent_state)
    summary_parts = [f"Intent: {intent_type}", f"Evidence: {evidence_level}"]
    constraint_parts = []
    if constraints.get("id"):
        constraint_parts.append(f"ID={constraints['id']}")
    if constraints.get("type"):
        constraint_parts.append(f"Type={constraints['type']}")
    if constraints.get("page_range"):
        constraint_parts.append(f"Pages={constraints['page_range']}")
    if constraints.get("spatial_hint"):
        constraint_parts.append(f"Spatial={constraints['spatial_hint']}")
    if constraints.get("section_title"):
        constraint_parts.append(f"Section={constraints['section_title']}")
    if constraints.get("keywords"):
        keywords = constraints["keywords"]
        keywords_str = ", ".join(keywords[:3])
        if len(keywords) > 3:
            keywords_str += "..."
        constraint_parts.append(f"Keywords=[{keywords_str}]")
    if constraint_parts:
        summary_parts.append(f"Constraints: {' | '.join(constraint_parts)}")
    return " | ".join(summary_parts)


__all__ = [
    "analyze_intent",
    "parse_intent_response",
    "validate_intent_state",
    "extract_constraints",
    "extract_spatial_hint",
    "is_count_query",
    "is_unit_query",
    "is_multi_unit_query",
    "is_document_query",
    "format_intent_summary",
]

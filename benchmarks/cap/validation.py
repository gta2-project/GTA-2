"""Minimal parsing and interface extraction for model-generated CaP programs.

Generated policies are trusted code.  This module deliberately does not apply
an AST allowlist or reject Python constructs, calls, imports, mutation, loops,
or attribute access.  It retains only the structural checks required by the CaP
pipeline to obtain perception queries and call ``policy(ctx)``.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Mapping

from .conditions import ApiKind


class PolicyValidationError(ValueError):
    """Raised when source cannot satisfy the minimum CaP runtime interface."""


@dataclass(frozen=True)
class ValidatedPolicy:
    """Parsed policy retained under its legacy name for API compatibility."""

    source: str
    tree: ast.Module
    keypoint_queries: Mapping[str, Mapping[str, str]]


def strip_markdown_fence(source: str) -> str:
    text = source.strip()
    match = re.fullmatch(r"```(?:python)?\s*\n(.*)\n```", text, flags=re.DOTALL)
    return match.group(1).strip() if match else text


def _extract_queries(tree: ast.Module):
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "KEYPOINT_QUERIES"
    ]
    if len(assignments) != 1:
        raise PolicyValidationError(
            "Program must define one literal KEYPOINT_QUERIES dictionary"
        )
    try:
        queries = ast.literal_eval(assignments[0].value)
    except (ValueError, TypeError) as exc:
        raise PolicyValidationError(
            "KEYPOINT_QUERIES must be a literal dictionary"
        ) from exc
    if not isinstance(queries, dict):
        raise PolicyValidationError("KEYPOINT_QUERIES must be a literal dictionary")

    normalized = {}
    for alias, query in queries.items():
        if not isinstance(alias, str) or not alias.isidentifier():
            raise PolicyValidationError(f"Invalid keypoint alias: {alias!r}")
        if not isinstance(query, dict) or set(query) != {"object", "description"}:
            raise PolicyValidationError(
                f"Query {alias!r} must contain exactly 'object' and 'description'"
            )
        object_name = query["object"]
        description = query["description"]
        if not isinstance(object_name, str) or not object_name.strip():
            raise PolicyValidationError(f"Query {alias!r} has an empty object")
        if not isinstance(description, str) or not description.strip():
            raise PolicyValidationError(f"Query {alias!r} has an empty description")
        normalized[alias] = {
            "object": object_name.strip(),
            "description": description.strip(),
        }
    return normalized


def validate_policy(source: str, api_kind: ApiKind) -> ValidatedPolicy:
    """Parse trusted source and extract the minimum CaP runtime interface.

    ``api_kind`` remains in the signature for compatibility but intentionally
    does not alter or restrict accepted Python code.
    """

    del api_kind
    cleaned = strip_markdown_fence(source)
    try:
        tree = ast.parse(cleaned, mode="exec")
    except SyntaxError as exc:
        raise PolicyValidationError(
            f"Generated policy is not valid Python: {exc}"
        ) from exc

    policy_nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "policy"
    ]
    if len(policy_nodes) != 1:
        raise PolicyValidationError("Program must define policy(ctx) exactly once")
    policy_node = policy_nodes[0]
    if (
        isinstance(policy_node, ast.AsyncFunctionDef)
        or len(policy_node.args.args) != 1
        or policy_node.args.args[0].arg != "ctx"
        or policy_node.args.defaults
        or policy_node.args.vararg
        or policy_node.args.kwarg
        or policy_node.args.kwonlyargs
    ):
        raise PolicyValidationError("policy must have the exact signature policy(ctx)")

    return ValidatedPolicy(
        source=cleaned,
        tree=tree,
        keypoint_queries=_extract_queries(tree),
    )

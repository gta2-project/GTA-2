"""Condition definitions for the CaP benchmark matrix."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class ApiKind(str, Enum):
    CONTROLLER = "controller"
    CARTESIAN = "cartesian"


class CapCondition(str, Enum):
    CONTROLLER_ZEROSHOT = "controller_zeroshot"
    CONTROLLER_FEWSHOT = "controller_fewshot"
    CARTESIAN_ZEROSHOT = "cartesian_zeroshot"
    CARTESIAN_FEWSHOT = "cartesian_fewshot"


@dataclass(frozen=True)
class ConditionSpec:
    name: CapCondition
    api_kind: ApiKind
    few_shot: bool


CONDITIONS: Dict[str, ConditionSpec] = {
    condition.value: ConditionSpec(
        name=condition,
        api_kind=(
            ApiKind.CONTROLLER
            if condition.value.startswith("controller_")
            else ApiKind.CARTESIAN
        ),
        few_shot=condition.value.endswith("_fewshot"),
    )
    for condition in CapCondition
}


def get_condition(name: str) -> ConditionSpec:
    try:
        return CONDITIONS[name]
    except KeyError as exc:
        supported = ", ".join(sorted(CONDITIONS))
        raise ValueError(
            f"Unknown CaP condition {name!r}. Supported conditions: {supported}"
        ) from exc

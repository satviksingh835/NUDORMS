"""Shared pipeline types.

Lives in its own module to avoid a circular import: orchestrator imports
every stage, and every stage needs StageResult.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StageResult:
    ok: bool
    metrics: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)
    failure_reason: str | None = None

"""Shared base for ASTER's orchestral tools: ``AsterBaseTool`` plus the field
helpers every tool module imports from here.

    from aster_toolkit.tool_base import (
        AsterBaseTool as BaseTool, OptionalRuntimeField, RuntimeField, StateField)

Three orchestral (1.8.0) behaviours this module corrects
--------------------------------------------------------
1. Argument leakage between calls. ``BaseTool.execute(**kwargs)`` builds each
   call from ``self.model_dump()`` updated with ``kwargs``; tool instances
   live for the whole GUI session, so an optional argument passed ONCE
   persisted into every later call that omitted it (a NIRISS download
   inherited planet_name, filters=['G395H'] and visits=[2] from earlier turns
   and found nothing). ``AsterBaseTool.execute`` resets every runtime field
   to its default unless the call supplies it.

2. Optional fields declared required. orchestral marks a runtime field
   REQUIRED unless it has a non-None default or a ``default_factory`` — and
   ``RuntimeField`` auto-injects ``default=None`` — so
   ``RuntimeField(default=None)`` produced a required field and
   ``execute()`` rejected calls that omitted it ("Missing Required Fields").
   ``OptionalRuntimeField`` declares a genuinely optional field.

3. Rollback crash. When ``_run`` raised, orchestral restored fields with
   validated ``setattr``; a required field still holding its unvalidated
   ``None`` default then raised a pydantic ValidationError out of
   ``execute()`` instead of returning the error text to the model.

``AsterBaseTool`` also treats an empty or whitespace-only string for a
non-required field as "not provided" (models often send ``""`` for
arguments they mean to skip), so it falls back to the field's default.

Found by tests/e2e/test_gui_e2e.py (unit tests construct a fresh tool per
call and cannot see 1 or 3); pinned by tests/scripts/test_tool_consolidation.py.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from orchestral.tools.base.field_utils import RuntimeField, StateField
from orchestral.tools.base.tool import BaseTool as _OrchestralBaseTool
from pydantic import Field as _PydanticField
from pydantic_core import PydanticUndefined

__all__ = ["AsterBaseTool", "OptionalRuntimeField", "RuntimeField", "StateField"]


def OptionalRuntimeField(description: str | None = None, **kwargs: Any):
    """An LLM-visible argument that is OPTIONAL in the schema; defaults to None.

    Built directly as a pydantic Field with orchestral's ``runtime`` marker
    and a ``default_factory`` (the only form orchestral treats as optional
    with a None default; pydantic forbids ``default`` alongside it).
    """
    kwargs.pop("default", None)
    return _PydanticField(
        json_schema_extra={"runtime": True},
        default_factory=lambda: None,
        description=description,
        **kwargs,
    )


def _is_blank(value: Any) -> bool:
    return isinstance(value, str) and not value.strip()


class AsterBaseTool(_OrchestralBaseTool):
    # ASTER owns the required-field rule rather than inheriting orchestral's,
    # which changed between releases: 1.4.x treats RuntimeField(default=None)
    # as OPTIONAL (any default counts), 1.8.0 as REQUIRED. The Mac `exo` env
    # ships 1.4.0, so without this the same tool had a different schema — and
    # a None slipping into a str field — depending on the machine. Rule (the
    # 1.8.0 one): a runtime field is required unless it has a non-None default
    # or a default_factory (OptionalRuntimeField). Every LLM client and the
    # MCP adapter read requirements through these three classmethods.
    @classmethod
    def _get_required_fields(cls) -> list[str]:
        fields = cls.model_fields
        return [
            name for name in cls._get_runtime_fields()
            if name in fields
            and fields[name].default_factory is None
            and fields[name].default in (None, PydanticUndefined)
        ]

    @classmethod
    def get_input_schema(cls) -> dict[str, Any]:
        schema = super().get_input_schema()
        schema["required"] = cls._get_required_fields()
        return schema

    @classmethod
    def get_tool_spec(cls):
        spec = super().get_tool_spec()
        spec.input_schema["required"] = cls._get_required_fields()
        return spec

    def execute(self, stream_callback: Optional[Callable[[str], None]] = None,
                **kwargs: Any) -> str:
        fields = type(self).model_fields
        required = set(self._get_required_fields())
        runtime = self._get_runtime_fields()
        # "" for an optional argument means "not provided".
        kwargs = {k: v for k, v in kwargs.items()
                  if not (k in runtime and k not in required and _is_blank(v))}
        for name in runtime:
            if name in kwargs or name in required or name not in fields:
                continue
            default = fields[name].get_default(call_default_factory=True)
            if default is PydanticUndefined:
                continue
            object.__setattr__(self, name, default)
        return super().execute(stream_callback=stream_callback, **kwargs)

    def _rollback_state(self, saved_state: dict[str, Any]) -> None:
        for field_name, value in saved_state.items():
            object.__setattr__(self, field_name, value)

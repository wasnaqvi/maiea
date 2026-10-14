"""Shared base class for ASTER's orchestral tools.

Why this exists
---------------
orchestral's ``BaseTool.execute(**kwargs)`` builds each call from
``self.model_dump()`` updated with ``kwargs``. Tool instances live for the
whole GUI session, so any optional argument a model passes ONCE silently
persists into every later call that omits it: after "list visit 2 of
WASP-39 b G395H", a later "download the NIRISS data for program 1366" ran
with planet_name='WASP-39 b', filters=['G395H'], visits=[2] still set and
found nothing. (Found by tests/e2e/test_gui_e2e.py; unit tests construct a
fresh tool per call and cannot see it.)

``AsterBaseTool.execute`` resets every LLM-visible (runtime) field to its
declared default unless the current call supplies it, so each call means
exactly what its arguments say. State fields (``base_directory`` etc.) are
not runtime fields and are untouched. Required fields are not reset: the call
must supply them, and orchestral reports them missing otherwise.

It also fixes an orchestral rollback crash: when ``_run`` raises, orchestral
restores fields with validated ``setattr``; a required field still holding
its unvalidated ``None`` default then raised a pydantic ValidationError OUT of
``execute()`` instead of returning the "Execution Error" message to the
model. Rollback here restores values without re-validation.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from orchestral.tools.base.tool import BaseTool as _OrchestralBaseTool
from pydantic_core import PydanticUndefined


class AsterBaseTool(_OrchestralBaseTool):
    def execute(self, stream_callback: Optional[Callable[[str], None]] = None,
                **kwargs: Any) -> str:
        fields = type(self).model_fields
        required = set(self._get_required_fields())
        for name in self._get_runtime_fields():
            if name in kwargs or name in required or name not in fields:
                continue
            default = fields[name].get_default(call_default_factory=True)
            if default is PydanticUndefined:
                continue          # required field: execute() reports it missing
            object.__setattr__(self, name, default)
        return super().execute(stream_callback=stream_callback, **kwargs)

    def _rollback_state(self, saved_state: dict[str, Any]) -> None:
        for field_name, value in saved_state.items():
            object.__setattr__(self, field_name, value)

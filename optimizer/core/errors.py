"""
errors.py

The pipeline's single error type. Anything missing, invalid or failing stops the pipeline for that
model with a clear reason; nothing substitutes other behaviour.
"""

from typing import Any, Dict, Optional


class PipelineAbort(Exception):
    """Stops the pipeline for one model. `reason` is human-readable; `step` is the pipeline step
    index (0-6) when known. StepPipeline.run attaches the current step if it is not set."""

    def __init__(self, reason: str, step: Optional[int] = None):
        super().__init__(reason)
        self.reason = reason
        self.step = step


def describe_failure(exc: BaseException) -> Dict[str, Any]:
    """{"error", "errorType", "step"} of a failed run, as reported by the CLI and the benchmark."""
    reason = exc.reason if isinstance(exc, PipelineAbort) else str(exc)
    return {
        # An exception raised without a message (e.g. a bare IndexError) is reported by its type
        "error": reason or type(exc).__name__,
        "errorType": type(exc).__name__,
        "step": getattr(exc, "step", None)
    }

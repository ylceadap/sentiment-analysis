"""Compatibility imports for ordinal probability helpers."""

from .models.ordinal import (
    ORDERED_LABELS,
    ORDINAL_VALUES,
    boundary_threshold_labels,
    compose_ordinal_probabilities,
    probability_argmax_labels,
    project_monotonic_boundaries,
    with_ordinal_diagnostics,
)

__all__ = [
    "ORDERED_LABELS",
    "ORDINAL_VALUES",
    "boundary_threshold_labels",
    "compose_ordinal_probabilities",
    "probability_argmax_labels",
    "project_monotonic_boundaries",
    "with_ordinal_diagnostics",
]

"""Backward-compatible CLI wrapper for Jina ordinal experiments."""

from .experiments.jina_ordinal import _gate_candidate, _select_thresholds, main, run_experiment

__all__ = ["_gate_candidate", "_select_thresholds", "main", "run_experiment"]

if __name__ == "__main__":
    main()

"""Backward-compatible CLI wrapper for embedding experiments."""

from .experiments.embedding import _gate_candidate, main, run_experiment, threshold_predictions

__all__ = ["_gate_candidate", "main", "run_experiment", "threshold_predictions"]

if __name__ == "__main__":
    main()

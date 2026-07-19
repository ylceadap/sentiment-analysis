# Challenger blind evaluation

The existing assignment holdout has already informed several research decisions and is no longer a
blind promotion benchmark. A challenger can be promoted only after the following sequence is frozen:

1. Collect a new labeled Dutch/English review set with no rows from the assignment CSV.
2. Freeze the candidate names, model artifacts, label rubric, and minimum per-label support.
3. Calculate the CSV SHA-256 and place it in `configs/blind_evaluation.yaml` before evaluation.
4. Materialize every candidate as a trusted local `SentimentModel` artifact.
5. Run `sentiment-blind-evaluate --confirm-unseen` once and archive its JSON in MLflow.

The evaluator rejects the original CSV, hash mismatches, normalized overlap, duplicate reviews,
insufficient label support, fewer than two candidates, and missing model artifacts. Jina candidates
remain excluded until their frozen heads are materialized; their non-commercial license is acceptable
for this assignment research but still needs to remain visible in any result.

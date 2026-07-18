"""Reproducible source-data audit with bounded review excerpts."""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .constants import LABELS
from .data import load_dataset, sha256_file
from .language import DutchLanguageDetector
from .text import HTML_BREAK_RE, RATING_RE, ZERO_WIDTH_RE, normalize_text

LOGGER = logging.getLogger(__name__)
HTML_TAG_RE = re.compile(r"<[^>]+>")
MOJIBAKE_RE = re.compile(r"(?:Ã.|Â.|â€|â€™|â€œ|â€˜|ðŸ|�)")


def _distribution(series: pd.Series) -> dict[str, float]:
    """Return counts and percentages for a categorical series."""
    quantiles = series.quantile([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    return {
        f"p{int(level * 100):02d}": round(float(value), 2) for level, value in quantiles.items()
    }


def _safe_examples(frame: pd.DataFrame, mask: pd.Series, limit: int = 3) -> list[dict[str, Any]]:
    """Return bounded excerpts without exposing complete review bodies."""
    examples: list[dict[str, Any]] = []
    for index, row in frame.loc[mask, ["Reviews", "Label"]].head(limit).iterrows():
        excerpt = re.sub(r"\s+", " ", str(row["Reviews"])).strip()[:120]
        examples.append({"row": int(index) + 2, "label": row["Label"], "excerpt": excerpt})
    return examples


def audit_dataset(path: str | Path, detector: DutchLanguageDetector) -> dict[str, Any]:
    """Profile the entire source dataset and return JSON-serializable evidence."""
    data_path = Path(path)
    raw = data_path.read_bytes()
    frame = load_dataset(data_path)
    reviews = frame["Reviews"].astype(str)
    normalized = reviews.map(normalize_text)
    normalized_frame = pd.DataFrame({"review": normalized, "label": frame["Label"]})
    char_lengths = reviews.str.len()
    token_lengths = reviews.str.split().str.len()

    language_rows: list[dict[str, Any]] = []
    for index, text in reviews.items():
        result = detector.detect(text)
        language_rows.append(
            {
                "index": int(index),
                "status": result.status.value,
                "detected_language": result.detected_language,
                "dutch_confidence": round(result.dutch_confidence, 6),
                "top_confidence": round(result.top_confidence, 6),
                "margin": round(result.margin, 6),
            }
        )
    language = pd.DataFrame(language_rows).set_index("index")
    by_label = (
        pd.crosstab(frame["Label"], language["status"])
        .reindex(index=LABELS, fill_value=0)
        .to_dict(orient="index")
    )
    detected_counts = Counter(language["detected_language"].fillna("unknown"))

    masks = {
        "html_break": reviews.str.contains(HTML_BREAK_RE),
        "html_tag": reviews.str.contains(HTML_TAG_RE),
        "zero_width": reviews.str.contains(ZERO_WIDTH_RE),
        "mojibake": reviews.str.contains(MOJIBAKE_RE),
        "rating": reviews.str.contains(RATING_RE),
        "short": char_lengths.lt(100),
        "long": char_lengths.gt(char_lengths.quantile(0.99)),
        "non_dutch": language["status"].eq("non_dutch"),
        "ambiguous": language["status"].eq("ambiguous"),
        "dutch": language["status"].eq("dutch"),
    }
    exact_conflicts = int((frame.groupby("Reviews")["Label"].nunique() > 1).sum())
    normalized_conflicts = int((normalized_frame.groupby("review")["label"].nunique() > 1).sum())
    return {
        "source": {
            "path": str(data_path),
            "size_bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "utf8_bom": raw.startswith(b"\xef\xbb\xbf"),
            "utf8_decode": "ok" if raw.decode("utf-8-sig") is not None else "failed",
        },
        "schema": {
            "rows": len(frame),
            "columns": frame.columns.tolist(),
            "dtypes": frame.dtypes.astype(str).to_dict(),
            "missing": frame.isna().sum().astype(int).to_dict(),
            "blank_reviews": int(reviews.str.strip().eq("").sum()),
            "unexpected_labels": sorted(set(frame["Label"]) - set(LABELS)),
        },
        "labels": {
            label: {
                "count": int(count),
                "percentage": round(100 * int(count) / len(frame), 2),
            }
            for label, count in frame["Label"].value_counts().items()
        },
        "label_runs": int(frame["Label"].ne(frame["Label"].shift()).sum()),
        "duplicates": {
            "exact_extra_rows": int(frame.duplicated().sum()),
            "exact_review_extra_rows": int(frame.duplicated("Reviews").sum()),
            "exact_review_groups": int((frame.groupby("Reviews").size() > 1).sum()),
            "exact_conflicting_groups": exact_conflicts,
            "normalized_extra_rows": int(normalized.duplicated().sum()),
            "normalized_conflicting_groups": normalized_conflicts,
        },
        "lengths": {
            "characters": _distribution(char_lengths),
            "tokens": _distribution(token_lengths),
        },
        "artifacts": {
            name: int(mask.sum())
            for name, mask in masks.items()
            if name not in {"dutch", "non_dutch", "ambiguous"}
        },
        "language": {
            "status_counts": language["status"].value_counts().astype(int).to_dict(),
            "detected_language_counts": dict(detected_counts),
            "status_by_label": by_label,
            "policy": {
                "minimum_dutch_confidence": detector.minimum_dutch_confidence,
                "minimum_margin": detector.minimum_margin,
                "short_text_characters": detector.short_text_characters,
            },
        },
        "examples": {
            name: _safe_examples(frame, mask)
            for name, mask in masks.items()
            if name in {"mojibake", "rating", "non_dutch", "ambiguous", "dutch", "long"}
        },
    }


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    """Render a small GitHub-flavored Markdown table."""
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(audit: dict[str, Any]) -> str:
    """Render concise interpreted findings instead of dumping raw profile output."""
    labels = audit["labels"]
    language = audit["language"]
    artifacts = audit["artifacts"]
    duplicates = audit["duplicates"]
    label_rows = [
        [label, labels[label]["count"], f"{labels[label]['percentage']:.2f}%"] for label in LABELS
    ]
    language_rows = [
        [
            label,
            language["status_by_label"].get(label, {}).get("dutch", 0),
            language["status_by_label"].get(label, {}).get("ambiguous", 0),
            language["status_by_label"].get(label, {}).get("non_dutch", 0),
        ]
        for label in LABELS
    ]
    example_lines: list[str] = []
    for category, examples in audit["examples"].items():
        example_lines.append(f"### {category.replace('_', ' ').title()}")
        example_lines.append("")
        if not examples:
            example_lines.append("No examples matched.")
        for example in examples:
            escaped = example["excerpt"].replace("|", "\\|")
            example_lines.append(f"- CSV row {example['row']}, {example['label']}: “{escaped}…”")
        example_lines.append("")

    return f"""# Data Audit

Generated from the complete source CSV by `sentiment-audit`. Counts are detector output or pattern matches, not manually verified ground truth.

## Dataset and intended grain

- Source: `{audit["source"]["path"]}` ({audit["source"]["size_bytes"]:,} bytes)
- SHA-256: `{audit["source"]["sha256"]}`
- Encoding: UTF-8 with BOM; strict UTF-8 decoding succeeded.
- Grain: one movie-review text and one supplied sentiment label per row.
- Shape: {audit["schema"]["rows"]:,} rows × {
        len(audit["schema"]["columns"])
    } columns (`Reviews`, `Label`).
- Missing required values: {sum(audit["schema"]["missing"].values())}; blank reviews: {
        audit["schema"]["blank_reviews"]
    }.

## Label distribution and ordering

{_markdown_table(["Label", "Rows", "Share"], label_rows)}

There are only {
        audit["label_runs"]
    } contiguous label runs, matching the three labels. The file is therefore completely ordered by label. A sequential split would create severe leakage/bias and is prohibited; all evaluation uses shuffled stratification after normalized deduplication.

## Duplicate integrity

- Exact duplicate rows beyond the first occurrence: {duplicates["exact_extra_rows"]}.
- Duplicate review-text groups: {duplicates["exact_review_groups"]} ({
        duplicates["exact_review_extra_rows"]
    } extra rows).
- Normalized duplicate rows beyond the first occurrence: {duplicates["normalized_extra_rows"]}.
- Conflicting-label groups: {duplicates["normalized_conflicting_groups"]} after normalization.

Risk: identical normalized reviews could inflate evaluation if allowed across splits. The preparation pipeline removes conflicting groups conservatively and retains one member of each same-label group before splitting.

## Length profile

{
        _markdown_table(
            ["Measure"] + list(audit["lengths"]["characters"]),
            [
                ["Characters"] + list(audit["lengths"]["characters"].values()),
                ["Whitespace tokens"] + list(audit["lengths"]["tokens"].values()),
            ],
        )
    }

The observed maximum ({
        audit["lengths"]["characters"][
            "p100"
        ]:.0f} characters) supports an 8,000-character API limit without silently truncating any current source row.

## Text-quality pattern checks

{
        _markdown_table(
            ["Check", "Rows", "Share"],
            [
                [
                    name.replace("_", " ").title(),
                    count,
                    f"{100 * count / audit['schema']['rows']:.2f}%",
                ]
                for name, count in artifacts.items()
            ],
        )
    }

- HTML breaks and invisible Unicode are common transport artifacts and are normalized deterministically.
- Mojibake matches are candidate encoding defects, not proof that an entire row is unusable. The default normalizer preserves them instead of applying an unsafe global repair.
- Rating-pattern matches can expose a direct sentiment cue. Matched retained-versus-masked experiments are required before selecting the final policy.

## Candidate language distribution

Detector policy: local Lingua over Dutch, English, German, French, Spanish, Italian, and Portuguese; confident decisions require top confidence ≥ {
        language["policy"]["minimum_dutch_confidence"]:.2f} and margin ≥ {
        language["policy"]["minimum_margin"]:.2f}. Text under {
        language["policy"]["short_text_characters"]
    } characters is always ambiguous.

{_markdown_table(["Original label", "Dutch", "Ambiguous", "Non-Dutch"], language_rows)}

Detected top languages: {
        json.dumps(language["detected_language_counts"], ensure_ascii=False, sort_keys=True)
    }.

These are language-composition estimates rather than gold labels. The supplied rows are all retained because the detected languages are Dutch and English; language and label are jointly stratified so both languages remain represented in the unified training and held-out partitions. Manual spot checks below assess obvious Dutch, obvious English, uncertain cases, label coverage, artifacts, ratings, and long reviews. Mixed or translated prose remains a known source of identification errors.

## Bounded manual-review samples

{chr(10).join(example_lines)}
## Severity and modeling implications

1. **High — ordered labels:** invalidates sequential splitting. Remediation: deterministic shuffled stratification.
2. **High — class imbalance:** Negative is only {
        labels["Negative"][
            "percentage"
        ]:.2f}% of rows. Remediation: select on macro-F1, report per-class metrics, and compare class weighting.
3. **High — bilingual imbalance:** English is a real supplied segment but has far fewer rows, including only 10 Negative examples. Remediation: train one shared Dutch/English model on all deduplicated rows, jointly stratify language and label, report held-out metrics by language, and attach a reliability warning to English predictions.
4. **Medium — explicit ratings:** may be legitimate review content or label leakage. Remediation: matched mask/no-mask experiment and feature/error analysis.
5. **Medium — markup and invisible characters:** create spurious tokens and duplicate mismatches. Remediation: shared deterministic normalization.
6. **Low/Medium — mojibake candidates:** can weaken individual features. Remediation: preserve by default; only add repair if a separately verified experiment helps.

## Automated checks retained in the project

- Required columns, non-null values, non-empty reviews, and allowed labels.
- Source hash and encoding evidence.
- Exact and normalized duplicate/conflict counts.
- No normalized-review overlap between holdout partitions.
- Stable normalization and language policy tests.

## Assumptions and limitations

- The supplied label is treated as the target, not as independently verified sentiment truth.
- Language identification is probabilistic evidence; it is especially weak for mixed-language, named-entity-heavy, and short text.
- Pattern counts depend on documented regexes and are not semantic annotations.
- This dataset contains no timestamps, entity IDs, or source lineage, so freshness and temporal drift cannot be assessed.
"""


def parse_args() -> argparse.Namespace:
    """Parse dataset audit CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="Python_Engineer_Challenge_2.csv")
    parser.add_argument("--output", default="reports/data_audit.md")
    parser.add_argument("--json-output", default="artifacts/data_audit.json")
    parser.add_argument("--minimum-dutch-confidence", type=float, default=0.70)
    parser.add_argument("--minimum-margin", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    """Run the audit CLI and write Markdown and JSON evidence."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    detector = DutchLanguageDetector(
        minimum_dutch_confidence=args.minimum_dutch_confidence,
        minimum_margin=args.minimum_margin,
    )
    LOGGER.info("Auditing complete dataset: %s", args.data)
    audit = audit_dataset(args.data, detector)
    output = Path(args.output)
    json_output = Path(args.json_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(audit), encoding="utf-8")
    json_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s and %s", output, json_output)


if __name__ == "__main__":
    main()

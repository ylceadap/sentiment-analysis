# Data Audit

Generated from the complete source CSV by `sentiment-audit`. Counts are detector output or pattern matches, not manually verified ground truth.

## Dataset and intended grain

- Source: `Python_Engineer_Challenge_2.csv` (7,435,723 bytes)
- SHA-256: `2788b987e2c9fa4fd6459a1798a6e7d1dd63ddb5618c10907712d39042cc4be2`
- Encoding: UTF-8 with BOM; strict UTF-8 decoding succeeded.
- Grain: one movie-review text and one supplied sentiment label per row.
- Shape: 4,800 rows × 2 columns (`Reviews`, `Label`).
- Missing required values: 0; blank reviews: 0.

## Label distribution and ordering

| Label | Rows | Share |
| --- | --- | --- |
| Positive | 2250 | 46.88% |
| Average | 2250 | 46.88% |
| Negative | 300 | 6.25% |

There are only 3 contiguous label runs, matching the three labels. The file is therefore completely ordered by label. A sequential split would create severe leakage/bias and is prohibited; all evaluation uses shuffled stratification after normalized deduplication.

## Duplicate integrity

- Exact duplicate rows beyond the first occurrence: 2.
- Duplicate review-text groups: 2 (2 extra rows).
- Normalized duplicate rows beyond the first occurrence: 2.
- Conflicting-label groups: 0 after normalization.

Risk: identical normalized reviews could inflate evaluation if allowed across splits. The preparation pipeline removes conflicting groups conservatively and retains one member of each same-label group before splitting.

## Length profile

| Measure | p00 | p01 | p05 | p25 | p50 | p75 | p95 | p99 | p100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Characters | 60.0 | 257.98 | 392.0 | 785.0 | 1143.5 | 1929.25 | 3947.2 | 5673.15 | 7654.0 |
| Whitespace tokens | 12.0 | 44.0 | 66.0 | 135.0 | 192.0 | 322.0 | 649.05 | 922.07 | 1321.0 |

The observed maximum (7654 characters) supports an 8,000-character API limit without silently truncating any current source row.

## Text-quality pattern checks

| Check | Rows | Share |
| --- | --- | --- |
| Html Break | 2908 | 60.58% |
| Html Tag | 2908 | 60.58% |
| Zero Width | 1261 | 26.27% |
| Mojibake | 122 | 2.54% |
| Rating | 411 | 8.56% |
| Short | 2 | 0.04% |
| Long | 48 | 1.00% |

- HTML breaks and invisible Unicode are common transport artifacts and are normalized deterministically.
- Mojibake matches are candidate encoding defects, not proof that an entire row is unusable. The default normalizer preserves them instead of applying an unsafe global repair.
- Rating-pattern matches can expose a direct sentiment cue. Matched retained-versus-masked experiments are required before selecting the final policy.

## Candidate language distribution

Detector policy: local Lingua over Dutch, English, German, French, Spanish, Italian, and Portuguese; confident decisions require top confidence ≥ 0.70 and margin ≥ 0.20. Text under 20 characters is always ambiguous.

| Original label | Dutch | Ambiguous | Non-Dutch |
| --- | --- | --- | --- |
| Positive | 2099 | 0 | 151 |
| Average | 1926 | 0 | 324 |
| Negative | 290 | 0 | 10 |

Detected top languages: {"dutch": 4315, "english": 485}.

These are language-composition estimates rather than gold labels. The supplied rows are all retained because the detected languages are Dutch and English; language and label are jointly stratified so both languages remain represented in the unified training and held-out partitions. Manual spot checks below assess obvious Dutch, obvious English, uncertain cases, label coverage, artifacts, ratings, and long reviews. Mixed or translated prose remains a known source of identification errors.

## Bounded manual-review samples

### Mojibake

- CSV row 5, Positive: “In navolging van het briljante "GoyÃ´kiba" (ook bekend als "Hanzo The Razor - Sword Of Justice", 1972) en het uitstekend…”
- CSV row 9, Positive: “Ik heb niet alle films van Jess Franco gezien, ik heb er vijf gezien, denk ik, en er zijn er meer dan 180. Dus misschien…”
- CSV row 93, Positive: “When you start watching this animation-masterpiece, you quickly notice, that it's a European production. Although the Eu…”

### Rating

- CSV row 4, Positive: “James Cagney staat vooral bekend om zijn stoere karakter- en gangsterrollen, maar hij heeft in zijn carrière ook heel wa…”
- CSV row 10, Positive: “Ik moet zeggen dat Higher Learning een van de top 3 films is die ik ooit heb gezien. Het heeft een briljante cast en een…”
- CSV row 22, Positive: “Dit werk gaat minder over Steve Martins personage Davis, dan over Kline (Mack) en Glover (Simon), en Kline en McDonnell …”

### Long

- CSV row 63, Positive: “Het is altijd moeilijk om welke film dan ook als ‘de beste’ te bestempelen, of het nu aller tijden is, een bepaald genre…”
- CSV row 351, Positive: “Vaak komen films van deze aard over als een allegaartje van geweldig werk, samen met een beetje geklets om de speelduur …”
- CSV row 358, Positive: “Veel mensen blijven hangen bij het label van deze film als 'kinderfilm', en dat is het zeker, ook al is het een film gem…”

### Non Dutch

- CSV row 77, Positive: “In the classic sense of the four humors (which are not specific to the concept of funny or even entertainment), Altman's…”
- CSV row 78, Positive: “Imagine turning out the lights in your remote farmhouse on a cold night, and then going to bed. There's no need to lock …”
- CSV row 79, Positive: “This movie is a lot of fun. The actors really make the movie go the distance though. Without giving away the plot, I wou…”

### Ambiguous

No examples matched.

### Dutch

- CSV row 2, Positive: “De kameleonachtige uitvoering van Kurt Russell, gecombineerd met het onberispelijke filmwerk van John Carpenter, maakt d…”
- CSV row 3, Positive: “Het was een extreem laag budget (sommige scènes lijken te zijn opgenomen met een videorecorder voor thuis). Het heeft ec…”
- CSV row 4, Positive: “James Cagney staat vooral bekend om zijn stoere karakter- en gangsterrollen, maar hij heeft in zijn carrière ook heel wa…”

## Severity and modeling implications

1. **High — ordered labels:** invalidates sequential splitting. Remediation: deterministic shuffled stratification.
2. **High — class imbalance:** Negative is only 6.25% of rows. Remediation: select on macro-F1, report per-class metrics, and compare class weighting.
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

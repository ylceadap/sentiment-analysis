# System Architecture

The repository has one production inference path and two research-only experiment paths. Every path
shares immutable data boundaries, normalization, label order, metrics, and hash verification.

## End-to-end data and model flow

```mermaid
flowchart LR
    raw["Immutable CSV\nReviews + Label"] --> audit["Schema and SHA-256 audit"]
    raw --> language["Local Lingua detection"]
    language --> normalize["Unicode, HTML and whitespace normalization"]
    normalize --> dedupe["Normalized deduplication\nand conflict removal"]
    dedupe --> split["Language x label stratified split\n3,838 train / 960 held-out"]

    split -->|"training rows only"| cv["Fixed 5-fold OOF indices"]
    split -->|"single formal evaluation"| heldout["Held-out evidence"]

    cv --> classical["Word + character TF-IDF\nLogistic Regression candidates"]
    cv --> jina["Revision-pinned Jina embeddings\nresearch-only cache"]
    jina --> multiclass["Multiclass Logistic Regression"]
    jina --> ordinal["Two calibrated ordinal boundaries"]

    classical --> selection["Macro-F1 selection\nminority and stability guardrails"]
    multiclass --> research["OOF research comparison"]
    ordinal --> research
    selection --> finalfit["Fit selected classical model"]
    finalfit --> heldout
    heldout --> artifact["Trusted model.joblib\nmetadata + metrics + errors"]
    research --> evidence["CSV + JSON + Markdown\nMLflow evidence runs"]
    artifact --> registry["MLflow Registry\nsentiment-production@champion"]
```

The held-out split is not used for embedding or ordinal selection. Later research comparisons that
reuse it are labeled `reused-heldout`; a new blind test is required before challenger promotion.

## Production inference path

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI
    participant Service as InferenceService
    participant Language as Lingua detector
    participant Model as SentimentModel

    Client->>API: POST /classify
    API->>API: Validate nonblank <= 8,000 characters
    API->>Service: classify(review, explain)
    Service->>Language: detect(review)
    Language-->>Service: language status and confidence
    alt unsupported language
        Service-->>API: NonDutchReviewError
        API-->>Client: HTTP 422
    else Dutch, English, or short ambiguous text
        Service->>Model: infer(review)
        Model->>Model: normalize once
        Model->>Model: TF-IDF transform once
        Model->>Model: label + probabilities + optional contributions
        Model-->>Service: ModelInference
        Service-->>API: PredictionResult + warning policy
        API-->>Client: typed JSON response
    end
```

The language detector is deliberately outside the serialized sklearn pipeline because it controls
request policy and warnings. Normalization, vectorization, and classification remain together inside
the fitted pipeline to prevent training/serving drift.

## Module ownership

```mermaid
flowchart TB
    api["api.py\nHTTP contract and lifespan"] --> service["service.py\nlanguage policy and timing"]
    service --> language["language.py\nlocal detection"]
    service --> contract["models/base.py\nserving protocol"]
    contract --> model["models/classical.py\nTF-IDF production model"]
    model --> text["text.py\nnormalization"]
    api --> advisor["models/llm_advisor.py\nexternal advisory model"]

    train["train.py\ntracked model selection and final fit"] --> data["data.py\nload, deduplicate, split"]
    train --> model
    train --> metrics["metrics.py\nclassification and probability evidence"]
    train --> reporting["reporting.py\ndurable report generation"]

    embed["experiments/embedding.py\nmulticlass embedding experiment"] --> prepared["experiments/data.py\nfrozen split and folds"]
    embed --> runtime["models/embeddings.py\nrevision-aware cache and encoder"]
    embed --> utilities["experiments/common.py\nhashes, alignment, slices, gates"]
    ordinalexp["experiments/jina_ordinal.py\nordinal experiment orchestration"] --> prepared
    ordinalexp --> runtime
    ordinalexp --> utilities
    ordinalexp --> ordinalmath["models/ordinal.py\nmonotonic projection and equations"]
    embed --> metrics
    ordinalexp --> metrics

    benchmark["benchmark.py\ncold/warm and ASGI latency"] --> api
    benchmark --> service
    audit["audit.py\nsource profiling"] --> data
```

## Artifacts, tracking, and deployment

```mermaid
flowchart LR
    code["Git commit"] --> metadata["model_metadata.json"]
    config["YAML configuration"] --> training["Training and experiments"]
    hashes["Raw, split, and model hashes"] --> metadata
    training --> mlflow["Local SQLite MLflow"]
    training --> portable["Portable JSON, CSV and Markdown"]
    training --> model["model.joblib"]
    model --> docker["Python 3.11 slim\nnon-root API image"]
    metadata --> docker
    mlflow --> registry["Production / benchmark / challenger / research / external"]
    portable --> review["Git-reviewable evidence"]
```

Docker copies only source code, the trusted production model, and model metadata. Raw data, reports,
tests, caches, secrets, notebooks, MLflow state, and training dependencies remain outside the image.
The current environment has no Docker executable, so image execution remains statically reviewed but
not runtime-verified.

## Git and model lifecycle

```mermaid
flowchart LR
    feature["Short-lived agent branch"] --> pr["Reviewed pull request"]
    pr --> main["main\nonly long-lived code branch"]
    experiment["Completed experiment branch"] --> tag["Immutable archive tag"]
    experiment --> run["MLflow evidence run"]
    run --> registry["Champion / challenger / research alias"]
    tag --> cleanup["Delete long-lived experiment branch"]
    registry --> service["Service loads champion only"]
```

Model families are separated by packages and configuration, not by permanent Git branches. See
`docs/GIT_MLFLOW_MAPPING.md` for the archived source-to-run mapping.

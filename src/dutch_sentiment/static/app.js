const form = document.querySelector("#classify-form");
const reviewInput = document.querySelector("#review");
const explainInput = document.querySelector("#explain");
const submitButton = document.querySelector("#submit-button");
const characterCount = document.querySelector("#character-count");
const healthStatus = document.querySelector("#health-status");
const statusDot = document.querySelector("#status-dot");
const agreementStatus = document.querySelector("#agreement-status");
const agreementCopy = document.querySelector("#agreement-copy");
const modelLatency = document.querySelector("#model-latency");
const modelEmpty = document.querySelector("#model-empty");
const modelContent = document.querySelector("#model-content");
const modelLabel = document.querySelector("#model-label");
const languagePill = document.querySelector("#language-pill");
const modelWarning = document.querySelector("#model-warning");
const probabilities = document.querySelector("#probabilities");
const explanationDetails = document.querySelector("#explanation-details");
const explanationOutput = document.querySelector("#explanation-output");
const llmLatency = document.querySelector("#llm-latency");
const llmEmpty = document.querySelector("#llm-empty");
const llmContent = document.querySelector("#llm-content");
const llmLabel = document.querySelector("#llm-label");
const llmStatus = document.querySelector("#llm-status");
const llmModelName = document.querySelector("#llm-model-name");
const llmWarning = document.querySelector("#llm-warning");
const errorBox = document.querySelector("#error-box");
const comparisonBody = document.querySelector("#comparison-body");
const comparisonNote = document.querySelector("#comparison-note");

const labels = ["Positive", "Average", "Negative"];
const comparisonRoles = {
  "Current Production TF-IDF": "Production",
  "TF-IDF Ordinal": "Challenger",
  "Jina Logistic": "Research",
  "Jina Ordinal": "Research",
  "DeepSeek V4 Flash 24-shot": "External evidence",
  "RobBERT v2 Improved Ensemble": "Challenger",
  "RobBERT v2 Logistic": "Test evidence",
};

function setCharacterCount() {
  characterCount.textContent = `${reviewInput.value.length} / ${reviewInput.maxLength}`;
}

function iconMarkup() {
  return '<svg aria-hidden="true" viewBox="0 0 24 24"><path d="M5 12h14M13 6l6 6-6 6" /></svg>';
}

function setLoading(isLoading) {
  submitButton.disabled = isLoading;
  submitButton.innerHTML = isLoading ? "Comparing..." : `${iconMarkup()} Compare`;
}

function formatPercent(value) {
  return `${Math.round(value * 1000) / 10}%`;
}

function formatLatency(value) {
  return typeof value === "number" ? `${value.toFixed(2)} ms` : "-- ms";
}

function renderProbabilities(values) {
  probabilities.replaceChildren();
  labels.forEach((label) => {
    const value = values[label] ?? 0;
    const row = document.createElement("div");
    row.className = "probability-row";
    row.innerHTML = `
      <div class="probability-meta">
        <span>${label}</span>
        <span>${formatPercent(value)}</span>
      </div>
      <div class="bar" aria-label="${label} probability ${formatPercent(value)}">
        <div class="bar-fill ${label}" style="width: ${Math.max(0, Math.min(100, value * 100))}%"></div>
      </div>
    `;
    probabilities.append(row);
  });
}

function applyLabel(element, label) {
  element.className = `prediction-label ${label || ""}`;
  element.textContent = label || "--";
}

function showModelResult(data) {
  modelEmpty.classList.add("hidden");
  modelContent.classList.remove("hidden");
  applyLabel(modelLabel, data.label);
  languagePill.textContent = data.detected_language;
  modelLatency.textContent = formatLatency(data.latency_ms);
  renderProbabilities(data.probabilities);

  if (data.warnings.length > 0) {
    modelWarning.textContent = data.warnings.join(" ");
    modelWarning.classList.remove("hidden");
  } else {
    modelWarning.classList.add("hidden");
  }

  if (data.explanation) {
    explanationOutput.textContent = JSON.stringify(data.explanation, null, 2);
    explanationDetails.classList.remove("hidden");
  } else {
    explanationDetails.classList.add("hidden");
  }
}

function showLLMResult(data) {
  llmEmpty.classList.add("hidden");
  llmContent.classList.remove("hidden");
  llmModelName.textContent = `${data.provider} / ${data.model} / ${data.prompt_profile}`;
  llmLatency.textContent = formatLatency(data.latency_ms);
  llmStatus.textContent = data.status;
  llmStatus.className = `status-pill ${data.status}`;

  applyLabel(llmLabel, data.status === "ok" ? data.label : null);

  if (data.warning) {
    llmWarning.textContent = data.warning;
    llmWarning.classList.remove("hidden");
  } else {
    llmWarning.classList.add("hidden");
  }
}

function showAgreement(value) {
  if (value === true) {
    agreementStatus.textContent = "Agree";
    agreementCopy.textContent = "The submitted model and LLM advisor returned the same label.";
  } else if (value === false) {
    agreementStatus.textContent = "Disagree";
    agreementCopy.textContent = "Treat this as a review case. The submitted local model remains the formal output.";
  } else {
    agreementStatus.textContent = "Model only";
    agreementCopy.textContent = "The LLM advisor is not available for this request.";
  }
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) {
      throw new Error("Health check failed");
    }
    const data = await response.json();
    healthStatus.textContent = `Ready (${data.model_version})`;
    statusDot.classList.add("ready");
  } catch {
    healthStatus.textContent = "Unavailable";
    statusDot.classList.add("failed");
  }
}

function metricCell(value) {
  const cell = document.createElement("td");
  cell.className = "metric-value";
  cell.textContent = Number(value).toFixed(4);
  return cell;
}

function renderComparison(rows) {
  comparisonBody.replaceChildren();
  rows.forEach((item) => {
    const row = document.createElement("tr");
    if (item.model === "Current Production TF-IDF") {
      row.classList.add("production-row");
    }

    const rank = document.createElement("td");
    rank.className = "rank-cell";
    rank.textContent = `#${item.rank}`;

    const model = document.createElement("th");
    model.scope = "row";
    model.textContent = item.model;

    const role = document.createElement("td");
    const roleBadge = document.createElement("span");
    const roleName = comparisonRoles[item.model] ?? "Evidence";
    roleBadge.className = `role-badge role-${roleName
      .toLowerCase()
      .replaceAll(" ", "-")}`;
    roleBadge.textContent = roleName;
    role.append(roleBadge);

    const negative = document.createElement("td");
    negative.className = "metric-value";
    negative.textContent = `${Number(item.negative_precision).toFixed(4)} / ${Number(
      item.negative_recall,
    ).toFixed(4)}`;

    row.append(
      rank,
      model,
      role,
      metricCell(item.macro_f1),
      metricCell(item.accuracy),
      negative,
    );
    comparisonBody.append(row);
  });
}

async function loadModelComparison() {
  try {
    const response = await fetch("/model-comparison");
    if (!response.ok) {
      throw new Error("Comparison evidence unavailable");
    }
    const data = await response.json();
    renderComparison(data.ranking);
  } catch {
    comparisonBody.innerHTML =
      '<tr><td colspan="6" class="table-message">Tracked comparison evidence is unavailable in this build.</td></tr>';
    comparisonNote.textContent =
      "The live inference contract is unchanged: only Production TF-IDF is the formal output.";
  }
}

reviewInput.addEventListener("input", setCharacterCount);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.classList.add("hidden");
  const review = reviewInput.value.trim();
  if (!review) {
    showError("Review text must not be empty.");
    return;
  }

  setLoading(true);
  try {
    const response = await fetch("/recommendations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ review, explain: explainInput.checked }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Classification failed.");
    }
    showModelResult(data.model_prediction);
    showLLMResult(data.llm_recommendation);
    showAgreement(data.agreement);
  } catch (error) {
    showError(error.message || "Classification failed.");
  } finally {
    setLoading(false);
  }
});

setCharacterCount();
checkHealth();
loadModelComparison();

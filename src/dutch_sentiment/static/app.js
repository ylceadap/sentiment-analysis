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
const llmRationale = document.querySelector("#llm-rationale");
const llmConfidenceRow = document.querySelector("#llm-confidence-row");
const llmConfidence = document.querySelector("#llm-confidence");
const llmWarning = document.querySelector("#llm-warning");
const errorBox = document.querySelector("#error-box");

const labels = ["Positive", "Average", "Negative"];

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
  llmModelName.textContent = `${data.provider} / ${data.model}`;
  llmLatency.textContent = formatLatency(data.latency_ms);
  llmStatus.textContent = data.status;
  llmStatus.className = `status-pill ${data.status}`;

  if (data.status === "ok") {
    applyLabel(llmLabel, data.label);
    llmRationale.textContent = data.rationale || "No rationale returned.";
    if (typeof data.confidence === "number") {
      llmConfidence.textContent = formatPercent(data.confidence);
      llmConfidenceRow.classList.remove("hidden");
    } else {
      llmConfidenceRow.classList.add("hidden");
    }
  } else {
    applyLabel(llmLabel, null);
    llmRationale.textContent =
      data.status === "unavailable"
        ? "The server is running without LLM credentials."
        : "The LLM request failed. The submitted local model result is still valid.";
    llmConfidenceRow.classList.add("hidden");
  }

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

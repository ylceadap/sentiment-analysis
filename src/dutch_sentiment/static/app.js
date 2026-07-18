const form = document.querySelector("#classify-form");
const reviewInput = document.querySelector("#review");
const explainInput = document.querySelector("#explain");
const submitButton = document.querySelector("#submit-button");
const characterCount = document.querySelector("#character-count");
const healthStatus = document.querySelector("#health-status");
const statusDot = document.querySelector("#status-dot");
const latency = document.querySelector("#latency");
const emptyState = document.querySelector("#empty-state");
const resultContent = document.querySelector("#result-content");
const predictionLabel = document.querySelector("#prediction-label");
const languagePill = document.querySelector("#language-pill");
const warningBox = document.querySelector("#warning-box");
const errorBox = document.querySelector("#error-box");
const probabilities = document.querySelector("#probabilities");
const explanationDetails = document.querySelector("#explanation-details");
const explanationOutput = document.querySelector("#explanation-output");

const labels = ["Positive", "Average", "Negative"];

function setCharacterCount() {
  characterCount.textContent = `${reviewInput.value.length} / ${reviewInput.maxLength}`;
}

function setLoading(isLoading) {
  submitButton.disabled = isLoading;
  submitButton.textContent = isLoading ? "Classifying..." : "Classify";
  if (!isLoading) {
    submitButton.insertAdjacentHTML(
      "afterbegin",
      '<svg aria-hidden="true" viewBox="0 0 24 24"><path d="M5 12h14M13 6l6 6-6 6" /></svg>',
    );
  }
}

function formatPercent(value) {
  return `${Math.round(value * 1000) / 10}%`;
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

function showResult(data) {
  errorBox.classList.add("hidden");
  emptyState.classList.add("hidden");
  resultContent.classList.remove("hidden");

  predictionLabel.className = `prediction-label ${data.label}`;
  predictionLabel.textContent = data.label;
  languagePill.textContent = data.detected_language;
  latency.textContent = `${data.latency_ms.toFixed(2)} ms`;
  renderProbabilities(data.probabilities);

  if (data.warnings.length > 0) {
    warningBox.textContent = data.warnings.join(" ");
    warningBox.classList.remove("hidden");
  } else {
    warningBox.classList.add("hidden");
  }

  if (data.explanation) {
    explanationOutput.textContent = JSON.stringify(data.explanation, null, 2);
    explanationDetails.classList.remove("hidden");
  } else {
    explanationDetails.classList.add("hidden");
  }
}

function showError(message) {
  resultContent.classList.add("hidden");
  emptyState.classList.add("hidden");
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
  const review = reviewInput.value.trim();
  if (!review) {
    showError("Review text must not be empty.");
    return;
  }

  setLoading(true);
  try {
    const response = await fetch("/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ review, explain: explainInput.checked }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Classification failed.");
    }
    showResult(data);
  } catch (error) {
    showError(error.message || "Classification failed.");
  } finally {
    setLoading(false);
  }
});

setCharacterCount();
checkHealth();

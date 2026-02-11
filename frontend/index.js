// Minimal JS strictly for SPY prediction interface

const el = id => document.getElementById(id);

// DOM references
const tabSingle = el('tab-single');
const tabBatch = el('tab-batch');
const panelSingle = el('panel-single');
const panelBatch = el('panel-batch');

const predictBtn = el('predictBtn');
const runBatch = el('runBatch');

const dateInput = el('date');
const headlineInput = el('news_heading');

const csvFile = el('csvFile');

const resultsBody = el('resultsBody');
const batchProgress = el('batchProgress');

const predictionCard = el('predictionCard');
const finalLabel = el('finalLabel');
const pastPredictions = el('pastPredictions');



// Tab switching
// tabSingle.onclick = () => {
//     tabSingle.classList.add('tab-active');
//     tabBatch.classList.remove('tab-active');
//     panelSingle.classList.remove('hidden');
//     panelBatch.classList.add('hidden');
// };

// tabBatch.onclick = () => {
//     tabBatch.classList.add('tab-active');
//     tabSingle.classList.remove('tab-active');
//     panelBatch.classList.remove('hidden');
//     panelSingle.classList.add('hidden');
// };
if (tabSingle && tabBatch && panelSingle && panelBatch) {
    tabSingle.onclick = () => {
      tabSingle.classList.add('tab-active');
      tabBatch.classList.remove('tab-active');
      panelSingle.classList.remove('hidden');
      panelBatch.classList.add('hidden');
      if (predictionCard) predictionCard.classList.add('hidden');
    };
    tabBatch.onclick = () => {
      tabBatch.classList.add('tab-active');
      tabSingle.classList.remove('tab-active');
      panelBatch.classList.remove('hidden');
      panelSingle.classList.add('hidden');
      if (predictionCard) predictionCard.classList.add('hidden');
    };
  } 

window.addEventListener('DOMContentLoaded', () => {
    const backendEl = el('backendUrlCode');
    if (backendEl) backendEl.innerText = window.location.origin;
    if (thresholdSlider) thresholdSlider.dispatchEvent(new Event('input'));
    if (tabSingle) tabSingle.click();
  }); 
  
// API call for single headline
async function callSinglePredict(date, headline) {
    const res = await fetch('/predict_single', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ date, headline })
    });

    return await res.json();
}

// Show result and last 5 predictions
function showResult(data) {
    predictionCard.classList.remove('hidden');

    finalLabel.innerText =
        "Predicted Next Day SPY Price: $" + data.predicted_price;

    pastPredictions.innerHTML = "";

    data.past_predictions.forEach(p => {
        const li = document.createElement("li");
        li.innerText = p;
        pastPredictions.appendChild(li);
    });
}

// Single predict button
predictBtn.onclick = async () => {
    const date = dateInput.value;
    const headline = headlineInput.value.trim();

    if (!date || !headline) {
        alert("Both date and headline are required");
        return;
    }

    predictBtn.disabled = true;

    try {
        const result = await callSinglePredict(date, headline);

        if (result.status === "error") {
            alert(result.message);
        } else {
            showResult(result);
        }

    } catch (e) {
        alert("Prediction error: " + e);
    }

    predictBtn.disabled = false;
};

// CSV batch upload
runBatch.onclick = async () => {
    const file = csvFile.files[0];

    if (!file) {
        alert("Please select a CSV file");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    batchProgress.innerText = "Uploading...";

    const res = await fetch("/predict_csv", {
        method: "POST",
        body: formData
    });

    const data = await res.json();

    if (data.status === "error") {
        alert(data.message);
        batchProgress.innerText = "Error";
        return;
    }

    resultsBody.innerHTML = "";

    data.results.forEach((r, i) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>${i + 1}</td>
            <td>${r.date}</td>
            <td>${r.headline}</td>
            <td>${r.predicted_price}</td>
        `;
        resultsBody.appendChild(tr);
    });

    batchProgress.innerText = "Completed";
};
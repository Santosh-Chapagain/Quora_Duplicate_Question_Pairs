const modelSelect = document.getElementById('model');
const question1Input = document.getElementById('question1');
const question2Input = document.getElementById('question2');
const predictBtn = document.getElementById('predictBtn');
const resultBox = document.getElementById('resultBox');
const resultLabel = document.getElementById('resultLabel');
const resultMeta = document.getElementById('resultMeta');
const resultRaw = document.getElementById('resultRaw');
const errorBox = document.getElementById('errorBox');

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove('hidden');
  resultBox.classList.add('hidden');
}

function showResult(data) {
  const label = data.label || 'Unknown';
  const modelName = data.model || modelSelect.options[modelSelect.selectedIndex].text;
  const rawOutput = data.raw_output ? `Raw output: ${data.raw_output}` : '';

  resultLabel.textContent = label;
  resultMeta.textContent = modelName;
  resultRaw.textContent = rawOutput;
  resultBox.classList.remove('hidden');
  errorBox.classList.add('hidden');
}

predictBtn.addEventListener('click', async () => {
  const question1 = question1Input.value.trim();
  const question2 = question2Input.value.trim();
  const model = modelSelect.value;

  if (!question1 || !question2) {
    showError('Please enter both questions before predicting.');
    return;
  }

  predictBtn.disabled = true;
  predictBtn.textContent = 'Predicting...';
  errorBox.classList.add('hidden');

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ question1, question2, model })
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'Prediction failed.');
    }

    showResult(data);
  } catch (error) {
    showError(error.message);
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = 'Predict duplicate status';
  }
});

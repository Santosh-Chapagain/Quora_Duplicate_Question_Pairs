# 🔍 Quora Duplicate Question Pairs Detection App

A premium, full-stack NLP project comparing **four generations of Natural Language Processing architectures** side-by-side on the Quora Question Pairs dataset. The application features a clean, responsive, and modern Web UI with dynamic, on-demand model loading.

---

## 🏗️ Project Architecture

To keep the repository production-ready, the project is structured cleanly, isolating web assets, utility scripts, and training assets:

```text
├── data/                                # Dataset storage (train2.csv)
├── scripts/                             # Utility scripts for retraining
│   └── retrain_bow.py                   # High-speed parallel RF retraining script
├── webapp/                              # Modern Flask Webapp
│   ├── static/                          # Styling and Frontend Logic
│   │   ├── css/style.css                # Premium vanilla CSS stylesheet
│   │   └── js/main.js                   # Client-side form handlers (no confidence score shown)
│   ├── templates/                       # HTML views
│   │   └── index.html                   # Beautiful frontend template
│   ├── app.py                           # Flask Server Entrypoint
│   └── model_utils.py                   # Custom Model Loader & Feature Engineering Layer
├── 02_Using_Only_BOW.ipynb              # Notebook: Bag of Words training
├── 03_BOW_With_Feature_Engineering.ipynb # Notebook: Advanced Feature engineering
├── 04_BOW_With_Advance_Feature_Eng...  # Notebook: RandomForest + Advanced BOW
├── 05_Using_LSTM.ipynb                  # Notebook: BiLSTM training & evaluation
├── 06_Using_Tranformer_Minilm.ipynb     # Notebook: SentenceTransformer Similarity
├── 07_FineTuned_TinyLlama.ipynb         # Notebook: LoRA adapter LLM Fine-Tuning
```

---

## 🤖 The 4 Generations of Models Compared

| Model Dropdown Name | Tech Stack | How it Works | Key Strengths & Weaknesses |
| :--- | :--- | :--- | :--- |
| **Keyword Matcher (Traditional & Fast)** | `CountVectorizer` + `RandomForest` | Merges **3,000 BOW features** per question with **22 advanced engineered features** (Fuzzy ratios, token set overlaps, character/word lengths, LCS). | 🚀 Ultra-fast inference.<br>❌ Blind to synonyms (e.g. fails on *quickly* vs *fastest way*). |
| **Sequence Analyzer (Deep Neural Network)** | `BiLSTM` (Keras/TensorFlow) | Passes word sequence embeddings through a **Bidirectional Long Short-Term Memory** neural network to capture sequential structure. | 🧠 Understands context and order of words.<br>❌ Sensitive to minor grammatical edits. |
| **Semantic Similarity (Sentence Transformer)** | `all-MiniLM-L6-v2` + `Cosine Classifier` | Encodes both sentences into a high-dimensional vector space, calculating **semantic distance** between embeddings. | 🎯 Excellent at catching synonyms.<br>❌ Can be overly conservative on noisy dataset labels. |
| **Generative AI Reasoning (TinyLlama LLM)** | `TinyLlama 1.1B` + `LoRA PEFT` | Uses a Large Language Model fine-tuned on Quora via **PEFT (Parameter-Efficient Fine-Tuning)** to read and reason through duplicate questions. | 🏆 **Gold Standard:** Possesses deep world knowledge and reasoning; robust against trick questions. |

---

## ⚡ Performance Optimizations & Highlights

### 🏎️ Parallelized Feature Extraction
The traditional BOW pipeline extracts 22 advanced engineered features (like fuzzy ratios and longest common substrings). By utilizing python's **`joblib.Parallel`** and `multiprocessing` inside [retrain_bow.py](file:///d:/Desktop/Quora_Duplicate_Project/scripts/retrain_bow.py), **50,000 question pairs are fully processed in just 26 seconds**, representing a $10\times$ speedup!

### 🎯 100% Tokenizer & Custom Layer Alignment
*   **Custom Keras Layer:** Replaced standard unportable `Lambda` layers inside the BiLSTM with a customized, clean `AbsDiff(Layer)` subclass to completely resolve legacy Keras `bad marshal data` errors.
*   **Tokenizer Synchronization:** Programmed the training notebook to export the exact fitted Tokenizer as `bilstm_tokenizer.pkl`. The web app loads this directly, eliminating scrambled index outputs.

### 🔌 CPU-Only Compatibility Fixes
Addressed PEFT parameters offloading errors by dynamically disabling parameter device maps when running on non-GPU/CPU environments (`device_map = "auto" if torch.cuda.is_available() else None`).

### 🎛️ Recall Optimization via Decision Threshold Tuning
Quora Question Pairs are naturally noisy and standard models tend to be highly conservative. To maximize **recall** (correctly catching actual duplicates like *"How to lose weight"* vs *"Weight loss methods"*), the decision thresholds for BOW, BiLSTM, and Sentence Transformers were adjusted to **`0.4` (40%)**. This makes standard models dramatically more effective!

---

## 🚀 How to Run the Project

### 1. Setup the Virtual Environment
Ensure your virtual environment is activated and install all necessary requirements:
```bash
# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Start the Web Application
Launch the Flask web application by executing:
```bash
python webapp/app.py
```
Open `http://127.0.0.1:5000` in your web browser to play with the modern interface.

### 3. (Optional) Retrain the BOW Random Forest Model
If you ever modify the training dataset (`train2.csv`), you can instantly rebuild your BOW models and vocabulary using our parallel retraining script:
```bash
python scripts/retrain_bow.py
```


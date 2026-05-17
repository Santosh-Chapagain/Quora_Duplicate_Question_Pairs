from flask import Flask, jsonify, render_template, request

from model_utils import MODEL_CHOICES, predict_duplicate
import traceback

app = Flask(__name__)


@app.get("/")
def home():
    return render_template("index.html", model_choices=MODEL_CHOICES)


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or request.form
    question1 = (payload.get("question1") or "").strip()
    question2 = (payload.get("question2") or "").strip()
    model_key = (payload.get("model") or "").strip()

    if not question1 or not question2:
        return jsonify({"error": "Please enter both questions."}), 400

    if model_key not in MODEL_CHOICES:
        return jsonify({"error": "Please choose a valid model."}), 400

    try:
        result = predict_duplicate(model_key, question1, question2)
        return jsonify(result)
    except Exception as exc:
        tb = traceback.format_exc()
        # Log to server console and return traceback in response for debugging
        print(tb)
        return jsonify({"error": str(exc), "traceback": tb}), 500


if __name__ == "__main__":
    app.run(debug=True)

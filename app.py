import os
import pickle
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# File paths for model and vectorizer pkl files
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

def load_pickle_file(file_path):
    """Safely loads a pickle file if it exists."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Pickle file not found at path: {file_path}")
    with open(file_path, "rb") as f:
        return pickle.load(f)

# Load the vectorizer and model on startup
try:
    vectorizer = load_pickle_file(VECTORIZER_PATH)
    model = load_pickle_file(MODEL_PATH)
    print("Model and Vectorizer loaded successfully.")
except Exception as e:
    print(f"Error loading pickle files: {e}")
    vectorizer = None
    model = None

@app.route("/", methods=["GET"])
def home():
    """Health check route and basic info."""
    return jsonify({
        "status": "online",
        "message": "Flask ML Model API is running on Render."
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    """Predict route expecting JSON input: {"text": "your input text here"}"""
    if model is None or vectorizer is None:
        return jsonify({"error": "Model or vectorizer not loaded."}), 500

    data = request.get_json(force=True)
    if not data or "text" not in data:
        return jsonify({"error": "Please provide a 'text' field in the JSON payload."}), 400

    input_text = data["text"]

    try:
        # Transform input text using loaded vectorizer
        transformed_input = vectorizer.transform([input_text])
        
        # Make prediction
        prediction = model.predict(transformed_input)
        
        # Convert prediction to python native type for JSON response
        result = prediction[0]
        if hasattr(result, "item"):
            result = result.item()

        return jsonify({
            "input_text": input_text,
            "prediction": result
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

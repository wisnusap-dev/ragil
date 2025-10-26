from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load model yang sudah kamu simpan
MODEL = joblib.load("model.pkl")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    # Terima data JSON dari request
    data = request.get_json(force=True)
    X = pd.DataFrame([data])  # ubah dict jadi DataFrame
    yhat = MODEL.predict(X)[0]
    proba = None
    if hasattr(MODEL, "predict_proba"):
        proba = float(MODEL.predict_proba(X)[:, 1][0])
    return jsonify({
        "prediction": int(yhat),
        "proba": proba
    })

if __name__ == "__main__":
    app.run(port=5000)

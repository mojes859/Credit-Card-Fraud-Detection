from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.joblib")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        values = data.get('values', [])

        if not values or len(values) != 30:
            return jsonify({"error": "Expected 30 numeric inputs"}), 400

        X_input = np.array(values).reshape(1, -1)
        pred = model.predict(X_input)[0]  # 0 or 1
        prob = model.predict_proba(X_input)[0][1]  # probability of fraud

        if pred == 0:
            status = 'green'
            text = 'Original Transaction'
        else:
            status = 'red'
            text = 'Fraud Transaction'

        return jsonify({
            "pred": int(pred),
            "text": text,
            "status": status,
            # "prob": round(prob * 100, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

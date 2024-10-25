from flask import Flask, request, jsonify
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the DictVectorizer and the model
with open('dv.bin', 'rb') as f_in:
    dv = pickle.load(f_in)

with open('model1.bin', 'rb') as f_in:
    model = pickle.load(f_in)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    input_data = request.get_json()

    # Transform the input using the loaded DictVectorizer
    X = dv.transform([input_data])

    # Predict the probability
    y_pred = model.predict_proba(X)[0, 1]

    # Create a response dictionary and return it as JSON
    result = {
        'probability_of_subscription': round(y_pred, 3)
    }
    return jsonify(result)

if __name__ == '__main__':
    # No need to use app.run() when using Gunicorn
    app.run(debug=True, host='0.0.0.0')

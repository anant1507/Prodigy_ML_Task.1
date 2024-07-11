from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained model here
# Example: Load a trained linear regression model
# Replace this with your actual model loading code
# Assuming model is already trained and saved as 'model.pkl'
model = joblib.load('model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get inputs from the form
        sqft = float(request.form['sqft'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        
        # Prepare input as a numpy array
        input_data = np.array([[sqft, bedrooms, bathrooms]])
        
        # Predict using the model
        predicted_price = model.predict(input_data)[0]
        
        # Return predicted price to the user
        return jsonify({'predicted_price': predicted_price})

    except ValueError:
        return jsonify({'error': 'Please enter valid inputs.'})

if __name__ == '__main__':
    app.run(debug=True)

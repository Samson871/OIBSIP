from flask import Flask, render_template, request
import numpy as np
import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# File paths
MODEL_PATH = 'iris_classifier.pkl'
ENCODER_PATH = 'label_encoder.npy'

# Function to check and train the model if it doesn't exist
def train_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        print("Model or Label Encoder not found. Training model...")

        # Load dataset
        df = pd.read_csv('dataset/Iris.csv', index_col='Id')

        # Separate features and target
        X = df.drop('Species', axis=1)
        y = df['Species']

        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Encode target labels
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)

        # Train the model
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train_encoded)

        # Save the trained model and label encoder
        joblib.dump(clf, MODEL_PATH)
        np.save(ENCODER_PATH, label_encoder.classes_)

        print("Model and Label Encoder trained and saved.")

# Call train_model to ensure model and encoder are available
train_model()

# Load the model and encoder
model = joblib.load(MODEL_PATH)
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load(ENCODER_PATH, allow_pickle=True)

# Function to validate if input is numeric
def is_numeric(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    sepal_length = request.form['sepal_length']
    sepal_width = request.form['sepal_width']
    petal_length = request.form['petal_length']
    petal_width = request.form['petal_width']

    # Validate if all inputs are numeric
    if not (is_numeric(sepal_length) and is_numeric(sepal_width) and is_numeric(petal_length) and is_numeric(petal_width)):
        return render_template('index.html', prediction_text="Error: Please enter valid numeric values.")

    # Convert input to float if valid
    sepal_length = float(sepal_length)
    sepal_width = float(sepal_width)
    petal_length = float(petal_length)
    petal_width = float(petal_width)

    # Prepare the feature array for prediction
    features = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)

    # Make a prediction
    prediction = model.predict(features)
    predicted_species = label_encoder.inverse_transform(prediction)[0]

    # Render the result in the HTML template
    return render_template('index.html', prediction_text=f'Predicted Species: {predicted_species}')

if __name__ == '__main__':
    app.run(debug=True)

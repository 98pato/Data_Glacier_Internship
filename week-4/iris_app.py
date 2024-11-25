from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("iris_model.pkl", "rb"))

# Define the label mapping
label_mapping = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get form values and convert them to a float array
    float_features = [float(x) for x in request.form.values()]
    features = np.array([float_features])
    
    # Make prediction and map the result to the species name
    prediction = model.predict(features)[0]
    species_name = label_mapping[prediction]  # Map numeric prediction to species name

    # Render the template with the prediction
    return render_template("index.html", prediction_text="The flower species is {}".format(species_name))

if __name__ == "__main__":
    app.run(debug=True)

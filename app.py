from flask import Flask, render_template, request, url_for
import pickle
import numpy as np

# Load ML model
from werkzeug.utils import redirect

model = pickle.load(open('model.pkl', 'rb'))

# Create application
app = Flask(__name__)


# Bind home function to URL
@app.route('/')
def index():
    return render_template('index.html')


# Bind predict function to URL
@app.route('/predict', methods=['GET','POST'])
def predict():
    # Put all form entries values in a list
    features = [float(i) for i in request.form.values()]
    print(features)
    # Convert features to array
    array_features = [np.array(features)]
    # Predict features
    prediction = model.predict(array_features)
    probability = model.predict_proba(array_features)
    disease_chance = probability[0][1] * 100
    disease_chances = "{:.2f}".format(disease_chance)


    # Check the output values and retrive the result with html tag based on the value

    if prediction[0] == 0:
        if disease_chance > 29:
            return render_template('result3.html', disease_chances = disease_chances)
        else:
            return render_template('result4.html', disease_chances = disease_chances)
    else:
        if disease_chance < 70:
            return render_template('result2.html', disease_chances = disease_chances)
        else:

            return render_template('result1.html', disease_chances = disease_chances)

if __name__ == '__main__':
    # Run the application
    app.run()
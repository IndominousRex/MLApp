#import libraries
import numpy as np
from flask import Flask, request, render_template
import pickle

# Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

# default page of our web-app


@app.route('/')
def home():
    return render_template('index.html')

# To use the predict button in our web-app


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = np.zeros((1, 30))
    int_features[0, 0] = request.form.get("Distance driven")
    int_features[0, 1] = request.form.get("Age")
    int_features[0, 2] = request.form.get("Power")
    d = {"First Owner": 3, "Second Owner": 5,
         "Fourth Owner or more": 4, "Third Owner": 6, }
    o = request.form.get("Owner")
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(np.e ^ prediction[0], 2)

    return render_template('index.html', prediction_text='Price of the bike is :{}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)

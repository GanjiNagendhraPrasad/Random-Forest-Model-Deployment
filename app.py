from flask import Flask, render_template, request
import numpy as np
import pickle

# Load Logistic Regression model
with open('rf.pkl', 'rb') as f:
    lr_model = pickle.load(f)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None

    if request.method == 'POST':
        sl = float(request.form['SepalLengthCm'])
        sw = float(request.form['SepalWidthCm'])
        pl = float(request.form['PetalLengthCm'])
        pw = float(request.form['PetalWidthCm'])

        features = np.array([[sl, sw, pl, pw]])

        pred = lr_model.predict(features)[0]

        if pred == 0:
            prediction = "Setosa"
        elif pred == 1:
            prediction = "Versicolor"
        else:
            prediction = "Virginica"

    return render_template(
        'index.html',
        prediction=prediction
    )

if __name__ == '__main__':
    app.run(debug=True)
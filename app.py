from flask import Flask, render_template, request, flash, redirect
import pickle
import numpy as np


app = Flask(__name__)


def predict(values):
    if len(values) == 8:
        to_predict = np.array(values).reshape(1, 8)
        loaded_model = pickle.load(open("models/diabetes.pkl", "rb"))
        result = loaded_model.predict(to_predict)
        return result[0]

    elif len(values) == 13:
        to_predict = np.array(values).reshape(1, 13)
        loaded_model = pickle.load(open("models/heart.pkl", "rb"))
        result = loaded_model.predict(to_predict)
        return result[0]

    elif len(values) == 18:
        to_predict = np.array(values).reshape(1, 18)
        loaded_model = pickle.load(open("models/kidney.pkl", "rb"))
        result = loaded_model.predict(to_predict)
        return result[0]

    elif len(values) == 10:
        to_predict = np.array(values).reshape(1, 10)
        loaded_model = pickle.load(open("models/liver.pkl", "rb"))
        result = loaded_model.predict(to_predict)
        return result[0]


@ app.route("/")
def home():
    return render_template('home.html')


@ app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')


@ app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')


@ app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')


@ app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver.html')


@ app.route("/predict", methods=['GET', 'POST'])
def predictPage():

    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        pred = predict(to_predict_list)

        return render_template('predict.html', pred=pred)


if __name__ == '__main__':
       app.run(host='localhost', port=5000, debug=True)


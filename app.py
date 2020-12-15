from flask import Flask,render_template,url_for,request
import numpy as np
import pickle
import pandas as pd
# import flasgger
# from flasgger import Swagger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.decomposition import TruncatedSVD
# from sklearn.externals import joblib
import re


app = Flask(__name__)
# Swagger(app)

def preprocessor(text):
    pattern = (r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|"\
             + r"[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
    return re.sub(pattern, '', text)

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        Reviews = request.form['Reviews']
        data = [Reviews]
        my_prediction = model.predict(data)

    return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':

    app.run(debug=True)
    
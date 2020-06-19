import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
#import pickle
from joblib import load
app = Flask(__name__)
model = load('rf.save')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    x_test = [[float(x) for x in request.form.values()]]
    print(x_test)
    prediction = model.predict(x_test)
    print(prediction)
    output=prediction[0]
    if(output==1):
        pred="Readmits in the hospital"
    else:
        pred="Does not readmit in the hospital"

    
    return render_template('index.html', prediction_text='Patient {}'.format(pred))


if __name__ == "__main__":
    app.run(debug=True)

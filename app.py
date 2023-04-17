import pickle
from flask import Flask,request,app,jsonify,url_for,render_template,redirect,flash,session,escape
import numpy as np
import pandas as pd

app=Flask(__name__)
#Load the model
model=pickle.load(open('regmodel.pkl','rb'))
#scalar=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=model.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
     data=[float(x) for x in request.form.values()]
     final_input=model.transform(np.array(data).reshape(1,-1))
     print(final_input)
     output=model.predict(final_input)[0]
     return render_template("login.html",prediction_text="The predicted Load is {}".format(output))

if __name__=="__main__":
    app.run(debug=True)
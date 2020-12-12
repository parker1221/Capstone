from flask import Flask, render_template
from flask import request
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/homepage')
def homepage():
    return render_template('homepage.html')

@app.route('/results')
def results():
    df = pd.read_csv("https://raw.githubusercontent.com/parker1221/Capstone/main/day.csv")
    del df['season']
    del df['mnth']
    del df['weekday']
    del df['casual']
    del df['registered']
    del df['atemp']
    del df['dteday']
    del df['instant']
    X_train, X_test, y_train, y_test = train_test_split(df.drop('cnt', axis = 1), df['cnt'])
    logReg = LogisticRegression()
    logReg.fit(X_train, y_train)
    year = request.args.get('year')
    year = int(year) - 2011
    holiday = request.args.get('holiday')
    weekend = request.args.get('weekend')
    weather = request.args.get('weather')
    temperature = request.args.get('temperature')
    humidity = request.args.get('humidity')
    windspeed = request.args.get('windspeed')
    toTest = np.array([[year,holiday,weekend,weather,temperature,humidity,windspeed]])[0]
    print(toTest)
    prediction = logReg.predict(np.array([[int(year), int(holiday),int(weekend),int(weather),float(temperature),float(humidity),float(windspeed)]]))[0]
    print(prediction)
    y_real = df.cnt
    del df ['cnt']
    y_predict = logReg.predict(df)
    error = mean_absolute_error(y_real,y_predict)
    plusMinusError = error/2
    print(int(plusMinusError))
    return render_template('results.html', prediction = prediction, plusMinusError = int(plusMinusError))

@app.route('/visuals')
def visuals():


    return render_template('visuals.html')


if __name__ == "__main__":
    app.run(debug = True)
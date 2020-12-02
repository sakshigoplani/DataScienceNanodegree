############################################################################
#                                                                         ##
#                                                                         ##
#  Author:     Sakshi Haresh Goplani                                      ##
#  Project:    Capstone Project - Overnight Stock Direction Prediction    ##
#  Email:      sakshigoplani9@gmail.com                                   ##
#                                                                         ##
############################################################################

""" Web Application

This script launches web application through Flask Server for display of Data
Visualizations and Model Predictions for a Stock Symbol

Usage: python run_app.py

"""


import json
import time
from datetime import datetime
import ssl
from sqlalchemy import create_engine
import joblib
from plotly.graph_objs import Bar
from flask import render_template, request, jsonify
from flask import Flask
import pandas as pd
import plotly
import sys
sys.path.insert(1, '../data/')
sys.path.insert(2, '../model/')
from trainmodel import TrainModel
from dataprocess import DataProcess


# METADATA #
__version__ = 3.6
__author__ = 'Sakshi Haresh Goplani'
################


# To bypass certificate errors
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


app = Flask(__name__)

# Class Objects
data_processor = DataProcess()
train_model = TrainModel()

# Pull all available data starting year 2000 until today
start_date = datetime(2000, 1, 1)
end_date = datetime.date(datetime.today())

# Input Arguments
apikey = 'JXYRIAIGOFQQEYU6'
apikey2 = 'OMX6JTTUJ7VZ4MOJ'
apikey3 = 'O4170I9AFRMU1MWM'
apikey4 = 'L7WQ5800OSKGRWRD'
symbol = 'SPY'
interval = 'daily'
interval_vwap = '60min'
time_period_cci_5 = '5'
time_period_cci_20 = '20'
time_period_sma_50 = '50'
time_period_sma_200 = '200'
time_period_ema_9 = '9'
time_period_ema_20 = '20'
time_period_rsi_14 = '14'
series_type = 'close'


@app.route('/')
@app.route('/index')
def index():
    """ Renders Index Page for Stock Prediction Flask App 

    Args: <N/A>

    Returns:
        Render Flask Template for index path
    """

    # Get Data
    lvls, cci_5, cci_20, sma_50, sma_200, ema_9, ema_20, rsi_14, macd, cad, df, target, lastrow, features = \
        data_processor.get_data(symbol, apikey, apikey2,
                                start_date, end_date, interval, time_period_cci_5,
                                time_period_cci_20, time_period_sma_50, time_period_sma_200,
                                time_period_ema_9, time_period_ema_20, time_period_rsi_14,
                                series_type)

    # Create Visuals
    graphs = [data_processor.plot_lvls_ma(symbol, lvls, ema_9, ema_20, sma_50, sma_200),
              data_processor.plot_cci_rsi_cad(cci_5, cci_20, rsi_14, cad),
              data_processor.plot_macd(macd)]

    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # Render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

@app.route('/go')
def go():
    """ Web page that handles user query and displays model results

    Args: <N/A>

    Returns:
        Render Flask Template for index path
    """

    # Save user input in query
    query = request.args.get('query', '')

    # Sleep to give Rest to API
    time.sleep(60)

    # Get Data
    lvls, cci_5, cci_20, sma_50, sma_200, ema_9, ema_20, rsi_14, macd, cad, df, target, lastrow, features = \
        data_processor.get_data(query, apikey3, apikey4,
                                start_date, end_date, interval, time_period_cci_5,
                                time_period_cci_20, time_period_sma_50, time_period_sma_200,
                                time_period_ema_9, time_period_ema_20, time_period_rsi_14,
                                series_type)

    # Create Visuals
    graphs = [data_processor.plot_lvls_ma(query, lvls, ema_9, ema_20, sma_50, sma_200),
              data_processor.plot_cci_rsi_cad(cci_5, cci_20, rsi_14, cad),
              data_processor.plot_macd(macd)]

    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # Train Basic CLF Model
    train_score, test_score, tomorrow_change, \
        probability_of_change = train_model.create_clf_basic(
            features, target, lastrow)

    # Train CLF CV Model
    train_score_cv, test_score_cv, tomorrow_change_cv, \
        probability_of_change_cv = train_model.create_clf_cv(
            features, target, lastrow)

    # Train Decision Tree Model
    train_score_dt, test_score_dt, tomorrow_change_dt, \
        probability_of_change_dt = train_model.create_dtree(
            features, target, lastrow)

    return render_template(
        'go.html',
        query=query,
        ids=ids, graphJSON=graphJSON, train_score=train_score,
        test_score=test_score, tomorrow_change=tomorrow_change,
        probability_of_change=probability_of_change,
        train_score_cv=train_score_cv, test_score_cv=test_score_cv,
        tomorrow_change_cv=tomorrow_change_cv,
        probability_of_change_cv=probability_of_change_cv,
        train_score_dt=train_score_dt, test_score_dt=test_score_dt,
        tomorrow_change_dt=tomorrow_change_dt,
        probability_of_change_dt=probability_of_change_dt
    )

def main():
    app.run(host='localhost', port=4200, debug=True)


if __name__ == '__main__':
    main()

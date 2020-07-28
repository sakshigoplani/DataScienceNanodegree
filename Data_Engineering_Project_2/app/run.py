############################################################################
#                                                                         ##
#                                                                         ##
#  Author:     Sakshi Haresh Goplani                                      ##
#  Project:    Data Engineering - Disaster Response Pipeline              ##
#  Email:      sakshigoplani9@gmail.com                                   ##
#                                                                         ##
############################################################################

""" Web Application

This script launches web application through Flask Server for interaction
with pkl model and sql database

Usage: python run.py

"""


import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar

# Deprecated
# from sklearn.externals import joblib
import joblib

from sqlalchemy import create_engine

# Download nltk libraries
import nltk
import ssl
# To bypass certificate errors
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download(['punkt', 'wordnet'])


app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine("sqlite:///../data/ETLDB.db")

df = pd.read_sql_table('ETLTable', engine)

# load model
model = joblib.load("../models/clf_cv_1.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # Message Count by Genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Message Count/Percentage by Category
    category_names = list(df.columns)[4:]
    category_counts = []
    category_precentages = {}

    for col in category_names:
        category_counts.append(df.shape[0] - df.groupby(col).count()['id'][0])
        category_precentages[col] = ((df.shape[0] - df.groupby(col).count()['id'][0]) / (df.shape[0]))*100

    top_category_percentages = {k:v for k,v in sorted(category_precentages.items(), 
                                key = lambda item : item[1], reverse=True)[:5]}

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=list(top_category_percentages.keys()),
                    y=list(top_category_percentages.values())
                )
            ],

            'layout': {
                'title': 'Percentage Distribution of Message Categories',
                'yaxis': {
                    'title': "Percentage Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='localhost', port=4200, debug=True)


if __name__ == '__main__':
    main()

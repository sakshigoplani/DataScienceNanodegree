############################################################################
#                                                                         ##
#                                                                         ##
#  Author:     Sakshi Haresh Goplani                                      ##
#  Tool:       Combine multiple IP-XACT Files for Qsys Processing         ##
#  Email:      sakshigoplani9@gmail.com                                   ##
#                                                                         ##
############################################################################

""" Model Builder Utility

This script takes in a path where database and table resides. It goes
over the data and fits and predicts to generate a model pkl file.

Usage: python train_classifier.py <database_filepath> <model_filepath>
                                    <tablename> <model_type> <gridsearch>

"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score
nltk.download(['punkt', 'wordnet'])


# METADATA #
__version__ = 3.6
__author__ = 'Sakshi Haresh Goplani'
################


class EmergencyMessageExtractor(BaseEstimator, TransformerMixin):

    def emergency_message(self, text):
        if "emergency" in text.lower() or "urgent" in text.lower():
            return True
        else:
            return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.emergency_message)
        return pd.DataFrame(X_tagged)


def argument_sanitization(database_filepath):
    """ Validate file paths

    Args:
        database_filepath (string): Path of the DB File

    Returns:
        N/A

    """

    if "sqlite:///" in database_filepath:
        database_filepath = re.sub("sqlite:///", "", database_filepath)
    if not os.path.isfile(database_filepath):
        logger.error("{} is not valid".format(database_filepath))


def pipeline_model_helper(pipeline_type='clf'):
    """ Return a pipeline of user's choice

    Args:
        pipeline_type (string): Type of Pipeline to return
                                'clf', 'kn' or 'customtrans'

    Returns:
        model (pipeline): Pipeline for model of choice
        parameters (dictionary): Dictionary of Params to CV

    """

    clf_pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    param_clf = {
    'clf__estimator__class_weight': (None, 'balanced', 'balanced_subsample'),
    # 'clf__estimator__criterion': ('gini', 'entropy'),
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__max_features': (None, 5000, 10000, 50000),
    'tfidf__norm': ('l2', 'l1'),
    # 'tfidf__use_idf': (True, False)
    }

    kn_pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('kn', MultiOutputClassifier(KNeighborsClassifier()))
    ])

    param_kn = {}

    customtrans_pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('emergency_message', EmergencyMessageExtractor())
        ])),
        ('clfcv', MultiOutputClassifier(RandomForestClassifier()))
    ])

    param_customtrans = {}

    if pipeline_type == 'clf':
        return clf_pipeline, param_clf
    elif pipeline_type == 'kn':
        return kn_pipeline, param_kn
    elif pipeline_type == 'customtrans':
        return customtrans_pipeline, param_customtrans
    else:
        return clf_pipeline, param_clf


def grid_search_model_helper(pipeline, parameters):
    """ Return a GridSeach Pipeline

    Args:
        pipeline (pipeline): Pipeline to setup GridSeach CV for
        parameters (dictionary): Parameters of Model to run CV for

    Returns:
        model (pipeline): GridSeach CV Model Pipeline

    """

    model = GridSearchCV(pipeline, param_grid=parameters)
    return model


def load_data(database_filepath, tablename):
    """ Load data from the database and
        return Features and Prediction Columns

    Args:
        database_filepath (string): Path of the DB File
        tablename (string): Name of the table inside DB File

    Returns:
        X, Y (numpy array): Feature and Prediction data
        category_columns (array): List of column names

    """

    # Check if file paths are valid
    argument_sanitization(database_filepath)

    # Load from SQL DB
    # engine = create_engine(database_filepath)
    # df = pd.read_sql("SELECT * FROM {}".format(tablename), engine)
    df = pd.read_csv("/nfs/sc/disks/swuser_work_adadlani/RANDOM_WB/ml/data/clf_basic.csv")

    # Generate Features and Prediction columns
    X = df.message.values
    Y = df.iloc[:, 5:].values
    category_names = df.iloc[:, 5:].columns

    return X, Y, category_names


def tokenize(text):
    """ Tokenize/Lemmatize/LowerCase/Strip/Clean Text

    Args:
        text (string): Text to tokenize

    Returns:
        clean_tokens (array): List of clean tokens

    """

    url_regex = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]" \
                "|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model(pipeline_type, gridsearch=False, parameters=None):
    """ Build Model

    Args:
        pipeline_type (string): Type of Pipeline to return
        gridseach (boolean): Whether or not to run GridSearch CV
        parameters (dictionary): Hyperparameters to sweep for GridSeach CV

    Returns:
        model (pipeline): Final pipeline with relevant stages

    """

    model, parameters = pipeline_model_helper(pipeline_type)
    if gridsearch and parameters is not None and parameters != {}:
        return grid_search_model_helper(model, parameters)
    else:
        return model


def evaluate_model(model, X_test, Y_test, category_names):
    """ Report Model Accuracy Metrics

    Args:
        model (model object): Model to predict on
        X_test, Y_test (numpy Array of Arrays): Feature and
                                                Prediction Dataframes
        category_names (array): List of Category Columns

    Returns:
        N/A

    """

    Y_pred = model.predict(X_test)
    for col in range(len(category_names)):
        logger.info("Column Name is: \n {} ".format(
            category_names[col])
            )
        logger.info("Classification Report for {}: \n {} \n\n".format(
            category_names[col],
            classification_report((Y_test[:, col]), (Y_pred[:, col])))
            )
        logger.info("Confusion Matrix for {}: \n {} \n\n".format(
            category_names[col],
            confusion_matrix(Y_test[:, col], Y_pred[:, col]))
            )
        logger.info("Accuracy for {}: \n {} \n\n".format(
            category_names[col], (Y_pred[:, col] == Y_test[:, col]).mean())
            )


def save_model(model, model_filepath):
    """ Save model to pkl file

    Args:
        model (model object): Model to predict on
        model_filepath (string): Path to store pkl file at

    Returns:
        N/A

    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 6:
        database_filepath, \
            model_filepath, \
            tablename, \
            model_type, \
            gridsearch = sys.argv[1:]
        logger.info(" Loading data...\n    DATABASE: {}"
                    .format(database_filepath))
        if model_type not in ['clf', 'kn', 'customtrans']:
            logger.error(
                "Please select model type from ['clf', 'kn', 'customtrans']"
                )
        if str(type(gridsearch)) is not 'bool':
            logger.error("Please select gridsearch as True or False")
        X, Y, category_names = load_data(database_filepath, tablename)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2
            )

        logger.info(" Building model...")
        model = build_model(model_type, gridsearch)

        logger.info(" Training model...")
        model.fit(X_train, Y_train)

        logger.info(" Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        logger.info(" Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        logger.info(" Trained model saved!")

    else:
        logger.info(
            """
                Please provide the filepath of the disaster messages database
                as the first argument and the filepath of the pickle file to
                save the model to as the second argument. \n\nExample: python
                train_classifier.py ../data/DisasterResponse.db classifier.pkl
            """
              )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    main()

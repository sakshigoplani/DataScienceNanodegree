############################################################################
#                                                                         ##
#                                                                         ##
#  Author:     Sakshi Haresh Goplani                                      ##
#  Project:    Capstone Project - Overnight Stock Direction Prediction    ##
#  Email:      sakshigoplani9@gmail.com                                   ##
#                                                                         ##
############################################################################

""" Utility Class for Model Training

This script provides a utility class for training different flavors of models for
prediction of overnight stock direction

Usage: <Called by run_app.py>

"""


import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import requests
import pandas_datareader.data as pdr
from datetime import datetime
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sns


# METADATA #
__version__ = 3.6
__author__ = 'Sakshi Haresh Goplani'
################


class TrainModel:

    def __init__(self):
        pass

    def create_clf_basic(self, features, target, lastrow):
        """ Creates model for Basic RandomForestClassifier

        Args:
            features (pandas dataframe): Feature Dataset in pandas dataframe format
            target (pandas dataframe): Predictor Dataset in pandas dataframe format
            lastrow (pandas dataframe): Row for today used to predict overnight direction

        Returns:
            train_score (float): Training Score
            test_score (float): Testing Score
            tomorrow_change (numpy array): Direction predicted (1 for Up, -1 for Down)
            probability_of_change (numpy array): List of Probabilities for upward movement and downward movement
        """

        classifier_entropy = RandomForestClassifier(
            criterion='entropy', random_state=1, oob_score=True, n_estimators=100, max_depth=5, min_samples_leaf=20, min_samples_split=30)
        features_train, features_test, targetclass_train, targetclass_test = train_test_split(
            features, target, test_size=0.25, random_state=0)
        classifier_entropy.fit(features_train, targetclass_train)
        train_score, test_score, tomorrow_change, probability_of_change = self.predict_utility(
            classifier_entropy, features_train, targetclass_train, features_test, targetclass_test, lastrow)
        return train_score, test_score, tomorrow_change, probability_of_change

    def grid_search_util(self, model, is_grid_search=False, parameters=None):
        """Calculates the number of rows and columns in dataset

        Args:
            df (pandas dataframe): Dataset in pandas dataframe format

        Returns:
            rows (int): Number of rows in dataset
            cols (int): Number of columns in dataset
        """
        if is_grid_search:
            model_gs = GridSearchCV(model, param_grid=parameters)
        else:
            model_gs = model
        return model_gs

    def create_clf_cv(self, features, target, lastrow):
        """ Creates model for RandomForestClassifier with GridSearch CV

        Args:
            features (pandas dataframe): Feature Dataset in pandas dataframe format
            target (pandas dataframe): Predictor Dataset in pandas dataframe format
            lastrow (pandas dataframe): Row for today used to predict overnight direction

        Returns:
            train_score (float): Training Score
            test_score (float): Testing Score
            tomorrow_change (numpy array): Direction predicted (1 for Up, -1 for Down)
            probability_of_change (numpy array): List of Probabilities for upward movement and downward movement
        """

        param_clf = {
            'class_weight': (None, 'balanced', 'balanced_subsample'),
            'criterion': ('gini', 'entropy'),
            'n_estimators': (100, 200),
            'max_depth': (2, 5)
        }
        classifier_entropy_cv = RandomForestClassifier(
            random_state=1, oob_score=True, min_samples_leaf=20, min_samples_split=30)
        classifier_entropy_cv_gs = self.grid_search_util(
            classifier_entropy_cv, True, param_clf)
        features_train, features_test, targetclass_train, targetclass_test = train_test_split(
            features, target, test_size=0.25, random_state=0)
        train_score, test_score, tomorrow_change, probability_of_change = self.predict_utility(
            classifier_entropy_cv_gs, features_train, targetclass_train, features_test, targetclass_test, lastrow)
        return train_score, test_score, tomorrow_change, probability_of_change

    def create_dtree(self, features, target, lastrow):
        """ Creates model for Decision Tree Classifier

        Args:
            features (pandas dataframe): Feature Dataset in pandas dataframe format
            target (pandas dataframe): Predictor Dataset in pandas dataframe format
            lastrow (pandas dataframe): Row for today used to predict overnight direction

        Returns:
            train_score (float): Training Score
            test_score (float): Testing Score
            tomorrow_change (numpy array): Direction predicted (1 for Up, -1 for Down)
            probability_of_change (numpy array): List of Probabilities for upward movement and downward movement
        """
        classifier_entropy_dt = DecisionTreeClassifier(
            criterion='entropy', random_state=1, max_depth=5, min_samples_split=6, min_samples_leaf=5)
        features_train, features_test, targetclass_train, targetclass_test = train_test_split(
            features, target, test_size=0.25, random_state=0)
        train_score, test_score, tomorrow_change, probability_of_change = self.predict_utility(
            classifier_entropy_dt, features_train, targetclass_train, features_test, targetclass_test, lastrow)
        return train_score, test_score, tomorrow_change, probability_of_change

    def predict_utility(self, model, features_train, targetclass_train, features_test, targetclass_test, lastrow):
        """ Performs repetitive task of prediction for different models

        Args:
            model (Model Object): Model object using which prediction needs to be done 
            features_train (pandas dataframe): Feature data in training dataset in pandas dataframe format
            targetclass_train (pandas dataframe): Predictor data in testing dataset in pandas dataframe format
            features_test (pandas dataframe): Feature data in testing dataset in pandas dataframe format
            targetclass_test (pandas dataframe): Predictor data in testing dataset in pandas dataframe format
            lastrow (pandas dataframe): Row for today used to predict overnight direction

        Returns:
            train_score (float): Training Score
            test_score (float): Testing Score
            tomorrow_change (numpy array): Direction predicted (1 for Up, -1 for Down)
            probability_of_change (numpy array): List of Probabilities for upward movement and downward movement
        """

        model.fit(features_train, targetclass_train)
        train_score = model.score(
            features_train, targetclass_train)
        test_score = model.score(
            features_test, targetclass_test)
        tomorrow_change = model.predict(lastrow)
        probability_of_change = model.predict_proba(lastrow)
        return train_score, test_score, tomorrow_change, probability_of_change

############################################################################
#                                                                         ##
#                                                                         ##
#  Author:     Sakshi Haresh Goplani                                      ##
#  Project:    Capstone Project - Overnight Stock Direction Prediction    ##
#  Email:      sakshigoplani9@gmail.com                                   ##
#                                                                         ##
############################################################################

""" Utility Class for Data Ingestion

This script provides a utility class for obtaining data for a stock symbol
which can be used to write features for our data

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
import urllib.parse
from sqlalchemy import create_engine
import argparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# METADATA #
__version__ = 3.6
__author__ = 'Sakshi Haresh Goplani'
################


class DataProcess:

    def __init__(self):
        pass

    def request_levels(self, symbol, apikey, start_date=datetime(2010, 1, 1), end_date=datetime.date(datetime.today())):
        """ Fetch data on Open, Close, High and Low daily stock price levels for a timeframe

        Args:
            symbol (string): Stock symbol
            apikey (string): API Key needed for AlphaVantage data request
            start_date (datetime object): Start fetching data from this date if available
            end_date (datetime object): End fetching data on this date

        Returns:
            lvls (pandas dataframe): Open, Close, High and Low daily stock price levels
        """

        rest_link = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={0}&apikey={1}&outputsize=full".format(symbol, apikey)
        response = requests.get(rest_link, verify=False)
        lvlsdict = json.loads(response.text)
        lvls = pd.DataFrame(lvlsdict['Time Series (Daily)']).T
        lvls.index = pd.to_datetime(lvls.index)
        lvls = lvls.sort_index(ascending = True)
        lvls = lvls.loc[start_date:end_date]
        lvls.columns = ['open', 'high', 'low', 'close', 'adjusted_close', 'volume', 'dividend_amount', 'split_coefficient']
        lvls = lvls.astype(float)
        lvls['return'] = (lvls['open'] - lvls['adjusted_close'].shift(1)) / lvls['adjusted_close'].shift(1)
        lvls['direction'] = np.where(lvls['return'] > 0, 1, -1)
        return lvls

    def request_cci(self, symbol, apikey, interval, time_period, start_date=datetime(2010, 1, 1), end_date=datetime.date(datetime.today())):
        """ Fetch data on Commodity Channel Index (CCI) levels for a timeframe

        Args:
            symbol (string): Stock symbol
            apikey (string): API Key needed for AlphaVantage data request
            interval (string): Interval between two data points (Daily, Hourly, etc.)
            time_period (string): Number of data points for trailing calculation
            start_date (datetime object): Start fetching data from this date if available
            end_date (datetime object): End fetching data on this date

        Returns:
            cci (pandas dataframe): CCI Data
        """     

        rest_link = "https://www.alphavantage.co/query?function=CCI&symbol={0}&apikey={1}&interval={2}&time_period={3}".format(symbol, apikey, interval, time_period)
        response = requests.get(rest_link, verify=False)
        ccidict = json.loads(response.text)
        cci = pd.DataFrame(ccidict['Technical Analysis: CCI']).T
        cci.index = pd.to_datetime(cci.index)
        cci = cci.sort_index(ascending = True)
        cci = cci.loc[start_date:end_date]
        cci.columns = ['cci_'+time_period+'_'+interval]
        cci = cci.astype(float)
        return cci

    def request_sma(self, symbol, apikey, interval, time_period, series_type, start_date=datetime(2010, 1, 1), end_date=datetime.date(datetime.today())):
        """ Fetch data on Simple Moving Average (SMA) levels for a timeframe

        Args:
            symbol (string): Stock symbol
            apikey (string): API Key needed for AlphaVantage data request
            interval (string): Interval between two data points (Daily, Hourly, etc.)
            time_period (string): Number of data points for trailing calculation
            series_type (string): Type of price level to consider for trailing calculation
            start_date (datetime object): Start fetching data from this date if available
            end_date (datetime object): End fetching data on this date

        Returns:
            sma (pandas dataframe): SMA Data
        """

        rest_link = "https://www.alphavantage.co/query?function=SMA&symbol={0}&apikey={1}&interval={2}&time_period={3}&series_type={4}".format(symbol, apikey, interval, time_period, series_type)
        response = requests.get(rest_link, verify=False)
        smadict = json.loads(response.text)
        sma = pd.DataFrame(smadict['Technical Analysis: SMA']).T
        sma.index = pd.to_datetime(sma.index)
        sma = sma.sort_index(ascending = True)
        sma = sma.loc[start_date:end_date]
        sma.columns = ['sma_'+time_period+'_'+interval]
        sma = sma.astype(float)
        return sma

    def request_ema(self, symbol, apikey, interval, time_period, series_type, start_date=datetime(2010, 1, 1), end_date=datetime.date(datetime.today())):
        """ Fetch data on Exponential Moving Average (EMA) levels for a timeframe

        Args:
            symbol (string): Stock symbol
            apikey (string): API Key needed for AlphaVantage data request
            interval (string): Interval between two data points (Daily, Hourly, etc.)
            time_period (string): Number of data points for trailing calculation
            series_type (string): Type of price level to consider for trailing calculation
            start_date (datetime object): Start fetching data from this date if available
            end_date (datetime object): End fetching data on this date

        Returns:
            ema (pandas dataframe): EMA Data
        """

        rest_link = "https://www.alphavantage.co/query?function=EMA&symbol={0}&apikey={1}&interval={2}&time_period={3}&series_type={4}".format(symbol, apikey, interval, time_period, series_type)
        response = requests.get(rest_link, verify=False)
        emadict = json.loads(response.text)
        ema = pd.DataFrame(emadict['Technical Analysis: EMA']).T
        ema.index = pd.to_datetime(ema.index)
        ema = ema.sort_index(ascending = True)
        ema = ema.loc[start_date:end_date]
        ema.columns = ['ema_'+time_period+'_'+interval]
        ema = ema.astype(float)
        return ema

    def request_rsi(self, symbol, apikey, interval, time_period, series_type, start_date=datetime(2010, 1, 1), end_date=datetime.date(datetime.today())):
        """ Fetch data on Relative Strength Index (RSI) levels for a timeframe

        Args:
            symbol (string): Stock symbol
            apikey (string): API Key needed for AlphaVantage data request
            interval (string): Interval between two data points (Daily, Hourly, etc.)
            time_period (string): Number of data points for trailing calculation
            series_type (string): Type of price level to consider for trailing calculation
            start_date (datetime object): Start fetching data from this date if available
            end_date (datetime object): End fetching data on this date

        Returns:
            rsi (pandas dataframe): RSI Data
        """

        rest_link = "https://www.alphavantage.co/query?function=RSI&symbol={0}&apikey={1}&interval={2}&time_period={3}&series_type={4}".format(symbol, apikey, interval, time_period, series_type)
        response = requests.get(rest_link, verify=False)
        rsidict = json.loads(response.text)
        rsi = pd.DataFrame(rsidict['Technical Analysis: RSI']).T
        rsi.index = pd.to_datetime(rsi.index)
        rsi = rsi.sort_index(ascending = True)
        rsi = rsi.loc[start_date:end_date]
        rsi.columns = ['rsi_'+time_period+'_'+interval]
        rsi = rsi.astype(float)
        return rsi

    def request_macd(self, symbol, apikey, interval, series_type, start_date=datetime(2010, 1, 1), end_date=datetime.date(datetime.today())):
        """ Fetch data on Moving Average Convergence Divergence (MACD) levels for a timeframe

        Args:
            symbol (string): Stock symbol
            apikey (string): API Key needed for AlphaVantage data request
            interval (string): Interval between two data points (Daily, Hourly, etc.)
            series_type (string): Type of price level to consider for trailing calculation
            start_date (datetime object): Start fetching data from this date if available
            end_date (datetime object): End fetching data on this date

        Returns:
            macd (pandas dataframe): MACD Data
        """

        rest_link = "https://www.alphavantage.co/query?function=MACD&symbol={0}&apikey={1}&interval={2}&series_type={3}".format(symbol, apikey, interval, series_type)
        response = requests.get(rest_link, verify=False)
        macddict = json.loads(response.text)
        macd = pd.DataFrame(macddict['Technical Analysis: MACD']).T
        macd.index = pd.to_datetime(macd.index)
        macd = macd.sort_index(ascending = True)
        macd = macd.loc[start_date:end_date]
        macd.columns = ['MACD_'+interval, 'MACD_Hist_'+interval, 'MACD_Signal_'+interval]
        macd = macd.astype(float)
        return macd

    def request_cad(self, symbol, apikey, interval, start_date=datetime(2010, 1, 1), end_date=datetime.date(datetime.today())):
        """ Fetch data on Chaikin Accumulation/Distribution (Chaikin A/D) levels for a timeframe

        Args:
            symbol (string): Stock symbol
            apikey (string): API Key needed for AlphaVantage data request
            interval (string): Interval between two data points (Daily, Hourly, etc.)
            start_date (datetime object): Start fetching data from this date if available
            end_date (datetime object): End fetching data on this date

        Returns:
            cad (pandas dataframe): CAD Data
        """

        rest_link = "https://www.alphavantage.co/query?function=AD&symbol={0}&apikey={1}&interval={2}".format(symbol, apikey, interval)
        response = requests.get(rest_link, verify=False)
        caddict = json.loads(response.text)
        cad = pd.DataFrame(caddict['Technical Analysis: Chaikin A/D']).T
        cad.index = pd.to_datetime(cad.index)
        cad = cad.sort_index(ascending = True)
        cad = cad.loc[start_date:end_date]
        cad.columns = ['cad']
        cad = cad.astype(float)
        return cad

    def request_market(self, apikey, start_date=datetime(2010, 1, 1), end_date=datetime.date(datetime.today())):
        """ Fetch data on S&P 500 Index(SPY), Dollar Index(UUP) and Junk Bond Index(JNK) as features for prediction

        Args:
            apikey (string): API Key needed for AlphaVantage data request
            start_date (datetime object): Start fetching data from this date if available
            end_date (datetime object): End fetching data on this date

        Returns:
            features (pandas dataframe): Feature data for SPY, UUP and JNK
        """

        symb = ['SPY', 'UUP', 'JNK']
        features = pd.DataFrame()
        rest_links = []
        for symbol in symb:
            print(symbol)
            rest_link = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={0}&apikey={1}&outputsize=full".format(symbol, apikey)
            response = requests.get(rest_link, verify=False)
            lvlsdict = json.loads(response.text)
            lvls = pd.DataFrame(lvlsdict['Time Series (Daily)']).T
            lvls.index = pd.to_datetime(lvls.index)
            lvls = lvls.sort_index(ascending = True)
            lvls = lvls.loc[start_date:end_date]
            lvls.columns = ['open', 'high', 'low', 'close', 'adjusted_close', 'volume', 'dividend_amount', 'split_coefficient']
            lvls = lvls.astype(float)
            # if symbol == 'VIX':
            #     features[symbol] = lvls['adjusted_close'].diff()
            # else:
            features[symbol] = lvls['adjusted_close'].pct_change(1)*100
            features.index = lvls.index
        return features

    def get_data(self, symbol, apikey, apikey2, 
                start_date, end_date, interval, time_period_cci_5, 
                time_period_cci_20, time_period_sma_50, time_period_sma_200,
                time_period_ema_9, time_period_ema_20, time_period_rsi_14,
                series_type):
        """ Create final feature dataframe with all the data fetched (Levels, CCI, RSI, CAD, MACD, EMA, SMA)

        Args:
            symbol (string): Stock symbol
            apikey (string): API Key needed for AlphaVantage data request
            apikey2 (string): API Key needed for AlphaVantage data request
            start_date (datetime object): Start fetching data from this date if available
            end_date (datetime object): End fetching data on this date
            time_period_cci_5 (pandas dataframe): Time Period for CCI with value 5
            time_period_cci_20 (pandas dataframe): Time Period for CCI with value 20
            time_period_sma_50 (pandas dataframe): Time Period for SMA with value 50
            time_period_sma_200 (pandas dataframe): Time Period for SMA with value 200
            time_period_ema_9 (pandas dataframe): Time Period for EMA with value 9
            time_period_ema_20 (pandas dataframe): Time Period for EMA with value 20
            time_period_rsi_14 (pandas dataframe): Time Period for RSI with value 14
            series_type (string): Type of price level to consider for trailing calculation

        Returns:
            lvls (pandas dataframe): Levels Dataset
            cci_5 (pandas dataframe): Dataset for CCI with time period 5
            cci_20 (pandas dataframe): Dataset for CCI with time period 20
            sma_50 (pandas dataframe): Dataset for SMA with time period 50
            sma_200 (pandas dataframe): Dataset for SMA with time period 200
            ema_9 (pandas dataframe): Dataset for EMA with time period 9
            ema_20 (pandas dataframe): Dataset for EMA with time period 20
            rsi_14 (pandas dataframe): Dataset for RSI with time period 14
            macd (pandas dataframe): Dataset for MACD
            cad (pandas dataframe): Dataset for CAD
            df (pandas dataframe): Dataset with all of the above features
            target (pandas dataframe): Dataset with Prediction Column
            features (pandas dataframe): Feature Dataset in pandas dataframe format
            lastrow (pandas dataframe): Row for today used to predict overnight direction
        """    

        lvls = self.request_levels(symbol, apikey, start_date, end_date)
        cci_5 = self.request_cci(symbol, apikey, interval, time_period_cci_5, start_date, end_date)
        cci_20 = self.request_cci(symbol, apikey, interval, time_period_cci_20, start_date, end_date)
        sma_50 = self.request_sma(symbol, apikey, interval, time_period_sma_50, series_type, start_date, end_date)
        sma_200 = self.request_sma(symbol, apikey, interval, time_period_sma_200, series_type, start_date, end_date)
        time.sleep(60) # Needed because using demo key we can only make 5 API Requests per minute
        ema_9 = self.request_ema(symbol, apikey2, interval, time_period_ema_9, series_type, start_date, end_date)
        ema_20 = self.request_ema(symbol, apikey2, interval, time_period_ema_20, series_type, start_date, end_date)
        rsi_14 = self.request_rsi(symbol, apikey2, interval, time_period_rsi_14, series_type, start_date, end_date)
        macd = self.request_macd(symbol, apikey2, interval, series_type, start_date, end_date)
        cad = self.request_cad(symbol, apikey2, interval, start_date, end_date)

        # Data Preparation
        df = pd.concat([lvls, cci_5, cci_20, sma_50, sma_200, ema_9, ema_20, rsi_14, macd, cad], axis=1, join='inner')
        df = df.dropna()
        df = df.drop(axis=1, columns='return')
        target = df['direction']
        features = df.drop(axis=1, columns='direction')
        target = target[1:]
        lastrow = features[-1:]
        features = features[:-1]

        return lvls, cci_5, cci_20, sma_50, sma_200, ema_9, ema_20, rsi_14, macd, cad, df, target, lastrow, features

    def plot_lvls_ma(self, symbol, lvls, ema_9, ema_20, sma_50, sma_200):
        """ Visualize Levels, SMA and EMA

        Args:
            symbol (string): Stock Symbol
            start_date (datetime object): Start fetching data from this date if available
            end_date (datetime object): End fetching data on this date
            lvls (pandas dataframe): Levels Dataset
            ema_9 (pandas dataframe): Dataset for EMA with time period 9
            ema_20 (pandas dataframe): Dataset for EMA with time period 20
            sma_50 (pandas dataframe): Dataset for SMA with time period 50
            sma_200 (pandas dataframe): Dataset for SMA with time period 200
        Returns:
            lvls_ma_dict (dict): Dictionary containing plotly graph information for plotting
        """

        candle_lvls = go.Candlestick(x=lvls.index, open=lvls['open'], high=lvls['high'], low=lvls['low'], close=lvls['close'], name='CandleStick Chart {0}'.format(symbol))
        ema_9_g = go.Scatter(x=ema_9.index, y=ema_9.ema_9_daily, line=dict(color='black', width=2), name='EMA 9')
        ema_20_g = go.Scatter(x=ema_20.index, y=ema_20.ema_20_daily, line=dict(color='yellow', width=2), name='EMA 20')
        sma_50_g = go.Scatter(x=sma_50.index, y=sma_50.sma_50_daily, line=dict(color='orange', width=2), name='SMA 50')
        sma_200_g = go.Scatter(x=sma_200.index, y=sma_200.sma_200_daily, line=dict(color='brown', width=2), name='SMA 200')
        fig = go.Figure(data=[candle_lvls, ema_9_g, ema_20_g, sma_50_g, sma_200_g])
        fig.update_yaxes(fixedrange=False)
        rangeselector=dict(
            visible = True,
            bgcolor = 'rgba(150, 200, 250, 0.4)',
            font = dict( size = 13 ),
            buttons=list([
                dict(count=1,
                    label='reset',
                    step='all'),
                dict(count=1,
                    label='1yr',
                    step='year',
                    stepmode='backward'),
                dict(count=3,
                    label='3 mo',
                    step='month',
                    stepmode='backward'),
                dict(count=1,
                    label='1 mo',
                    step='month',
                    stepmode='backward'),
                dict(step='all')
            ]))
            
        fig['layout']['xaxis']['rangeselector'] = rangeselector
        lvls_ma_dict = fig.to_dict()
        return lvls_ma_dict

    def plot_cci_rsi_cad(self, cci_5, cci_20, rsi_14, cad):
        """ Visualize subplot for CCI, RSI and CAD
        Args:
            cci_5 (pandas dataframe): Dataset for CCI with time period 5
            cci_20 (pandas dataframe): Dataset for CCI with time period 20
            rsi_14 (pandas dataframe): Dataset for RSI with time period 14
            cad (pandas dataframe): Dataset for CAD
        Returns:
            cci_rsi_cad_dict (dict): Dictionary containing plotly graph information for plotting
        """

        cci_5_g = go.Scatter(x=cci_5.index, y=cci_5.cci_5_daily, line=dict(color='black', width=1), name='CCI 5')
        cci_20_g = go.Scatter(x=cci_20.index, y=cci_20.cci_20_daily, line=dict(color='black', width=1), name='CCI 20')
        rsi_14_g = go.Scatter(x=rsi_14.index, y=rsi_14.rsi_14_daily, line=dict(color='black', width=1), name='RSI 14')
        cad_g = go.Scatter(x=cad.index, y=cad.cad, line=dict(color='black', width=1), name='Chaikin A/D')
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("RSI 14", "CCI 5", "CCI 20", "Chaikin A/D"))
        fig.append_trace(rsi_14_g, row=1, col=1)
        fig.append_trace(cci_5_g, row=2, col=1)
        fig.append_trace(cci_20_g, row=3, col=1)
        fig.append_trace(cad_g, row=4, col=1)
        fig.update_xaxes(row=4, col=1, rangeslider_thickness=0.05)
        fig.update_yaxes(fixedrange=False)
        cci_rsi_cad_dict = fig.to_dict()
        return cci_rsi_cad_dict

    def plot_macd(self, macd):
        """ Visualize MACD Plot
        Args:
            macd (pandas dataframe): Dataset for MACD
        Returns:
            macd_dict (dict): Dictionary containing plotly graph information for plotting
        """

        macd_histo_g = go.Bar(x=macd.index, y=macd.MACD_Hist_daily, name='MACD Histogram')
        macd_daily_g = go.Scatter(x=macd.index, y=macd.MACD_daily, line=dict(color='red', width=2), name='MACD Daily')
        macd_signal_daily_g = go.Scatter(x=macd.index, y=macd.MACD_Signal_daily, line=dict(color='green', width=2), name='MACD Signal')
        fig = go.Figure(data=[macd_histo_g, macd_daily_g, macd_signal_daily_g])
        fig.update_yaxes(fixedrange=False)
        rangeselector=dict(
            visible = True,
            bgcolor = 'rgba(150, 200, 250, 0.4)',
            font = dict( size = 13 ),
            buttons=list([
                dict(count=1,
                    label='reset',
                    step='all'),
                dict(count=1,
                    label='1yr',
                    step='year',
                    stepmode='backward'),
                dict(count=3,
                    label='3 mo',
                    step='month',
                    stepmode='backward'),
                dict(count=1,
                    label='1 mo',
                    step='month',
                    stepmode='backward'),
                dict(step='all')
            ]))
            
        fig['layout']['xaxis']['rangeselector'] = rangeselector
        macd_dict = fig.to_dict()
        return macd_dict

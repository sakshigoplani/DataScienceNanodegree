############################################################################
#                                                                         ##
#                                                                         ##
#  Author:     Sakshi Haresh Goplani                                      ##
#  Project:    Date Engineering - Disaster Response Pipeline              ##
#  Email:      sakshigoplani9@gmail.com                                   ##
#                                                                         ##
############################################################################

""" Data Builder Utility

This script takes in a path where CSV files resides. It goes
over the data, cleans/processes it and saves the clean data in SQL Table.

Usage: python process_data.py <messages_filepath> <categories_filepath> 
                                <database_filepath> <tablename>

"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


# METADATA #
__version__ = 3.6
__author__ = 'Sakshi Haresh Goplani'
################


def argument_sanitization(messages_filepath, categories_filepath):
    """ Validate file paths

    Args:
        messages_filepath (string): Path to messages.csv
        categories_filepath (string): Path to categories.csv

    Returns:
        N/A

    """

    if not os.path.isfile(messages_filepath):
        logger.error("{} is not valid".format(messages_filepath))
    if not os.path.isfile(categories_filepath):
        logger.error("{} is not valid".format(categories_filepath))


def load_data(messages_filepath, categories_filepath):
    """ Read and Merge data

    Args:
        messages_filepath (string): Path to messages.csv
        categories_filepath (string): Path to categories.csv

    Returns:
        df (pandas dataframe): Merged messages and categories dataframe
    """

    # Confirm paths are valid and read in datasets into dataframe
    argument_sanitization(messages_filepath, categories_filepath)
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge Messages and Categories dataframes into single dataframe
    df = messages.merge(categories, how='outer', on='id')
    return df


def clean_data(df):
    """ Clean Data into relevant columns

    Args:
        df (pandas dataframe): Merged messages and categories dataframe

    Returns:
        df (pandas dataframe): Clean dataframe

    """

    # Create a dataframe of the individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # Select the first row of the categories dataframe
    row = categories.iloc[1, :]

    # Extract a list of new column names for categories
    category_colnames = list(map(lambda x: x.split('-')[0], row))
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1 numerals
    for column in categories:
        categories[column] = categories[column].str.split('-').str.get(1)
        categories[column] = pd.to_numeric(categories[column])

    # Replace categories column in df with new category columns
    df = df.drop('categories', axis=1)

    # Concat cleansed categories with df
    df = pd.concat([df, categories], axis=1)

    # Drop duplicates
    df.drop_duplicates(subset=['id'], keep='first', inplace=True)

    return df


def save_data(df, database_filename, tablename):
    """ Save data to SQL Database

    Args:
        df (pandas dataframe): Final cleansed dataframe
        database_filename (string): Name of the DB File
        tablename (string): Name of the table to create in DB

    Returns:
        N/A

    """

    engine = create_engine(database_filename)
    df.to_sql(tablename, engine, index=False)


def main():
    if len(sys.argv) == 5:

        messages_filepath, \
            categories_filepath, \
            database_filepath, \
            tablename = sys.argv[1:]

        logger.info(" Loading data...\n    MESSAGES: {}\n    CATEGORIES: {} "
                    .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        logger.info(" Cleaning data...")
        df = clean_data(df)

        logger.info(" Saving data...\n    DATABASE: {}\n    TABLE NAME: {} "
                    .format(database_filepath, tablename))
        save_data(df, database_filepath, tablename)

        logger.info(" Cleaned data saved to database!")

    else:
        logger.error(
            """
            Please provide the filepaths of the messages and categories
            datasets as the first and second argument respectively, as
            well as the filepath of the database to save the cleaned data
            to as the third argument. \n\nExample: python process_data.py
            disaster_messages.csv disaster_categories.csv
            DisasterResponse.db
            """
            )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    main()

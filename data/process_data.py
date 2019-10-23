import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np

def load_data(messages_filepath, categories_filepath):
    """
    Load the data from message file and category file from the given paths
    
    INPUT:
        messages_filepath: filepath to message csv file
        categories_filepath: filepath to category csv file

    OUTPUT:
        merged_df: merged result of the two csv files
    """
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)
    merged_df = messages_df.merge(categories_df, on=('id'))
    return merged_df


def clean_data(df):
    """
    Clean the data by splitting and converting categorical data to int, and removes duplicates
    
    INPUT:
        df: merged data frame from load_data function

    OUTPUT:
        cleaned_df: contains clean data
    """
    categories = df['categories'].str.split(";", expand = True)
    row = categories.iloc[0,:]
    category_colnames = row.str.split('-').str.get(0)
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    df.drop('categories', axis = 1, inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    cleaned_df = pd.concat([df, categories], axis=1)
    # drop duplicates
    cleaned_df.drop_duplicates(inplace = True)
    return cleaned_df


def save_data(df, database_filename):
    """
    Save the data to SQL database
    
    INPUT:
        df: clean data frame from clean_data function
        database_filename: filename of the SQL database file
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, index=False) 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
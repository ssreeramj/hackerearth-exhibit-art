import pandas as pd


def del_cols(dataframe):
    remove_cols = ['Customer Id', 'Artist Name', 'Customer Location']
    dataframe = dataframe.drop(remove_cols, axis=1)

    return dataframe
    

def get_date_diff(dataframe):
    dataframe['Scheduled Date'] = pd.to_datetime(dataframe['Scheduled Date'])
    dataframe['Delivery Date'] = pd.to_datetime(dataframe['Delivery Date'])
    
    dataframe['Diff'] = (dataframe['Scheduled Date'] - dataframe['Delivery Date']).apply(lambda x: x.days)
    dataframe = dataframe.drop(['Scheduled Date', 'Delivery Date'], axis=1)

    return dataframe 
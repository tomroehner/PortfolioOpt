import pandas as pd
import numpy as np
import yfinance as yf


def create_clean_csv(name_file_path, output_file_path, start_date, end_date):
    ticker_symbols = read_name_file(name_file_path)
    hist = fetch_data(ticker_symbols, start_date, end_date)
    hist = remove_invalid_columns(hist)
    hist = fill_missing_values(hist)
    hist.to_csv(output_file_path)


def read_name_file(name_file_path):
    """
    Reads a text file containing ticker symbols and returns them as a list

    :param name_file_path:
        Path of the input file. The input file should be a text file containing one ticker symbol per line.
    :return:
        list of ticker symbols
    """
    with open(name_file_path) as name_file:
        symbols = [line.rstrip() for line in name_file]
    return symbols


def fetch_data(ticker_symbols, start_date, end_date, data_type='Close'):
    """
    Retrieves daily stock data for stocks listed in the input 'name_file' via the 'yfinance' API for the
    time period defined by the 'start_date' and 'end_date' parameters

    :param ticker_symbols:
        List of ticker symbols. This parameter is ignored if the parameter name_file_path is provided
    :param start_date:
        Start date in the format of 'YYYY-MM-DD'
    :param end_date:
        End date in the format of 'YYYY-MM-DD'
    :param data_type:
        Type of stock price data
    :return:
        Pandas DataFrame containing the stock data for the specified selection of stocks
    """
    stocks = yf.Tickers(ticker_symbols)
    df = stocks.history(start=start_date, end=end_date)[data_type]
    return df


def remove_invalid_columns(df: pd.DataFrame):
    """
    Removes the columns of the DataFrame that contain a missing value in the first row

    :param df:
        The DataFrame to be modified
    :return:
        The modified DataFrame
    """
    no_cols_prev = len(df.columns.tolist())
    result = df.dropna(axis=1, subset=[df.index[0]])
    no_cols_after = len(result.columns.tolist())
    print(no_cols_prev - no_cols_after, 'Columns had to be removed')
    return result


def fill_missing_values(stock_prices: pd.DataFrame):
    """
    Fills in missing values in the stock price data by copying the value from the previous row

    :param stock_prices:
        Stock price data
    :return:
        The modified DataFrame
    """
    [rows, cols] = np.where(np.asarray(np.isnan(stock_prices)))
    for x in range(rows.size):
        i = rows[x]
        j = cols[x]
        if (i - 1) >= 0:
            stock_prices.iloc[i, j] = stock_prices.iloc[i - 1, j]
        else:
            raise ValueError('Dataframe should not have missing values in the first row')
    return stock_prices


def drop_other_columns(df: pd.DataFrame, headers: list):
    """
    Drops all columns of the DataFrame whose headers are not given via the 'headers' parameter

    :param df:
        DataFrame to be modified
    :param headers:
        List of column headers that should not be removed from the DataFrame
    :return:
        The modified DataFrame
    """
    cols = df.columns
    undesired_cols = [col for col in cols if col not in headers]
    return df.drop(columns=undesired_cols)

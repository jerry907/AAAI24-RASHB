import numpy as np
import pandas as pd

def to_csv(data, filename) -> None:
    """
    Converts a dataset from np.array to pd.DataFrame then writes to a .csv file

    Input:
        data (np.array): data to be written to csv
        filename (str): filename/path of csv to be written to
    
    Return:
        None
    """
    df = pd.DataFrame(data)
    df.to_csv(r"{}".format(filename), header=False, index=False)
    return None

def from_csv(filename) -> np.array:
    """
    Reads a dataset in .csv format and converts to a np.array

    Input:
        filename (str): filename/path of csv to be read
    
    Return:
        out (np.array): data in np.array format
    """
    df = pd.read_csv(r"{}".format(filename), header=None)
    out = df.to_numpy()
    return out

def subset_data(data, rows, columns) -> np.array:
    """
    Takes a subset of the given data

    If rows/columns are None, return all rows/columns

    Input:
        in_data (array like) or (pd.DataFrame): input data to take subset from
        rows (list of ints): index(es) of rows to subset from data
        columns (list of ints): index(es) of columns to subset from data
    
    Output:
        out (np.array): data as an array of arrays
    """
    # convert to dataframe
    if type(data) == pd.DataFrame:
        df = data
    else:
        df = pd.DataFrame(data)
    
    # check if selecting all rows/columns
    if rows is None:
        rows = range(df.shape[0])
    if columns is None:
        columns = range(df.shape[1])
    
    # get subset and convert to np.array
    out = df.iloc[rows, columns]
    out = out.to_numpy()

    return out
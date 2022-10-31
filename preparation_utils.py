import numpy as np
import pandas as pd

WEEK_DATA = [
    'tempmax',
    'temp',
    'humidity',
    'precip',
    'dew',
    'windspeed',
    'winddir',
    'pressure'
]

def GetNullListIndex(df: pd.DataFrame) -> list:
    """TODO"""
    indexes = []

    for index, row in df.iterrows():
        for col in WEEK_DATA:
            if True in np.isnan(row[col]):
                indexes.append(index)
                break
            else:
                continue

    return indexes

def GenerateFeatures(df: pd.DataFrame) -> pd.DataFrame:
    """TODO"""
    df = df.copy()
    for metric in WEEK_DATA:
        variance_col_name = f'{metric}_variance'
        df[variance_col_name] = df[metric].apply(lambda x: np.var(x))

        delta_col_name = f'{metric}_delta'
        df[delta_col_name] = df[metric].apply(lambda x: x[-1] - x[0])

    return df

def RemoveColumns(df: pd.DataFrame) -> pd.DataFrame:
    """TODO"""    
    # Create columns to remove
    drop_cols = WEEK_DATA
    drop_cols.extend(['index', 'DATE'])
    
    df = df.copy()
    
    return df.drop(labels=drop_cols, axis=1)
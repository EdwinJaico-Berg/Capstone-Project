import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    indexes = []

    for index, row in df.iterrows():
        for col in WEEK_DATA:
            if True in np.isnan(row[col]):
                indexes.append(index)
                break
            else:
                continue

    return indexes

def CalculateWeekVariance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for metric in WEEK_DATA:
        col_name = f'{metric}_variance'
        df[col_name] = df[metric].apply(lambda x: np.var(x))

    return df


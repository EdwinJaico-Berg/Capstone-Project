from tkinter.messagebox import NO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import boxcox

def LoadDf() -> pd.DataFrame:
    """TODO"""
    df_1 = pd.read_pickle('sample_data/30k_engineered.pkl')
    df_2 = pd.read_pickle('sample_data/large_fires_cleaned.pkl')

    # Drop date column
    df_1.drop('DATE', axis=1, inplace=True)

    # Concatenate the columns
    df = pd.concat([df_1, df_2])

    df.reset_index(drop=True, inplace=True)

    return df

def BasicCategoricalPreprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """TODO"""
    # Get dummy variables for the state column
    state_dummies = pd.get_dummies(df['STATE'], prefix='state')
    
    # Append the dummy variables and remove the state column
    df = pd.concat([df, state_dummies], axis=1)
    df.drop('STATE', axis=1, inplace=True)  

    return df

def BasicNumericPreprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """TODO"""
    df = df.copy()

    boxcox_cols = [col for col in df.columns
                if 'variance' in col]

    boxcox_cols.remove('precip_variance')

    for col in boxcox_cols:
        df[col] = boxcox(df[col])[0]

    return df, boxcox_cols


def HistogramSubplots(df: pd.DataFrame, columns: int=3, figsize: tuple=(30,30)) -> None:
    '''
    Plots the histograms for the columns in the given DataFrame.

    Parameters
    ----------
    df: pd.DataFrame from which the histograms are created.
    columns: number of columns the subplot will be split into.

    Returns
    -------
    None
    '''
    from math import ceil
    
    # Generate rows from columns
    rows = ceil(len(df.columns) / columns)

    # Create subplot grid
    plt.subplots(rows, columns, dpi=300, figsize=figsize)

    for index, col in enumerate(df.columns):
        # Calculate the position
        position = index + 1

        # Get subplot
        plt.subplot(rows, columns, position)

        # plot histogram
        sns.histplot(x=df[col])
        plt.axvline(x=df[col].mean(), label=f'Mean: {df[col].mean():.2f}', color='red')
        plt.axvline(x=df[col].median(), label=f'Median: {df[col].median():.2f}', color='green')
        plt.legend()

    plt.tight_layout()
    plt.show()

def ScatterSubplots(df: pd.DataFrame, target: str, columns: int=3):
    """TODO"""
    from math import ceil

    y = df[target]
    df = df.drop(target, axis=1)
    
    # Generate rows from columns
    rows = ceil(len(df.columns) / columns)

    # Create subplot grid
    plt.subplots(rows, columns, dpi=300, figsize=(30,30))

    for index, col in enumerate(df.columns):
        # Calculate the position
        position = index + 1

        # Get subplot
        plt.subplot(rows, columns, position)

        # plot histogram
        sns.scatterplot(x=df[col], y=y)

    plt.tight_layout()
    plt.show()


def PlotVarianceRatio(model: PCA, threshold: float=0.9) -> None:
    '''
    Given a PCA model, plot the cumulative sum of the variances and show the number of components that exceed the threshold value.

    Parameters
    ----------
    model: a PCA model from which the variances will be extracted.
    threshold: a float that tells us how much of the variance we want to capture with our components.

    Returns
    -------
    None
    '''
    # Calculate the cumulative sum from the explained variance
    cumulative_sum = np.cumsum(model.explained_variance_ratio_)
    x = list(range(1, len(cumulative_sum) + 1))

    index = np.where(cumulative_sum > threshold)[0][0]

    # Plot line plot and threshold
    plt.figure(dpi=300, figsize=(15, 5))
    sns.lineplot(x=x, y=cumulative_sum)
    plt.axhline(threshold, c='r', linestyle='--')
    plt.xlabel('Number of PCs')
    plt.ylabel('Cumulative Sum of Explained Variance')
    plt.plot([index + 1, index + 1], [0, cumulative_sum[index]], color='green', linestyle='--')
    plt.plot([0, index + 1], [cumulative_sum[index], cumulative_sum[index]], color='green', linestyle='--')
    plt.xticks([index + 1])
    plt.yticks(list(plt.yticks()[0]) + [cumulative_sum[index]])
    plt.xlim([0, len(cumulative_sum) + 1])
    plt.ylim([0, 1.1])
    plt.show()


def CustomTransform(df: pd.DataFrame, target: str) -> tuple:
    '''
    Function that returns a MinMax scaled and PCA transformed train, test, split

    Parameters
    ----------
    df: DataFrame from which the target and predictor variables are extracted.
    target: string of the target.

    Returns
    -------
    Tuple with X_train, X_test, y_train, y_test .
    '''
    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # Scale data
    mm_scaler = MinMaxScaler()
    X_train = mm_scaler.fit_transform(X_train)
    X_test = mm_scaler.transform(X_test)

    # Reduce dimensionality
    pca = PCA(n_components=0.9)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    return X_train, X_test, y_train, y_test

def train_validate_test(X, y, test_size: float=0.2, validation_size:float=0.3) -> tuple:
    '''
    TODO
    '''
    return NotImplementedError, 'Moved to model_utils'

def false_positive_rate(y_true, y_pred) -> float:
    '''
    Function that calculates the false positive rate from the actual and prediction class of the given data
    '''
    
    false_positives = (y_true == 0) & (y_pred == 1) 

    false_positive_number = false_positives.sum()

    true_negatives = (y_true == 0) & (y_pred == 0)  

    true_negative_number = true_negatives.sum()

    # Find the ratio of (FP) to (TN + FP)
    FPR = false_positive_number/(true_negative_number + false_positive_number)
    
    return FPR

def evaluate(string_list: str) -> list:
    ''''
    Function that converts the string representation of a list into a list, accounting for null values.
    We will be assuming that null values are written as nan, and that lists are separated by commas

    Parameters
    ----------
    string_list: string representation of a list

    Returns
    -------
    List of the values in the string representation of the list
    '''

    output = []

    for value in string_list[1:-1].split(','):
        try:
            output.append(eval(value))
        except NameError:
            output.append(np.nan)

    return output


def sample_rows(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Combines the head and tail of a DataFrame and returns the combination
    '''
    head = df.head(2)
    samples = df.sample(2)
    tail = df.tail(2)
    head_tail = pd.concat([head, samples, tail])
    return(head_tail)
    

def count_percentage_df(series: pd.Series) -> pd.DataFrame:
    """
    This function takes in a series and returns a DataFrame that shows the percentage
    and count of the values.
    """
    cp_dict = {
        'Count': [],
        'Percentage of Total': []
    }

    count_values = series.value_counts()
    percentage_values = series.value_counts(normalize = True)
    index = series.value_counts().index.values

    for count, percent in zip(count_values, percentage_values):
        cp_dict['Count'].append(count)
        cp_dict['Percentage of Total'].append(percent)
    
    df = pd.DataFrame(cp_dict, index=index)

    return df

def mean(variable_list: list) -> float:
    """
    Takes a list and returns the mean of all available values
    """
    count = 0
    sum = 0

    for value in variable_list:
        if np.isnan(value):
            continue
        else:
            count += 1
            sum += value
        
    
    if count > 0:
        mean = sum / count
        return mean
    else:
        return np.nan


def PlotCorrelationMatrix(df: pd.DataFrame) -> None:
    """TODO"""
    corr = df.corr()
    plt.figure(figsize=(20,10), dpi=300)
    matrix = np.triu(corr)
    sns.heatmap(corr, annot=True, mask=matrix, vmin=-1, vmax=1)
    plt.show()

def PlotFireSizeDistribution(df: pd.DataFrame) -> None:
    plt.subplots(1, 2, figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.title('Fire Size Distribution')
    sns.boxplot(x=df['FIRE_SIZE'])
    plt.axvline(x=df['FIRE_SIZE'].mean(), color='red', label=f'Mean: {df["FIRE_SIZE"].mean():.1f}')
    plt.xlabel('Fire Size (Acres)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title('Fire Size Distribution')
    plt.axvline(x=df['FIRE_SIZE'].median(), color='red', label=f'Median: {df["FIRE_SIZE"].median():.1f}')
    sns.boxplot(x=df['FIRE_SIZE'], showfliers=False)
    plt.xlabel('Fire Size (Acres)')
    plt.legend()

    plt.show()

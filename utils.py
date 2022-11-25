import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import boxcox

def BasicEda(df: pd.DataFrame, name: str, components: list = None) -> None:
    """
    Prints the basic summary of the DataFrame, including the shape, percentage
    of null and duplicate rows, and the numeric/categorical split.

    Parameters
    ----------
    df: The DataFrame on which the EDA will be performed.
    name: The name that we want to give to the given DataFrame.
    components: {None, 'shape', 'null', 'duplicates', 'columns', 'dtypes'}, default None. 
    List that determines which components of the EDA to print. Mulitple values can be put in the list. 
    - ``shape``: Prints the rows and columns of the DataFrame.
    - ``null``: Prints the null row count and percentage.
    - ``duplicates``: Prints the duplicate row count and percentage.
    - ``columns``: Prints the dtype of the columns in the DataFrame.
    - ``dtypes``: Prints the categorical and numerical column count.

    Returns
    -------
    None
    """
    title = name.upper()
    rows = df.shape[0]
    columns = df.shape[1]
    null_rows = len(df[df.isna().all(axis=1)])
    percentage_null_rows = null_rows / rows
    types = df.dtypes
    num_cols = len(df.select_dtypes('number').columns)
    cat_cols = len(df.select_dtypes('object').columns)

    print(f'{title}', '-' * len(title), sep='\n', end='\n\n')

    if (not components) or ('shape' in components):
        print(f'Rows: {rows}    Columns: {columns}', end='\n\n')
    if (not components) or ('null' in components):
        print(f'Total null rows: {null_rows}')
        print(f'Percentage null rows: {percentage_null_rows: .3f}%', end='\n\n')
    if (not components) or ('duplicates' in components):
        duplicate_rows = df.duplicated().sum()
        percentage_duplicate_rows = duplicate_rows / rows
        print(f'Total duplicate rows: {duplicate_rows}')
        print(f'Percentage duplicate rows: {percentage_duplicate_rows: .3f}%', end='\n\n')
    if (not components) or ('columns' in components):
        print(types, end='\n\n')
    if (not components) or ('dtypes' in components):
        print(f'Number of categorical columns: {cat_cols}')
        print(f'Number of numeric columns: {num_cols}')

def LoadDf() -> pd.DataFrame:
    """
    Function that loads the DataFrame from the samples created.

    Parameters
    ----------
    None.

    Returns
    -------
    pd.DataFrame that contains the approximately 30,000 wildfire observations with weather and
    emissions data.
    """
    df_1 = pd.read_pickle('sample_data/30k_engineered.pkl')
    df_2 = pd.read_pickle('sample_data/large_fires_cleaned.pkl')

    # Drop date column
    df_1.drop('DATE', axis=1, inplace=True)

    # Concatenate the columns
    df = pd.concat([df_1, df_2])

    df.reset_index(drop=True, inplace=True)

    return df

def BasicCategoricalPreprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame and does the basic preprocessing for the categorical columns, which 
    includes the binarisation of the `STATE` column in the DataFrame.

    Parameters
    ----------
    df: pd.DataFrame containing the wildfire data. 

    Returns
    -------
    pd.DataFrame with the original columns and the binarised `STATE` column.
    """
    # Get dummy variables for the state column
    state_dummies = pd.get_dummies(df['STATE'], prefix='state')
    
    # Append the dummy variables and remove the state column
    df = pd.concat([df, state_dummies], axis=1)
    df.drop('STATE', axis=1, inplace=True)  

    return df

def BasicNumericPreprocessing(df: pd.DataFrame) -> tuple:
    """
    Function that takes a DataFrame and conducts basic preprocessing for numerical columns. 
    This consists of transforming variance columns using a BoxCox transformation.

    Parameters
    ----------
    df: pd.DataFrame containing numerical columns.

    Returns
    -------
    A tuple consisting of the formatted DataFrame, and the list of columns that were transformed.
    """
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
    None.
    '''
    from math import ceil
    
    # Generate rows from columns
    rows = ceil(len(df.columns) / columns)

    # Create subplot grid
    plt.subplots(rows, columns, dpi=300, figsize=figsize)

    for position, col in enumerate(df.columns, start=1):
        # Get subplot
        plt.subplot(rows, columns, position)

        # plot histogram
        sns.histplot(x=df[col])
        plt.axvline(x=df[col].mean(), label=f'Mean: {df[col].mean():.2f}', color='red')
        plt.axvline(x=df[col].median(), label=f'Median: {df[col].median():.2f}', color='green')
        plt.legend()

    plt.tight_layout()
    plt.show()

def ScatterSubplots(df: pd.DataFrame, target: str, columns: int=3) -> None:
    """
    Function that plots the scatter plots of a target against the other features in a given 
    DataFrame within a subplot. 

    Parameters
    ----------
    df: The pd.DataFrame from which the features and target are taken,
    target: The target variable which the scatter plots will be plotted against.
    columns: default 3. Number of columns that the subplot will have.

    Returns
    -------
    None
    """
    from math import ceil

    y = df[target]
    df = df.drop(target, axis=1)
    
    # Generate rows from columns
    rows = ceil(len(df.columns) / columns)

    # Create subplot grid
    plt.subplots(rows, columns, dpi=300, figsize=(30,30))

    for position, col in enumerate(df.columns, start=1):
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
    threshold: default 0.9. A float that tells us how much of the variance we want to capture with our components.

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

def FPR(y_true: np.array, y_pred: np.array) -> float:
    '''
    Function that calculates the false positive rate from the actual and prediction class of the given data.

    Parameters
    ----------
    y_true: np.array or pd.Series that contains the actual classes for the corresponding observations.
    y_pred: np.array or pd.Series of the predicted values generated by a classification model.
    
    Returns
    -------
    The false positive rate as a float.
    '''
    
    false_positives = (y_true == 0) & (y_pred == 1) 

    false_positive_number = false_positives.sum()

    true_negatives = (y_true == 0) & (y_pred == 0)  

    true_negative_number = true_negatives.sum()

    # Find the ratio of (FP) to (TN + FP)
    fpr = false_positive_number/(true_negative_number + false_positive_number)
    
    return fpr

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
    Combines the two rows from the head and tail of a DataFrame along with two random rows and returns the combination.

    Parameters
    ----------
    df: pd.DataFrame which we want to generate the samples from.

    Returns
    -------
    pd.DataFrame containing the sample rows generated.
    '''
    head = df.head(2)
    samples = df.sample(2)
    tail = df.tail(2)
    head_tail = pd.concat([head, samples, tail])
    return(head_tail)
    

def count_percentage_df(series: pd.Series) -> pd.DataFrame:
    """
    This function takes in a pd.Series of categorical variables and returns a DataFrame that shows the percentage
    and count of the values.

    Parameters
    ----------
    series: a pd.Series listing categorical variables.

    Returns
    -------
    pd.DataFrame of the count and percentage representation of the variables in the list.
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
    Takes a list and returns the mean of all available values.

    Parameters
    ----------
    variable_list: a list of floats and null values.

    Returns
    -------
    The mean as a float. If the list is made of only null values, then np.nan is returned.
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


def PlotCorrelationMatrix(df: pd.DataFrame, annot: bool=True) -> None:
    """
    Function that takes a pd.DataFrame and plots the correlation matrix.

    Parameters
    ----------
    df: pd.DataFrame from which the numeric variables will be taken.

    Returns
    -------
    None.
    """
    corr = df.corr()
    plt.figure(figsize=(20,10), dpi=300)
    matrix = np.triu(corr)
    sns.heatmap(corr, annot=annot, mask=matrix, vmin=-1, vmax=1)
    plt.show()

def PlotFireSizeDistribution(df: pd.DataFrame) -> None:
    """
    Function that plots a boxplot of the wildfire sizes in a given DataFrame. Two plots will be created
    and presented within a subplot. The first will show the boxplot including the outliers along with a 
    line showing the mean, while the second will omit outliers and plot a line for the median.

    Parameters
    ----------
    df: pd.DataFrame from which the fire sizes will be taken.

    Returns
    -------
    None
    """
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

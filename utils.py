from tkinter.messagebox import NO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def LoadDf() -> pd.DataFrame:
    """TODO"""
    df_1 = pd.read_pickle('sample_data/30k_engineered.pkl')
    df_2 = pd.read_pickle('sample_data/large_fires_cleaned.pkl')

    # Drop date column
    df_1.drop('DATE', axis=1, inplace=True)

    # Concatenate the columns
    df = pd.concat([df_1, df_2])

    return df

def BasicPreprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """TODO"""
    # Get dummy variables for the state column
    state_dummies = pd.get_dummies(df['STATE'], prefix='state')
    
    # Append the dummy variables and remove the state column
    df = pd.concat([df, state_dummies], axis=1)
    df.drop('STATE', axis=1, inplace=True)  

    return df

def PlotBoundaries(model, X, Y, figsize=(8, 6)):
    '''
    Helper function that plots the decision boundaries of a model and data (X,Y)
    Code modified from: https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html
    '''

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=figsize)
    plt.contourf(xx, yy, Z, alpha=0.4)

    # Plot
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=20, edgecolor='k')
    plt.show()

def PlotEnsembleBoundaries(ensembles, X, Y, shape, figsize=(10, 7)):
    '''
    Helper function to plot the boundaries of ensemble methods.
    Code modified from: https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_iris.html
    '''
    fig, axes = plt.subplots(shape[0], shape[1], figsize=figsize)
    for i, (ax, model) in enumerate(zip(axes.ravel(), ensembles)):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.4)

        # Plot
        ax.scatter(X[:, 0], X[:, 1], c=Y, s=20, edgecolor='k')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

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
        plt.legend(fancybox=True, shadow=True)

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
    plt.figure(dpi=300)
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

def PlotTrainTest(x_vals: list, x_label: str, train_scores: list, test_scores: list, validation: bool=False) -> None:
    '''
    Function that plots the train and test/validation accuracies.

    Parameters
    ----------
    x_vals: list of the x-axis values for the plot.
    x_label: string for the x-axis label.
    train_scores: list of the train scores.
    test_scores: list of train or validation scores.
    validation: bool for whether we are plotting the validation or test set. Default value is False.

    Returns
    -------
    None
    '''
    # Create line label
    line_label = 'Validation Score' if validation else 'Test Score'
    
    # Create plot
    plt.figure(figsize=(15, 5), dpi=300)
    plt.plot(x_vals, train_scores,label="Train Score",marker='.')
    plt.plot(x_vals, test_scores,label=line_label,marker='.')
    plt.xlabel(x_label)
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show();

def GeneateXy(df: pd.DataFrame, target: str, sample: float=None) -> tuple:
    '''
    Create the X and y variables from a given data frame.

    Parameters
    ----------
    df: DataFrame from which the X and y variables are extracted.
    target: string of the column which we will be considering the target.
    sample: float value that determines how large a sample we are taking. If not given, the entire DataFrame will be considered.

    Returns
    -------
    Returns a tuple containing the X variables and the y target. These will be a pandas DataFrame or Series.
    '''
    if sample:
        df = df.sample(frac=sample)

    X = df.drop(target, axis=1)
    y = df[target]

    return X, y

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
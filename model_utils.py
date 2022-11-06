import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer


def GenerateXy(df: pd.DataFrame) -> tuple:
    """
    Creates the X and y variables for a classification model from a given DataFrame.

    Parameters
    ----------
    df: pd.DataFrame.

    Returns
    -------
    A (X, y) tuple.
    """
    X = df.drop(['FIRE_SIZE_CLASS', 'FIRE_SIZE'], axis=1)
    y = df['FIRE_SIZE_CLASS']

    return X, y

def train_validate_test(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Function that creates a train, validation, and a test set. The X_remainder to y_remainder
    split is set to 0.2, whilst the X_train to X_validation is set to 0.3.

    Parameters
    ----------
    X: pd.DataFrame that contains the predictor features.
    y: pd.Series that contains the target variable.

    Returns
    -------
    tuple containing the X_trian, X_validation, X_test, y_train, y_validation, y_test variables.
    """
    # Startify if the target is categorical
    stratify_y = y if y.dtype == 'O' else None

    # Create remainder and test
    X_rem, X_test, y_rem, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=1, stratify=stratify_y)

    # Prevent stratification at this level if not categorical
    stratify_rem = y_rem if stratify_y else None

    # Create train and validation
    X_train, X_validation, y_train, y_validation = \
    train_test_split(X_rem, y_rem, test_size=0.3, random_state=1, stratify=y_rem)
    
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def PlotConfusionMatrix(model, X_test, y_test) -> None:
    """
    Function that plots the confusion matrix from a given model and X_test and y_test and prints
    the classification report underneath.

    Parameters
    ----------
    model: the classification model that has been created
    X_test: the predictor variables.
    y_test: the actual classification of the observations.

    Returns
    -------
    None
    """
    # Make classifications based on the test features, and assign the classifications to a variable
    y_pred = model.predict(X_test)

    predictions = y_test.unique()
    predictions.sort()

    # Build the confusion matrix as a dataframe
    confusion_df = pd.DataFrame(confusion_matrix(y_test, y_pred))
    confusion_df.index = [i for i in predictions]
    confusion_df.columns = [i for i in predictions]

    # Heatmap of the above
    plt.figure(figsize=(12, 8), dpi=300)
    sns.heatmap(confusion_df, annot=True, fmt='d', cmap='OrRd') # Passing in fmt='d' prevents the counts from being displayed in scientific notation
    plt.title('Confusion Heatmap')
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.show()

    print(classification_report(y_test, y_pred))

def CreateTransformedTTS(X: pd.DataFrame, y: pd.Series, validation:bool=False) -> tuple:
    """
    Creates a column transformer and fit_transforms the train, test and optionally the validation set.

    Parameters
    ----------
    X: pd.DataFrame containing the predictor variables
    y: pd.Series containing the target variable

    Returns
    -------
    tuple of the transformed train, validation, and test set along with the y equivalents.
    """
    # Create train and test set
    if validation:
        # Create train, validation, and test
        X_train, X_validation, X_test, y_train, y_validation, y_test = train_validate_test(X, y)
        # Create column transformer
        col_transformer = CreateColumnTransformer(X_train)

        # Fit to X_train and transform
        X_train_transformed = col_transformer.fit_transform(X_train)
        X_validation_transformed = col_transformer.transform(X_validation)
        X_test_transformed = col_transformer.transform(X_test)

        return X_train_transformed, X_validation_transformed, X_test_transformed, y_train, y_validation, y_test

    else:
        strat = y if y.dtype == 'O' else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=strat, test_size=0.2, random_state=1)
    
        # Create column transformer
        col_transformer = CreateColumnTransformer(X_train)

        # Fit to X_train and transform
        X_train_transformed = col_transformer.fit_transform(X_train)
        X_test_transformed = col_transformer.transform(X_test)

        return X_train_transformed, X_test_transformed, y_train, y_test

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

def PlotTrainTest(x_vals: list, x_label: str, train_scores: list, test_scores: list, validation: bool=False, log: bool=False) -> None:
    '''
    Function that plots the train and test/validation accuracies in a line graph.

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
    if log:
        plt.xscale('log')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show();

def CreateColumnTransformer(X: pd.DataFrame) -> ColumnTransformer:
    """
    Function that creates the column transformer for the numerical columns in the DataFrame.

    Parameters
    ----------
    X: pd.DataFrame to which the column transformer will be applied to.

    Returns
    -------
    A column transformer that can be fit to a DataFrame of features.
    """
    # Create the scaling columns
    robust_cols = ['precip_variance', 'ch4', 'n2o', 'co2']
    minmax_cols = ['FIRE_YEAR', 'DISCOVERY_DOY', 'LATITUDE', 'LONGITUDE']
    state_cols = [col for col in X.columns
                if 'state' in col]
    ss_cols = [col for col in X.columns
            if col not in robust_cols
            if col not in minmax_cols
            if col not in state_cols]
    
    # Create column transformation list
    col_transforms = [('standard scale', StandardScaler(), ss_cols),
                    ('minmax scale', MinMaxScaler(), minmax_cols), 
                    ('robust scale', RobustScaler(), robust_cols)]

    # Create the column transformer
    col_transformer = ColumnTransformer(col_transforms, remainder='passthrough')

    return col_transformer

def PlotCoefficients(model, X) -> None:
    """
    Function that plots the features that have the top five positive and negative coefficients
    in a model.

    Parameters
    ----------
    model: any sklearn regression or classification model.
    X: the DataFrame which the model was fit to.

    Returns
    -------
    None 
    """
    # Place coefficients in sorted DataFrame
    coefficients_df = sorted_coefficients_df(model, X)

    # Initialise subplot
    plt.subplots(1, 2, figsize=(15, 5))
    
    # Create positive coefficient subplot
    plt.subplot(1, 2, 1)
    plt.title('Top 5 Positive Linear Regression Coefficients')
    sns.barplot(x='coef', y=coefficients_df.head().index, data=coefficients_df.head())
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature Name')

    # Create negative coefficient subplot
    plt.subplot(1, 2, 2)
    plt.title('Top 5 Negative Linear Regression Coefficients')
    sns.barplot(x='coef', y=coefficients_df.tail().index, data=coefficients_df.tail())
    plt.xlabel('Coefficient Value')
    plt.show()

def sorted_coefficients_df(model, X):
    try:
        # Get the coefficients
        coefficients = model.coef_
        coefficients_df = pd.DataFrame(data={'coef':coefficients}, index=X.columns).sort_values(by='coef', ascending=False)
    except ValueError:
        # Get the coefficients
        coefficients = model.coef_[0]
        coefficients_df = pd.DataFrame(data={'coef':coefficients}, index=X.columns).sort_values(by='coef', ascending=False)

    return coefficients_df
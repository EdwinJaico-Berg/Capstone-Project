import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score

def ScoreModel(model, X_test, y_test) -> None:
    """
    A simple function that combines the various evaluation metrics for a linear regression
    and prints them.

    Parameters
    ----------
    model: linear regression model that is being evaluated.
    X_test: pd.DataFrame containing the test data set.
    y_test: pd.Series containing the dependent variable.

    Returns
    -------
    None
    """
    y_pred = model.predict(X_test)

    print('Model Evaluation')
    print(f'R-squared: {model.score(X_test, y_test)}')
    print(f'RMSE:      {mean_squared_error(y_test, y_pred)}')
    print(f'MAE:       {mean_absolute_error(y_test, y_pred)}')

def PlotAlphaRegression(X_train: pd.DataFrame, y_train: pd.Series, model, alphas: list) -> None:
    """
    Function that plots the cross-validation scores against the alpha value of a Ridge or Lasso regression model.

    Parameters
    ----------
    X_train: pd.DataFrame that contains the predictor variables
    y_train: pd.Series containing the dependent variable for the given training observations
    model: {'Lasso', 'Ridge'}. Variable that determines which regression model is being run.
    alphas: list containing the alpha values to be tested.

    Returns
    -------
    None
    """
    cross_val_scores = []

    for alpha in alphas:
        print(f'Model {alpha}/{max(alphas)}', end='\r')
        model_ = model(alpha=alpha)
        cv_score = np.mean(cross_val_score(model_, X_train, y_train, cv = 5))
        cross_val_scores.append(cv_score)

    plt.figure(figsize=(15, 5))
    plt.title('Regression')
    sns.lineplot(x=alphas, y=cross_val_scores, marker='o', label='Cross Validation Score')
    plt.xlabel('Alpha')
    plt.ylabel('Cross-Validation Score')
    plt.show()

def ridge_lasso_evaluation(Model, X: pd.DataFrame, y: pd.Series, alphas: list) -> pd.DataFrame:
    '''
    Takes in a list of alphas and runs a series of Lasso or Ridge regressions iterating through the alphas. 
    The coefficients for each of the fitted models is then appended to a DataFrame which is returned once all
    the alphas have been modeled.

    Parameters
    ----------
    Model: {'Lasso', 'Ridge'}. The model that is being evaluated.
    X: pd.DataFrame containing the predictor variables
    y: pd.Series containing the target variable
    alphas: list of alphas that is being evaluated

    Returns
    -------
    pd.DataFrame containing the coefficient values for models with different alpha values.
    '''
    # Create an empty data frame
    df = pd.DataFrame(index=X.columns)
    
    # For each alpha value in the list of alpha values,
    for alpha in alphas:
        print(f'Fitting model {alpha}/{max(alphas)}', end='\r')

        # Create a lasso regression with that alpha value,
        model = Model(alpha=alpha)
        
        # Fit the lasso regression
        model.fit(X, y)
        
        # Create a column name for that alpha value
        column_name = f'Alpha = {alpha}'

        # Create a column of coefficient values
        df[column_name] = model.coef_
        
    # Return the datafram    
    return df

def PlotAlphaHeatmap(model, X_train, y_train, alphas, figsize=(30,30)) -> None:
    """
    Plots a heat map of the coefficients.

    Parameters
    ----------
    model: {'Lasso', 'Ridge'}. The model that is being evaluated.
    X_train: pd.DataFrame of the values the model is being trained on.
    y_train: pd.Series containing the target of the given observations.
    alphas: list of alphas being evaluated.
    figsize: tuple containing the dimensions for the plot. default = (30,30)

    Returns
    -------
    None
    """
    df = ridge_lasso_evaluation(model, X_train, y_train, alphas)
    plt.figure(figsize=figsize)
    sns.heatmap(df)
    plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score

def ScoreModel(model, X_test, y_test) -> None:
    """TODO"""
    y_pred = model.predict(X_test)

    print('Model Evaluation')
    print(f'R-squared: {model.score(X_test, y_test)}')
    print(f'RMSE:      {mean_squared_error(y_test, y_pred)}')
    print(f'MAE:       {mean_absolute_error(y_test, y_pred)}')

def PlotAlphaRegression(X_train, y_train, model, alphas: list) -> None:
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
    Takes in a list of alphas. Outputs a dataframe containing the coefficients of lasso regressions from each alpha.
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

def PlotAlphaHeatmap(model, X_train, y_train, alphas, figsize=(30,30)):
    df = ridge_lasso_evaluation(model, X_train, y_train, alphas)
    plt.figure(figsize=figsize)
    sns.heatmap(df)
    plt.show()
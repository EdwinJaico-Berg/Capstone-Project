import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def feature_importance_df(model, X) -> pd.DataFrame:
    """
    Creates a DataFrame that lists the variables and their importance in descending order.

    Parameters
    ----------
    model: The decision tree model that is being evaluated
    X: the DataFrame that contains the features of the model.

    Returns:
    --------
    DataFrame 
    """
    importances = model.tree_.compute_feature_importances(normalize=False)
    importances_df = pd.DataFrame({'Variable': X.columns,
                                'Importance': importances})
    return importances_df.sort_values(by='Importance', ascending=False)

def plot_feature_importances(model, X) -> None:
    """
    Plots the top 10 most important features of a model.

    Parameters
    ----------
    model: the decision tree model being evaluated.
    X: the DataFrame that contains the features of the model.

    Returns
    -------
    None
    """
    df = feature_importance_df(model, X)
    
    plt.figure(figsize=(15, 5))
    plt.title('Top 10 Features')
    sns.barplot(x='Importance', y='Variable', data=df.head(10))
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Name')
    plt.show()
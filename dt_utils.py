import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def feature_importance_df(model, X):
    importances = model.tree_.compute_feature_importances(normalize=False)
    importances_df = pd.DataFrame({'Variable': X.columns,
                                'Importance': importances})
    return importances_df.sort_values(by='Importance', ascending=False)

def plot_feature_importances(model, X):
    df = feature_importance_df(model, X)
    
    plt.figure(figsize=(15, 5))
    plt.title('Top 10 Positive Linear Regression Coefficients')
    sns.barplot(x='Importance', y='Variable', data=df.head(10))
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature Name')
    plt.show()
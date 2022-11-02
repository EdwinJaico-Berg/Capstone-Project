import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer


def GenerateXy(df):
    X = df.drop(['FIRE_SIZE_CLASS', 'FIRE_SIZE'], axis=1)
    y = df['FIRE_SIZE_CLASS']

    return X, y

def train_validate_test(X, y):
    # Create remainder and test
    X_rem, X_test, y_rem, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # Create train and validation
    X_train, X_validation, y_train, y_validation = \
    train_test_split(X_rem, y_rem, test_size=0.3, random_state=1, stratify=y_rem)
    
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def PlotConfusionMatrix(model, X_test, y_test) -> None:
    # Make classifications based on the test features, and assign the classifications to a variable
    y_pred = model.predict(X_test)

    predictions = y_test.unique()
    predictions.sort()

    # Build the confusion matrix as a dataframe
    confusion_df = pd.DataFrame(confusion_matrix(y_test, y_pred))
    confusion_df.index = [f'Actually {i}' for i in predictions]
    confusion_df.columns = [f'Predicted {i}' for i in predictions]
    # Heatmap of the above
    plt.figure(figsize=(12, 8))
    sns.heatmap(confusion_df, annot=True, fmt='d', cmap='OrRd') # Passing in fmt='d' prevents the counts from being displayed in scientific notation
    plt.xticks(rotation=45)
    plt.title('Confusion Heatmap')
    plt.show()

    print(classification_report(y_test, y_pred))

def CreateTransformedColumns(X, y):
    """TODO"""
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
    from sklearn.compose import ColumnTransformer
    
    # Create remainder and test
    X_rem, X_test, y_rem, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # Create train and validation
    X_train, X_validation, y_train, y_validation = \
    train_test_split(X_rem, y_rem, test_size=0.3, random_state=1, stratify=y_rem)
    
    robust_cols = ['precip_variance', 'ch4', 'n2o', 'co2']
    minmax_cols = ['FIRE_YEAR', 'DISCOVERY_DOY', 'LATITUDE', 'LONGITUDE']
    state_cols = [col for col in X_train.columns
                  if 'state' in col]
    ss_cols = [col for col in X_train.columns
               if col not in robust_cols
               if col not in minmax_cols
               if col not in state_cols]


    # Create column transformation list
    col_transforms = [('standard scale', StandardScaler(), ss_cols),
                    ('minmax scale', MinMaxScaler(), minmax_cols), 
                    ('robust scale', RobustScaler(), robust_cols)]

    # Create the column transformer
    col_transformer = ColumnTransformer(col_transforms, remainder='passthrough')

    # Fit to X_train
    col_transformer.fit(X_train)

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
    if log:
        plt.xscale('log')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show();

def CreateColumnTransformer(X: pd.DataFrame) -> ColumnTransformer:
    """TODO"""
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
    # The coefficient, notice it returns an array with one spot for each feature
    coefficients = model.coef_

    coefficients_df = pd.DataFrame(data={'coef':coefficients}, index=X.columns).sort_values(by='coef', ascending=False)

    plt.subplots(1, 2, figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.title('Top 5 Positive Linear Regression Coefficients')
    sns.barplot(x='coef', y=coefficients_df.head().index, data=coefficients_df.head())
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature Name')

    plt.subplot(1, 2, 2)
    plt.title('Top 5 Negative Linear Regression Coefficients')
    sns.barplot(x='coef', y=coefficients_df.tail().index, data=coefficients_df.tail())
    plt.xlabel('Coefficient Value')
    plt.show()
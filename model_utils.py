import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge


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

    # Build the confusion matrix as a dataframe
    confusion_df = pd.DataFrame(confusion_matrix(y_test, y_pred))
    confusion_df.index = [f'Actually {i}' for i in range(1, 8)]
    confusion_df.columns = [f'Predicted {i}' for i in range(1, 8)]

    # Heatmap of the above
    plt.figure(figsize=(12, 8), dpi=300)
    sns.heatmap(confusion_df, annot=True, fmt='d', cmap='OrRd') # Passing in fmt='d' prevents the counts from being displayed in scientific notation
    plt.xticks(rotation=45)
    plt.title('Confusion Heatmap')
    plt.show()

    print(classification_report(y_test, y_pred))

# Create a function called lasso,
def ridge_lasso_evaluation(Model, X: pd.DataFrame, y: pd.Series, alphas: list) -> pd.DataFrame:
    '''
    Takes in a list of alphas. Outputs a dataframe containing the coefficients of lasso regressions from each alpha.
    '''
    # Create an empty data frame
    df = pd.DataFrame(index=X.columns)
    
    # For each alpha value in the list of alpha values,
    for alpha in alphas:
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
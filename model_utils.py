import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

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
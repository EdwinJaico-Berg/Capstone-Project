import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from model_utils import GenerateXy

def GenerateSMOTE(df: pd.DataFrame, validation: bool = False) -> tuple:
    """TODO"""
    X, y = GenerateXy(df)
    
    X_rem, X_test, y_rem, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=1)

    if validation:
        X_train, X_validation, y_train, y_validation = train_test_split(X_rem, y_rem, stratify=y_rem, test_size=0.3, random_state=1)
        
        X_train_sm, y_train_sm = SMOTE(random_state=1).fit_resample(X_train, y_train)

        return X_train_sm, X_validation, X_test, y_train_sm, y_validation, y_test

    X_rem_sm, y_rem_sm  = SMOTE(random_state=1).fit_resample(X_rem, y_rem)

    return X_rem_sm, X_test, y_rem_sm, y_test
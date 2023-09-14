from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

import numpy as np
import pandas as pd


def catboost_kfold_feature_importance(X_train, y_train, cat_features=None, n_splits=5, random_state=5):
    """
    Perform K-Fold cross-validation with CatBoost and calculate feature importances.

    Args:
    - X_train: DataFrame, training features.
    - y_train: Series, training target.
    - cat_features: List of categorical feature names (default is None).
    - n_splits: Number of K-Fold splits (default is 5).
    - random_state: Random seed for reproducibility (default is 5).

    Returns:
    - fi_df: DataFrame, feature importances with fold-wise and average values.
    """    
    # Initialize DataFrame to store feature importances
    fi_df = pd.DataFrame({'Feature': X_train.columns})

    # Initialize K-Fold cross-validator
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Create empty array to store fold AUC scores
    fold_scores = np.zeros(n_splits)

    # Initialize CatBoost model
    model = CatBoostClassifier(random_state=random_state, cat_features=cat_features, verbose=False)

    # Perform K-Fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_train_fold, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Fit the CatBoost model
        model.fit(X_train_fold, y_train_fold, eval_set=(X_val, y_val), verbose=100, early_stopping_rounds=100)

        # Calculate fold AUC score
        y_pred_val = model.predict_proba(X_val)[:, 1]
        fold_score = roc_auc_score(y_val, y_pred_val)
        fold_scores[fold] = fold_score

        # Record feature importances for this fold
        feature_importance = model.get_feature_importance()
        fi_df[f'Fold_{fold + 1}'] = feature_importance

    # Calculate and append average feature importance
    fi_df['Average'] = fi_df.iloc[:, 1:].mean(axis=1)

    fi_df.to_csv('feature_importance.csv', index=False)

    return fi_df

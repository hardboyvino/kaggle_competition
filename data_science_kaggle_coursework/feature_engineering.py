import pandas as pd
import numpy as np

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import PolynomialFeatures

feature_engineering_options = [

]

def create_missingness_features(source_df, target_df, features):
    """
    Create different types of features based on the presence of holes
    (missing data) in the source_df
    """
    target_df = target_df.to_frame()
    target_df["no_null"] = source_df.loc[:, features].isna().sum(axis=1)
    target_df["null_mean"] = source_df[features].isna().mean(axis=1)
    target_df["null_std"] = source_df[features].isna().std(axis=1)
    target_df["null_var"] = source_df[features].isna().var(axis=1)
    target_df["null_skew"] = source_df[features].isna().skew(axis=1)
    target_df["null_kurt"] = source_df[features].isna().kurt(axis=1)
    target_df["null_sem"] = source_df[features].isna().sem(axis=1)

    # Frequency mapping for the number of nulls per row
    freq_map_dict = source_df.isna().sum(axis=1).value_counts(dropna=False)
    target_df["null_sum_freq"] = (
        source_df.isna().sum(axis=1).map(freq_map_dict).astype(np.int32)
    )

    return target_df

def generate_domain_features(df, df_features):
    """
    Generate domain-specific features as ratios between the given columns in a DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the original features.
    df_features : list
        A list of feature names to be used for generating domain-specific features.

    Returns:
    --------
    pandas.DataFrame
        The DataFrame with the domain-specific features added.
    """
    # Get the list of features to create domain-specific features
    features = [col for col in df.columns if col in df_features]
    new_features = []
    
    df_new_features = pd.DataFrame()

    # Iterate through the features and create domain-specific features as ratios
    for i in range(len(features)):
        for j in range(len(features)):
            # Check if the features are different
            if i != j:
                # Generate a new feature name for the domain-specific feature
                new_feature_name = f"{features[i]}_{features[j]}_ratio"
                
                # Create the domain-specific feature by dividing the values of the two original features
                # If the denominator is 0, use a small value (1e-6) to avoid division by zero
                df_new_features[new_feature_name] = df[features[i]] / np.where(df[features[j]] == 0, 1e-6, df[features[j]])
                
                # Add the new feature name to the list of new features
                new_features.append(new_feature_name)
    
    df_combined = pd.concat([df, df_new_features], axis=1)

    return df, df_new_features, df_combined


def generate_interactive_features(df, df_features):
    """
    Generate interaction features between the given columns in a DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the original features.
    df_features : list
        A list of feature names to be used for generating interaction features.

    Returns:
    --------
    pandas.DataFrame
        The DataFrame with the interaction features added.
    """
    df_interactive = pd.DataFrame()

    # Get the list of features to create interaction terms
    features = [col for col in df.columns if col in df_features]
    new_features = []

    # Iterate through the features and create interaction terms
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            # Generate a new feature name for the interaction term
            new_feature_name = f"{features[i]}_{features[j]}"
            
            # Create the interaction feature by multiplying the values of the two original features
            df_interactive[new_feature_name] = df[features[i]] * df[features[j]]
            
            # Add the new feature name to the list of new features
            new_features.append(new_feature_name)

    combined_df = pd.concat([df, df_interactive], axis=1)
    
    return df, df_interactive, combined_df


def generate_polynomial_features(df, degree, df_features):
    """
    Generate polynomial features for the specified columns in a DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the original features.
    degree : int
        The degree of the polynomial features to generate.
    df_features : list
        A list of feature names to be used for generating polynomial features.

    Returns:
    --------
    pandas.DataFrame
        The DataFrame with the polynomial features added.
    """
    # Get the list of features to create polynomial features
    features = [col for col in df.columns if col in df_features]

    # Create a PolynomialFeatures object with the specified degree, no interaction features, and no bias term
    poly = PolynomialFeatures(degree, interaction_only=False, include_bias=False)

    # Fit and transform the selected features in the DataFrame
    poly_features = poly.fit_transform(df[features])

    # Get the feature names for the generated polynomial features
    poly_features_names = poly.get_feature_names_out(features)

    # Create a new DataFrame with the generated polynomial features
    poly_df = pd.DataFrame(poly_features, columns=poly_features_names)

    # Keep only the columns with polynomial features of the specified degree
    poly_df = poly_df[[f"{col}^{degree}" for col in features]]

    # Concatenate the original DataFrame and the polynomial features DataFrame
    result_combined = pd.concat([df, poly_df], axis=1)

    return df, poly_df, result_combined


def make_mi_scores_classification(X, y):
    mi_scores = mutual_info_classif(X, y)
    mi_scores_df = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores})
    # mi_scores_df = mi_scores_df.sort_values(by='MI Score', asc=False)
    return mi_scores_df


def make_mi_scores_regression(X, y):
    mi_scores = mutual_info_regression(X, y)
    mi_scores_df = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores})
    mi_scores_df = mi_scores_df.sort_values(by='MI Score', ascending=False)
    return mi_scores_df


def add_missing_indicator_columns(df):
    """
    Add indicator columns to the DataFrame for each feature that is missing or contains NaN values.

    Args:
    - df: DataFrame

    Returns:
    - df: DataFrame with indicator columns added
    """
    missing_indicator_columns = df.isna().astype(int)
    missing_indicator_columns.columns = [col + '_missing' for col in missing_indicator_columns.columns]
    return pd.concat([df, missing_indicator_columns], axis=1)


def add_missing_indicator_single_column(df, feature_names):
    """
    Add indicator columns to the DataFrame for each feature that is missing or contains NaN values.

    Args:
    - df: DataFrame

    Returns:
    - df: DataFrame with indicator columns added
    """
    missing_df = pd.DataFrame()

    for feature in feature_names:
        missing_df[feature] = df[[feature]].isna().astype(int)
        missing_df.columns = [col + '_missing' for col in missing_df.columns]
    return missing_df
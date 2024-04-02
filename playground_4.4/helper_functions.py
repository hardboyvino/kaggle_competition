"""
LIST OF FUNCTIONS
1. Plot correlation matrix
2. Brute force feature engineering
3. Describe dataframe uniqueness
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

def plot_correlation_matrix(data, features=None, figsize=(30, 24)):
    if features == None:
        X = data
    else:
        X = data[features]

    # Calculate the correlation matrix
    corr = X.corr(method='spearman')

    # Generate a mas for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Create a heatmap to visualize the correlation matrix
    plt.figure(figsize=figsize)
    sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.05, cbar_kws={"shrink": 0.5})
    plt.title("Correlation Matrix")
    plt.show()



epsilon = 1e-10

def brute_force_feat_engineering(df, numerical_features, binary_cols):
    df_new = df.copy()

    # Transformation of numerical features
    for feature in numerical_features:
        # Log transformation (add small value to avoid log(0))
        df_new[f"{feature}_log"] = np.log(df[feature] + 1e-10)

        # Square root transformation
        df_new[f"{feature}_sqrt"] = np.sqrt(df[feature])

        # Polynomial transformations
        df_new[f"{feature}_degree2"] = df[feature] ** 2
        df_new[f"{feature}_degree3"] = df[feature] ** 3

    # Pairwise numerical feature operations
    for i in range(len(numerical_features)):
        for j in range(i, len(numerical_features)):
            if i != j:
                feat1 = numerical_features[i]
                feat2 = numerical_features[j]

                # Multiplication
                df_new[f"{feat1}_x_{feat2}"] = df[feat1] * df[feat2]

                # Addition
                df_new[f"{feat1}_plus_{feat2}"] = df[feat1] + df[feat2]

                # Subtraction (both ways)
                df_new[f"{feat1}_minus_{feat2}"] = df[feat1] - df[feat2]
                df_new[f"{feat2}_minus_{feat1}"] = df[feat2] - df[feat1]

                # Ratio (both ways)
                df_new[f"{feat1}_div_{feat2}"] = df[feat1] / (df[feat2] + epsilon)
                df_new[f"{feat2}_div_{feat1}"] = df[feat2] / (df[feat1] + epsilon)

                # Mean
                df_new[f"Mean_{feat1}_{feat2}"] = df[[feat1, feat2]].mean(axis=1)

                # Standard Deviation
                df_new[f"Std_{feat1}_{feat2}"] = df[[feat1, feat2]].std(axis=1)

                # Relative ratio (both ways)
                df_new[f"relative_{feat1}_{feat2}"] = df[feat1] / (df[feat1] + df[feat2] + epsilon)
                df_new[f"relative_{feat2}_{feat1}"] = df[feat2] / (df[feat1] + df[feat2] + epsilon)

                # Compare value of features
                df_new[f"{feat1}_gt_{feat2}"] = (df[feat1] > df[feat2]).astype(int)

                # Normalized ratio (both ways)
                df_new[f"{feat1}_by_squared_{feat2}"] = df[feat1] / ((df[feat2] ** 2) + epsilon)
                df_new[f"{feat2}_by_squared_{feat1}"] = df[feat2] / ((df[feat1] ** 2) + epsilon)

                # Symmetry metric
                df_new[f"sym_diff_{feat1}_{feat2}"] = np.abs(df[feat1] - df[feat2]) / (df[feat1] + df[feat2] + epsilon)

    # Additional feature engineering for 3 numerical features
    for i in range(len(numerical_features)):
        for j in range(i + 1, len(numerical_features)):
            for k in range(j + 1, len(numerical_features)):
                feat1, feat2, feat3 = numerical_features[i], numerical_features[j], numerical_features[k]

                # Addition of two features and then subtraction a third
                df_new[f"{feat1}_plus_{feat2}_minus_{feat3}"] = df[feat1] + df[feat2] - df[feat3]
                df_new[f"{feat1}_plus_{feat3}_minus_{feat2}"] = df[feat1] + df[feat3] - df[feat2]
                df_new[f"{feat2}_plus_{feat3}_minus_{feat1}"] = df[feat2] + df[feat3] - df[feat1]

                # Addition of all three features
                df_new[f"{feat1}_plus_{feat2}_plus_{feat3}"] = df[feat1] + df[feat2] + df[feat3]

                # Addition of two features to then be subtracted from the third
                df_new[f"{feat1}_plus_{feat2}_sub_by_{feat3}"] = df[feat3] - (df[feat1] + df[feat2])
                df_new[f"{feat1}_plus_{feat3}_sub_by_{feat2}"] = df[feat2] - (df[feat1] + df[feat3])
                df_new[f"{feat2}_plus_{feat3}_sub_by_{feat1}"] = df[feat1] - (df[feat2] + df[feat3])

                # Multiplication of all three features
                df_new[f"{feat1}_x_{feat2}_x_{feat3}"] = df[feat1] * df[feat2] * df[feat3]

                # One feature divided by the sum of the other two features (three combinations)
                df_new[f"{feat1}_div_sum_of_{feat2}_and_{feat3}"] = df[feat1] / (df[feat2] + df[feat3] + epsilon)
                df_new[f"{feat2}_div_sum_of_{feat1}_and_{feat3}"] = df[feat2] / (df[feat1] + df[feat3] + epsilon)
                df_new[f"{feat3}_div_sum_of_{feat1}_and_{feat2}"] = df[feat3] / (df[feat1] + df[feat2] + epsilon)

                # One feature divided by the multiplication of the other two features (three combinations)
                df_new[f"{feat1}_div_prod_of_{feat2}_and_{feat3}"] = df[feat1] / (df[feat2] * df[feat3] + epsilon)
                df_new[f"{feat2}_div_prod_of_{feat1}_and_{feat3}"] = df[feat2] / (df[feat1] * df[feat3] + epsilon)
                df_new[f"{feat3}_div_prod_of_{feat1}_and_{feat2}"] = df[feat3] / (df[feat1] * df[feat2] + epsilon)

                # One feature multiplied by the sum of the other two features (three combinations)
                df_new[f"{feat1}_prod_sum_of_{feat2}_and_{feat3}"] = df[feat1] * (df[feat2] + df[feat3])
                df_new[f"{feat2}_prod_sum_of_{feat1}_and_{feat3}"] = df[feat2] * (df[feat1] + df[feat3])
                df_new[f"{feat3}_prod_sum_of_{feat1}_and_{feat2}"] = df[feat3] * (df[feat1] + df[feat2])

                # The mean of all three features
                df_new[f"Mean_of_{feat1}_{feat2}_{feat3}"] = df[[feat1, feat2, feat3]].mean(axis=1)

                # The standard deviation of all three features
                df_new[f"Std_of_{feat1}_{feat2}_{feat3}"] = df[[feat1, feat2, feat3]].std(axis=1)

    for col1, col2 in combinations(binary_cols, 2):
        df_new[f"{col1}+{col2}"] = df[col1] + df[col2]

    return df_new



def describe_with_uniques(df, percentiles=[0.01, 0.99]):
    """
    Extends the pandas describe function to include unique counts and the percentage of unique values.

    Parameters:
    - df: DataFrame, the DataFrame to describe.
    - percentiles: list, the percentiles to include in the output. Default is [0.01, 0.99].

    Returns:
    - A DataFrame with the extended description.

    Example usage:
    train_description = describe_with_uniques(df_train)
    print(train_description)

    """
    # Ensure the pandas display format is set to avoid scientific notation
    pd.set_option("display.float_format", lambda x: "%.2f" % x)

    # Calculate the basic description with specified percentiles
    description = df.describe(percentiles=percentiles).T

    # Calculate unique counts for each column and append to description
    unique_counts = df.nunique()
    description["unique"] = unique_counts

    # Calculate the percentage of unique values and append to description
    description["perc_unique"] = description["unique"] / description["count"]

    # Calculate the ratios of the percentiles and the min or max
    description['max/99%'] = description['max'] / description['99%']
    description['min/1%'] = description['1%'] / (description['min'])

    return description

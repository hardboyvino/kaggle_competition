import logging
import pandas as pd
import numpy as np

process_data_options = [
    "change_dtype",
    "remove_normal_outlier",
]

def change_dtype(df, source_dtype, target_dtype):
    """
    Change the data type of selected columns in a DataFrame.

    This function changes the data type of columns that match
    the source data type to the target data type.

    Parameters:
        df (pd.DataFrame): The input DataFrame to modify.
        source_dtype (str or type): The source data type to identify columns for conversion.
        target_dtype (str or type): The target data type to which columns will be converted.

    Returns:
        pd.DataFrame: A new DataFrame with the specified data type changes applied.

    Example Usage:
        # Create a sample DataFrame
        data = {'col1': [1, 2, 3], 'col2': [4.0, 5.0, 6.0]}
        df = pd.DataFrame(data)

        # Change data types
        new_df = change_dtype(df, float, int)

        # Print the changes made
        print(new_df)

    Example Output:
        Output DataFrame (new_df):
           col1  col2
        0     1     4
        1     2     5
        2     3     6
    """

    for col in df.select_dtypes([source_dtype]).columns:
        df[col] = df[col].astype(target_dtype)
    return df


def remove_normal_outlier(feature_name, source_df, target_df):
    """
    Remove outliers from a specified feature in a DataFrame using the 3-sigma rule.

    Parameters:
        feature_name (str): The name of the feature (column) for which outliers should be removed.
        source_df (pd.DataFrame): The source DataFrame containing the data.
        target_df (pd.DataFrame): The target DataFrame where outliers will be removed and the result stored.

    Returns:
        pd.DataFrame: The target DataFrame with outliers removed for the specified feature.

    Example Usage:
        data = {'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000]}
        source_df = pd.DataFrame(data)
        target_df = source_df.copy()

        # Remove outliers from the 'A' column using the remove_normal_outlier function
        target_df = remove_normal_outlier('A', source_df, target_df)

        # Display the target DataFrame with outliers removed
        print(target_df)

    Expected Output:
        Output will vary based on the input data, but for the provided example:

            A
        0   1.0
        1   2.0
        2   3.0
        3   4.0
        4   5.0
        5   6.0
        6   7.0
        7   8.0
        8   9.0
        9  10.0

    In statistics, the 68-95-99.7 rule, also known as the empirical rule (or 3-sigma rule), 
    is a shorthand used to remember the percentage of values that lie within an interval 
    estimate in a normal distribution: 68%, 95%, and 99.7% of the values lie within 
    one, two, and three standard deviations of the mean, respectively.
    """
    # Calculate the mean and standard deviation of the specified feature
    col_mean = source_df[feature_name].mean()
    col_std = source_df[feature_name].std()

    # Define the upper and lower limits for outlier removal
    upper_limit = col_mean + 3 * col_std
    lower_limit = col_mean - 3 * col_std

    # Clip the values in the target DataFrame to remove outliers
    target_df[feature_name] = source_df[feature_name].clip(
        lower=lower_limit, upper=upper_limit
    )

    return target_df


def add_missing_indicator_columns(df):
    """
    Add indicator columns to the DataFrame for each feature that is missing or contains NaN values.

    Parameters:
        df (pd.DataFrame): The input DataFrame to which missing indicator columns will be added.

    Returns:
        pd.DataFrame: A DataFrame with missing indicator columns added.

    Example Usage:
        data = {'A': [1, 2, np.nan, 4],
                'B': [np.nan, 2, 3, 4],
                'C': [1, 2, 3, 4]}
        df = pd.DataFrame(data)

        # Add missing indicator columns
        df_with_missing_indicators = add_missing_indicator_columns(df)

        # Display the resulting DataFrame
        print(df_with_missing_indicators)

    Output:
        Original DataFrame:
           A    B  C
        0  1.0  NaN  1
        1  2.0  2.0  2
        2  NaN  3.0  3
        3  4.0  4.0  4

        DataFrame with Missing Indicator Columns:
           A    B  C  A_missing  B_missing
        0  1.0  NaN  1          0          1
        1  2.0  2.0  2          0          0
        2  NaN  3.0  3          1          0
        3  4.0  4.0  4          0          0
    """
    missing_indicator_columns = df.isna().astype(int)
    missing_indicator_columns.columns = [col + '_missing' for col in missing_indicator_columns.columns]
    return pd.concat([df, missing_indicator_columns], axis=1)


import pandas as pd
import numpy as np

def missing_indicator_columns_only(df):
    """
    Create new DataFrame with indicator columns for each feature that is missing or contains NaN values.

    Parameters:
        df (pd.DataFrame): The input DataFrame possibly containing missing values.

    Returns:
        pd.DataFrame: A DataFrame containing only the new missing indicator columns.

    Example Usage:
        data = {'A': [1, 2, np.nan, 4],
                'B': [np.nan, 2, 3, 4],
                'C': [1, 2, 3, 4]}
        df = pd.DataFrame(data)

        # Add missing indicator columns and store them in a new DataFrame
        missing_indicators = add_missing_indicator_columns(df)

        # Display the resulting DataFrame
        print(missing_indicators)

    Output:
        Missing Indicator Columns:
           A_missing  B_missing
        0          0          1
        1          0          0
        2          1          0
        3          0          0
    """
    missing_indicator_columns = df.isna().astype(int)
    missing_indicator_columns.columns = [col + '_missing' for col in missing_indicator_columns.columns]
    return missing_indicator_columns

"""
A collection of functions useful for Exploratory Data Analysis.

To get started and know what sort of functions are available use the following:
`import eda
eda.exploratory_data_analysis_options()`

This will print a list of all the available functions. The function naming, like with every programming
convention, is named to be self explanatory but extensive docstring is available that provides more detailed
information abouts the parameters, return and example usage.
"""
import numpy as np
import pandas as pd


def exploratory_data_analysis_options():
    exploratory_data_analysis_options = [
        "check_null",
        "check_duplicate",
        "count_unique_values",
        "do_value_counts",
        "check_id_column",
        "check_index",
        "get_feature_names",
        "check_value_counts_across_train_test",
        "get_freq_encoding_feature_names",
        "get_bin_feature_names",
        "get_power_feature_names",
        "get_row_wise_stat_feature_names",
        "get_cat_interaction_features",
        "get_features_with_no_variance",
        "check_if_floats_are_int",
        "create_summary_df",
        "generate_unique_count_summary",
        "calculate_skew_summary",
        "get_columns_with_null_values",
    ]
    for option in exploratory_data_analysis_options:
        print(option)


def check_null(df):
    """
    Calculate the percentage of null (missing) values for each column in a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.Series: A pandas Series containing the percentage of null values for each column.

    Example Usage:
        null_percentages = check_null(data_df)
        print(null_percentages)
    """
    return df.isna().sum() * 100 / len(df)


def check_duplicate(df, subset):
    """
    Check for duplicate rows in a DataFrame and return the count of duplicate rows.

    Parameters:
        df (pd.DataFrame): The input DataFrame to check for duplicates.
        subset (list or None): A list of column names to consider when checking for duplicates.
                              If None, all columns are considered. Default is None.

    Returns:
        int: The count of duplicate rows in the DataFrame.

    Example Usage:
        # Check for duplicate rows in the entire DataFrame
        total_duplicates = check_duplicate(data_df)
        print(f"Total duplicate rows: {total_duplicates}")

        # Check for duplicate rows based on a subset of columns
        subset_duplicates = check_duplicate(data_df, subset=['column1', 'column2'])
        print(f"Subset duplicate rows: {subset_duplicates}")
    """
    if subset is not None:
        return df.duplicated(subset=subset, keep=False).sum()
    else:
        return df.duplicated(keep=False).sum()


def count_unique_values(df, feature_name):
    """
    Count the number of unique values in a specific column of a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        feature_name (str): The name of the column in the DataFrame for which you want to count unique values.

    Returns:
        int: The count of unique values in the specified column.

    Example Usage:
        data = {'A': [1, 2, 2, 3, 4],
                'B': ['apple', 'banana', 'apple', 'cherry', 'date']}
        df = pd.DataFrame(data)

        # Count unique values in column 'A'
        unique_count_A = count_unique_values(df, 'A')
        print(f"Unique count in column 'A': {unique_count_A}")

        # Count unique values in column 'B'
        unique_count_B = count_unique_values(df, 'B')
        print(f"Unique count in column 'B': {unique_count_B}")

    Output:
        Unique count in column 'A': 4
        Unique count in column 'B': 4
    """
    return df[feature_name].nunique()


def do_value_counts(df, feature_name):
    """
    Calculate the percentage of unique values in a specified DataFrame column.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        feature_name (str): The name of the column for which to calculate the value counts.

    Returns:
        pd.Series: A Series containing the percentage of each unique value in the specified column.

    Example Usage:
        # Calculate value counts for a column and return percentages
        value_percentages = do_value_counts(data_df, 'column_name')
        print(value_percentages)

    This function takes a DataFrame and a column name as input and calculates the percentage of each unique value
    in the specified column. It uses the `value_counts` method with `normalize=True` to compute the percentage
    of each unique value, including NaN (missing) values if present. The resulting Series is then sorted in
    descending order by percentage and multiplied by 100 to convert it into percentages.

    Example:
        Suppose we have the following DataFrame 'data_df' with a column named 'category':

        |   | category  |
        |---|-----------|
        | 0 | A         |
        | 1 | B         |
        | 2 | A         |
        | 3 | C         |
        | 4 | NaN       |

        Calling `do_value_counts(data_df, 'category')` will return a Series like this:

        A     40.0
        B     20.0
        C     20.0
        NaN   20.0
        Name: category, dtype: float64

        This Series shows the percentage of each unique value in the 'category' column, including the percentage
        of missing values (NaN).
    """
    return (
        df[feature_name]
        .value_counts(normalize=True, dropna=False)
        .sort_values(ascending=False)
        * 100
    )


def check_index(df, data_set_name):
    """
    Check the properties of the DataFrame's index, including continuity and monotonicity, and visualize it.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        data_set_name (str): A descriptive name or label for the dataset used in the title of the plot.

    Returns:
        None

    Example Usage:
        # Check and visualize the index of a DataFrame
        check_index(data_df, "My Dataset")

    Description:
        This function examines the properties of the index of the input DataFrame 'df'
        and provides information about its continuity and monotonicity. It also generates
        a plot to visualize the index values.

        Continuity:
        - The function checks if the index is continuous, meaning that it does not contain
          any missing values or gaps in its sequence.

        Monotonicity:
        - The function checks if the index is monotonically increasing, indicating that
          the values in the index are strictly increasing from left to right.

        Visualization:
        - The function generates a plot to visualize the index values, providing a visual
          representation of the index's pattern.
    """
    print(f"Is the index monotonic : {df.index.is_monotonic}")
    # Plot the column
    pd.Series(df.index).plot(title=data_set_name)
    plt.show()


def check_id_column(df, column_name, data_set_name):
    """
    Check the properties of a specific identifier column in the DataFrame, including continuity
    and monotonicity, and visualize it.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the identifier column to check.
        data_set_name (str): A descriptive name or label for the dataset used in the title of the plot.

    Returns:
        None

    Example Usage:
        # Check and visualize the 'feature_3' column in a DataFrame
        check_id_column(data_df, 'feature_3', "My Dataset")

    Description:
        This function examines the properties of a specific identifier column ('column_name')
        in the input DataFrame 'df' and provides information about its continuity and monotonicity.
        It also generates a plot to visualize the values in the specified identifier column.

        Continuity:
        - The function checks if the specified identifier column is continuous, meaning that it
          does not contain any missing values or gaps in its sequence.

        Monotonicity:
        - The function checks if the specified identifier column is monotonically increasing,
          indicating that the values in the column are strictly increasing from top to bottom.

        Visualization:
        - The function generates a plot to visualize the values in the specified identifier column,
          providing a visual representation of the column's pattern.
    """
    print(f"Is the {column_name} monotonic : {df[column_name].is_monotonic}")
    # Plot the column
    df[column_name].plot(title=data_set_name)
    plt.show()


def get_feature_names(df, feature_name_substring):
    """
    Returns a list of column names in the DataFrame 'df' that contain the specified 'feature_name_substring'.

    Parameters:
        df (pd.DataFrame): The input DataFrame to search for matching feature names.
        feature_name_substring (str): A substring to search for within the column names.

    Returns:
        list of str: A list of column names containing the specified substring.

    Example Usage:
        data = {
            'feature_1a': [1, 2, 3],
            'feature_1b': [4, 5, 6],
            'feature_2a': [7, 8, 9],
            'feature_2b': [10, 11, 12]
        }
        df = pd.DataFrame(data)

        # Get a list of column names containing 'feature_1'
        feature_list = get_feature_names(df, 'feature_1')
        print(feature_list)
        # Output: ['feature_1a', 'feature_1b']

        # Get a list of column names containing '2b'
        feature_list = get_feature_names(df, '2b')
        print(feature_list)
        # Output: ['feature_2b']
    """
    return [col_name for col_name in df.columns if feature_name_substring in col_name]


def check_value_counts_across_train_test(
    train_df, test_df, feature_name, normalize=True
):
    """
    Calculate the value counts of a specified feature in both a training DataFrame and a test DataFrame.

    Parameters:
        train_df (pd.DataFrame): The training DataFrame containing the data for analysis.
        test_df (pd.DataFrame): The test DataFrame containing the data for analysis.
        feature_name (str): The name of the feature (column) for which value counts are calculated.
        normalize (bool, optional): Whether to normalize the value counts as percentages.
                                    Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame showing the value counts of the specified feature in both the training and test datasets.
                      The DataFrame has three columns: 'feature_name', 'train', and 'test'.
                      - 'feature_name': The name of the specified feature.
                      - 'train': The value counts in the training DataFrame (as percentages if normalize=True).
                      - 'test': The value counts in the test DataFrame (as percentages if normalize=True).

    Example Usage:
        # Create sample training and test DataFrames
        train_data = {'feature': ['A', 'B', 'A', 'C', 'B']}
        test_data = {'feature': ['A', 'C', 'B', 'B', 'C']}
        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)

        # Calculate and display the value counts across train and test DataFrames
        result_df = check_value_counts_across_train_test(train_df, test_df, 'feature')
        print(result_df)
    """
    # Calculate value counts for the feature in both train and test DataFrames
    train_counts = (
        train_df[feature_name]
        .sort_index()
        .value_counts(normalize=normalize, dropna=True)
        * 100
    )
    test_counts = (
        test_df[feature_name]
        .sort_index()
        .value_counts(normalize=normalize, dropna=True)
        * 100
    )

    # Create a DataFrame to display the results
    count_df = pd.concat([train_counts, test_counts], axis=1).reset_index(drop=True)
    count_df.columns = [feature_name, "train", "test"]

    return count_df


def get_freq_encoding_feature_names(df):
    return get_feature_names(df, "freq")


def get_power_feature_names(df):
    power_features = []
    power_feature_keys = ["_square", "_cube", "_fourth", "_cp", "_cnp"]
    for name in df.columns:
        for key in power_feature_keys:
            if key in name:
                power_features.append(name)
    return power_features


def get_row_wise_stat_feature_names():
    return [
        "max",
        "min",
        "sum",
        "mean",
        "std",
        "skew",
        "kurt",
        "med",
        "ptp",
        "percentile_10",
        "percentile_60",
        "percentile_90",
        "quantile_5",
        "quantile_95",
        "quantile_99",
    ]


def get_bin_feature_names(df, bin_size=10):
    return get_feature_names(df, f"cut_bin_{bin_size}")


def get_cat_interaction_features():
    return ["f1_f86", "f1_f55", "f1_f27", "f86_f55", "f86_f27", "f55_f27"]


def get_features_with_no_variance(df):
    return df.columns[df.nunique() <= 1]


def check_if_floats_are_int(df):
    """
    Check if the columns of type float in a DataFrame actually contain whole numbers.

    Parameters:
        df (pd.DataFrame): The input DataFrame to check.

    Returns:
        list: A list of column names containing float columns with only whole numbers.

    Example Usage:
        # Create a sample DataFrame
        data = {'column1': [1.0, 2.0, 3.5, 4.0],
                'column2': [1.0, 2.3, 4.0, 5.0],
                'column3': [1, 2, 3, 4]}

        df = pd.DataFrame(data)

        # Check for float columns with whole numbers
        int_features = check_if_floats_are_int(df)
        print("Columns with float values that are actually integers:")
        for feature in int_features:
            print(f"- {feature}")

        # Output:
        # Columns with float values that are actually integers:
        # - column1
        # - column3
    """
    int_features = []
    for column in df.columns:
        if np.all(np.mod(df[column], 1) == 0):
            print(f"Feature {column} does not have any decimals")
            int_features.append(column)
        else:
            print(f"Feature {column} have decimals")
    return int_features


def create_summary_df(dataframe, target_column):
    """
    Create a summary DataFrame from a given DataFrame with counts and percentages.

    Parameters:
        dataframe (pandas.DataFrame): The input DataFrame containing the data.
        target_column (str): The name of the target column for which to generate the summary.

    Returns:
        pandas.DataFrame: A DataFrame with columns 'target_column_value',
        'target_column_counts', and 'target_column_percentage' containing value counts
        and percentages of the target_column.

    Example Usage:
        # Create a sample DataFrame
        data = {'Category': ['A', 'B', 'A', 'C', 'B', 'A']}
        df = pd.DataFrame(data)

        # Generate a summary DataFrame for the 'Category' column
        summary = create_summary_df(df, 'Category')
        print(summary)

    Output:
        target_column_value  target_column_counts  target_column_percentage
        A                     3                      50.0
        B                     2                      33.33
        C                     1                      16.67
    """
    counts = dataframe[target_column].value_counts()
    percentages = dataframe[target_column].value_counts(normalize=True).mul(100)

    summary_df = pd.DataFrame(
        {
            f"{target_column}_value": counts.index,
            f"{target_column}_counts": counts.values,
            f"{target_column}_percentage": percentages.values,
        }
    )

    return summary_df


def generate_unique_count_summary(df):
    """
    Generate a summary DataFrame showing the unique count, dtype, and fraction of unique values
    for each column in the input DataFrame.

    Parameters:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame:
        A summary DataFrame with columns 'name', 'nunique', 'dtype', and 'fraction'.

    Example Usage:
        # Create a sample DataFrame
        data = {'A': [1, 2, 2, 3, 4],
                'B': ['apple', 'banana', 'apple', 'cherry', 'banana'],
                'C': [1.1, 2.2, 3.3, 4.4, 5.5]}

        df = pd.DataFrame(data)

        # Generate a unique count summary
        summary = generate_unique_count_summary(df)

        # Summary Output
                name  nunique    dtype  fraction
            0     A        4    int64      80.0
            1     C        5  float64     100.0
            2     B        3   object      60.0

    """
    unique_count_df = pd.DataFrame(
        {"name": df.columns, "nunique": df.nunique(), "dtype": df.dtypes}
    )
    unique_count_df["fraction"] = (unique_count_df["nunique"] / len(df)).mul(100)
    unique_count_df = unique_count_df.sort_values(by="fraction")
    unique_count_df = unique_count_df.reset_index(drop=True)

    return unique_count_df


def calculate_skew_summary(dataframe, numeric_features):
    """
    Calculate the skewness for each numeric feature in the input DataFrame.

    Parameters:
        dataframe (pandas.DataFrame): The input DataFrame.
        numeric_features (list): List of column names corresponding to numeric features.

    Returns:
        pandas.DataFrame: A summary DataFrame with columns 'feature' and 'skew'.

    Example Usage:
        # Calculate skewness for a DataFrame called 'data_df' with numeric features
        numeric_features = ['feature1', 'feature2', 'feature3']
        skew_summary = calculate_skew_summary(data_df, numeric_features)
        print(skew_summary)

    This function takes a pandas DataFrame and a list of column names representing numeric
    features within the DataFrame. It then computes the skewness for each of these numeric
    features and returns a summary DataFrame with two columns: 'feature' and 'skew'. The
    'feature' column contains the names of the numeric features, and the 'skew' column
    contains the corresponding skewness values.

    Skewness measures the asymmetry of the probability distribution of a real-valued random
    variable about its mean. A skewness value of 0 indicates a perfectly symmetrical
    distribution. Positive skewness indicates a right-skewed distribution (tail on the right),
    while negative skewness indicates a left-skewed distribution (tail on the left).
    """
    df_skew = pd.DataFrame(dataframe[numeric_features].skew())
    df_skew = df_skew.reset_index()
    df_skew.columns = ["feature", "skew"]

    return df_skew


def get_columns_with_null_values(df):
    """
    Get a list of column names in the DataFrame that contain null (NaN) values.

    Parameters:
        dataframe (pandas.DataFrame): The input DataFrame.

    Returns:
        list: A list of column names with null values.

    Example Usage:
        # Obtain a list of column names with null values
        null_columns = get_columns_with_null_values(data_df)
        print(f"Columns with null values: {null_columns}")
    """
    return [col for col in df.columns if df[col].isnull().any()]

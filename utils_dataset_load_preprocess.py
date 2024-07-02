from imports import *

def load_csv_dataset(dataset_dir, has_label=True):
    """
    Loads data from a CSV file.

    Args:
    dataset_dir (str): Path to the CSV file.
    has_label (bool): Indicates if the dataset has labels.

    Returns:
    DataFrame: Loaded data.
    """
    data = pd.read_csv(dataset_dir)
            
    if has_label:
        X = data.iloc[:, :-1]
        Y = data.iloc[:, -1]
        return X, Y
    else:
        X = data
        return X
    
def split_features(X,nume_index=3):
    """
    Splits the features into numerical and categorical.

    Args:
    X (DataFrame): The features of the dataset.
    nume_index (int) : The indices of numerical features

    Returns:
    (DataFrame, DataFrame): Tuple of numerical and categorical features.
    """
    numerical_features = X.iloc[:, :nume_index]  # First three columns are numerical
    categorical_features = X.iloc[:, nume_index:]  # Remaining columns are binary categorical
    return numerical_features, categorical_features

def scaling_features(numerical_features, scaler_type='MinMaxScaler', verbose=False):
    """
    Standardizes numerical features.

    Args:
    numerical_features (DataFrame): Numerical features of the dataset.
    scaler_type (str or None): Type of scaler to use ('StandardScaler', 'MinMaxScaler', or None).
    verbose (bool): If True, prints statistics before and after scaling.

    Returns:
    DataFrame: Standardized numerical features if scaler_type is not None, else the original numerical features.
    """
    if scaler_type == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_type == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif scaler_type is None:
        scaler = None
    else:
        raise ValueError("scaler_type must be either 'StandardScaler', 'MinMaxScaler', or None")
    
    if verbose:
        print("Statistics before scaling:")
        print(numerical_features.describe())
        if scaler_type is not None:
            print("\nScaling using", scaler_type)
    
    if scaler is not None:
        scaled_features = scaler.fit_transform(numerical_features)
        scaled_features_df = pd.DataFrame(scaled_features, columns=numerical_features.columns, index=numerical_features.index)
        if verbose:
            print("\nStatistics after scaling:")
            print(scaled_features_df.describe())
        return scaled_features_df
    else:
        return numerical_features

def concatenate_features(numerical_features, categorical_features):
    """
    Concatenates numerical and categorical features.

    Args:
    numerical_features (DataFrame): The numerical features of the dataset.
    categorical_features (DataFrame): The categorical features of the dataset.

    Returns:
    DataFrame: Concatenated features.
    """
    concatenated_features = pd.concat([numerical_features, categorical_features], axis=1)
    return concatenated_features

def select_important_features(X, y, model_type='RandomForest', num_features=10, verbose=False):
    """
    Selects important features based on feature importance.

    Args:
    X (DataFrame): The features of the dataset.
    y (Series): The target variable.
    model_type (str): The model to determine feature importance ('RandomForest' or 'LinearRegression').
    num_features (int): The number of top features to select.
    verbose (bool): If True, prints the selected features and their importance.

    Returns:
    DataFrame: Dataset with only the selected important features.
    """
    if model_type == 'RandomForest':
        model = RandomForestClassifier()
        model.fit(X, y)
        importances = model.feature_importances_
    elif model_type == 'LinearRegression':
        model = LinearRegression()
        model.fit(X, y)
        importances = np.abs(model.coef_)
    else:
        raise ValueError("model_type must be either 'RandomForest' or 'LinearRegression'")
    
    feature_importances = pd.DataFrame({'feature': X.columns, 'importance': importances})
    feature_importances = feature_importances.sort_values(by='importance', ascending=False)
    
    selected_features = feature_importances.head(num_features)['feature'].values
    
    if verbose:
        print("Selected features and their importance:\n")
        print(feature_importances.head(num_features))
    
    return X[selected_features]


def get_dataset_variables(X_train, X_test, y_train, y_test):
    max_length_train = max(array.shape[1] for array in X_train)
    max_length_test = max(array.shape[1] for array in X_test)
    max_length = max(max_length_train, max_length_test)

    min_length_train = min(array.shape[1] for array in X_train)
    min_length_test = min(array.shape[1] for array in X_test)
    min_length = min(min_length_train, min_length_test)

    num_variables = X_train[0].shape[0]
    unique_classes = np.unique(np.concatenate((y_train, y_test)))
    num_classes = len(unique_classes)

    return max_length, min_length, num_variables, num_classes
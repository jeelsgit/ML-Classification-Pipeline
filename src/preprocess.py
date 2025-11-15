import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

def load_and_prepare_data(dataset_name='iris'):
    """
    Loads a specified dataset, converts it to a DataFrame, and splits it.
    
    Args:
        dataset_name (str): The name of the dataset to load ('iris' or 'breast_cancer').
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    if dataset_name == 'iris':
        data = load_iris()
        target_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    elif dataset_name == 'breast_cancer':
        data = load_breast_cancer()
        target_names = {0: 'malignant', 1: 'benign'}
    else:
        raise ValueError("Dataset not supported. Choose 'iris' or 'breast_cancer'.")

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target').map(target_names)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """
    Scales numerical features using StandardScaler.
    
    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        
    Returns:
        tuple: X_train_scaled, X_test_scaled
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for better handling
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled

def engineer_features(X):
    """
    Creates new features from existing ones.
    For the Iris dataset, we'll create 'area' features.
    For the Breast Cancer dataset, we'll create a 'mean radius error ratio'.
    
    Args:
        X (pd.DataFrame): The original feature set.
        
    Returns:
        pd.DataFrame: The DataFrame with new features added.
    """
    X_new = X.copy()
    
    # Check for Iris dataset features
    if 'sepal length (cm)' in X_new.columns:
        X_new['sepal_area'] = X_new['sepal length (cm)'] * X_new['sepal width (cm)']
        X_new['petal_area'] = X_new['petal length (cm)'] * X_new['petal width (cm)']
        print("Engineered 'sepal_area' and 'petal_area' features.")
        
    # Check for Breast Cancer dataset features
    elif 'mean radius' in X_new.columns:
        X_new['radius_error_ratio'] = X_new['mean radius'] / (X_new['radius error'] + 1e-6) # Add small value to avoid division by zero
        print("Engineered 'radius_error_ratio' feature.")
        
    return X_new
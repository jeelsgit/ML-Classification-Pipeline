import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def get_logistic_regression_model():
    """Returns a Logistic Regression model."""
    return LogisticRegression(random_state=42, max_iter=200)

def get_knn_model(n_neighbors=5):
    """Returns a K-Nearest Neighbors model."""
    return KNeighborsClassifier(n_neighbors=n_neighbors)

def get_svm_model(C=1.0, kernel='rbf'):
    """Returns a Support Vector Machine model."""
    return SVC(C=C, kernel=kernel, probability=True, random_state=42)

def save_model(model, filename):
    """
    Saves a trained model to a file using joblib.
    
    Args:
        model: The trained sklearn model object.
        filename (str): The path and filename to save the model.
    """
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename):
    """
    Loads a trained model from a file.
    
    Args:
        filename (str): The path and filename of the saved model.
        
    Returns:
        The loaded sklearn model object.
    """
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model

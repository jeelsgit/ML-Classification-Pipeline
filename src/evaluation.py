import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluates a trained model and prints key metrics.
    
    Args:
        model: The trained sklearn model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
        model_name (str): Name of the model for printing.
        
    Returns:
        float: The accuracy score.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"--- {model_name} Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return accuracy

def plot_confusion_matrix(y_test, y_pred, model_name, save_path=None):
    """
    Plots a confusion matrix.
    
    Args:
        y_test (pd.Series): True labels.
        y_pred (np.array): Predicted labels.
        model_name (str): Name of the model for the plot title.
        save_path (str, optional): Path to save the plot.
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=y_test.unique(), yticklabels=y_test.unique())
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_model_comparison(model_names, accuracies, save_path=None):
    """
    Plots a bar chart comparing model accuracies.
    
    Args:
        model_names (list): List of model names.
        accuracies (list): List of corresponding accuracies.
        save_path (str, optional): Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x=model_names, y=accuracies, palette='viridis')
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xlabel('Model')
    plt.ylim(0, 1.0) # Accuracy is between 0 and 1
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.2f}', ha='center', va='bottom')
    if save_path:
        plt.savefig(save_path)
    plt.show()
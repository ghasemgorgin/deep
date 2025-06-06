import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# --- Global Constants ---
RANDOM_STATE = 42
N_HIDDEN_NEURONS = 2000
RBF_WIDTH = 0.08 # Used by gaussian function if not overridden
N_CLASSES = 6    # Number of classes in the dataset
KFOLD_SPLITS = 5 # Number of folds for K-Fold cross-validation
DATASET_PATH = 'Exam2/data/Dataset.csv'
OUTPUT_DIR = 'Exam2/outputs'

# --- Activation Functions ---
def tanh(x: np.ndarray) -> np.ndarray:
    """Tanh activation function."""
    return np.tanh(x)

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def hardlim(x: np.ndarray) -> np.ndarray:
    """Hard limit activation function (binary step)."""
    return np.where(x >= 0, 1, 0)

def gaussian(x: np.ndarray, width: float = RBF_WIDTH) -> np.ndarray:
    """Gaussian activation function."""
    return np.exp(-(x**2) / (2 * width**2))

activation_functions = {
    'tanh': tanh,
    'sigmoid': sigmoid,
    'hardlim': hardlim,
    'gaussian': gaussian
}

# --- Data Loading ---
def load_data(file_path: str) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Loads feature and target data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        tuple[np.ndarray, np.ndarray] | tuple[None, None]: (X_data, y_data) or (None, None) if loading fails.
    """
    try:
        data = pd.read_csv(file_path)
        X_data = data.iloc[:, :-1].values
        y_data = data.iloc[:, -1].values
        print(f"Data loaded successfully from {file_path}")
        return X_data, y_data
    except FileNotFoundError:
        print(f"ERROR: Dataset not found at {file_path}")
        return None, None

# --- ELM Core Functions ---
def train_elm(X_train_data: np.ndarray, y_train_one_hot: np.ndarray,
              num_hidden_neurons: int, activation_func) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None,None,None]:
    """
    Trains an Extreme Learning Machine (ELM).

    Args:
        X_train_data (np.ndarray): Scaled training features.
        y_train_one_hot (np.ndarray): One-hot encoded training target.
        num_hidden_neurons (int): Number of hidden neurons.
        activation_func (callable): Activation function for the hidden layer.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None,None,None]:
            Input weights, biases, output weights. Returns (None,None,None) on pseudo-inverse failure.
    """
    n_samples, n_features = X_train_data.shape
    # Initialize input weights and biases randomly
    input_weights = np.random.randn(n_features, num_hidden_neurons) * np.sqrt(2.0 / n_features)
    biases = np.random.randn(1, num_hidden_neurons)

    # Calculate hidden layer output matrix H
    H = np.dot(X_train_data, input_weights) + biases
    H = activation_func(H)

    # Calculate output weights using Moore-Penrose pseudo-inverse
    try:
        H_pinv = np.linalg.pinv(H)
    except np.linalg.LinAlgError:
        print("ERROR: Pseudo-inverse calculation failed during ELM training. Skipping this training attempt.")
        return None, None, None
    output_weights = np.dot(H_pinv, y_train_one_hot)
    return input_weights, biases, output_weights

def predict_elm(X_test_data: np.ndarray, input_weights: np.ndarray,
                biases: np.ndarray, output_weights: np.ndarray, activation_func) -> np.ndarray:
    """
    Predicts class labels using the trained ELM.

    Args:
        X_test_data (np.ndarray): Scaled test features.
        input_weights (np.ndarray): Trained input weights.
        biases (np.ndarray): Trained biases.
        output_weights (np.ndarray): Trained output weights.
        activation_func (callable): Activation function used during training.

    Returns:
        np.ndarray: Predicted class labels.
    """
    H_test = np.dot(X_test_data, input_weights) + biases
    H_test = activation_func(H_test)
    y_pred_probabilities = np.dot(H_test, output_weights)
    return np.argmax(y_pred_probabilities, axis=1) # Return class with highest probability

# --- Experiment Execution ---
def run_experiment_for_activation(
    X_input: np.ndarray, y_input: np.ndarray,
    num_hidden_neurons: int, current_activation_func,
    num_classes_local: int, random_state_local: int, k_splits: int
) -> list[float]:
    """
    Runs K-Fold cross-validation for ELM with a specific activation function.
    Handles data scaling within each fold.

    Args:
        X_input (np.ndarray): Full input feature set.
        y_input (np.ndarray): Full input target set.
        num_hidden_neurons (int): Number of hidden neurons for ELM.
        current_activation_func (callable): The activation function to test.
        num_classes_local (int): Number of target classes.
        random_state_local (int): Random state for KFold.
        k_splits (int): Number of splits for KFold.

    Returns:
        list[float]: List of accuracy scores for each fold.
    """
    kf = KFold(n_splits=k_splits, shuffle=True, random_state=random_state_local)
    fold_accuracies = []
    # One-hot encode y for ELM training (target for classification)
    y_one_hot_encoded = np.eye(num_classes_local)[y_input.astype(int)]

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_input)):
        X_train, X_test = X_input[train_idx], X_input[test_idx]
        y_train_one_hot, y_test_original_labels = y_one_hot_encoded[train_idx], y_input[test_idx]

        # Scaling: Fit on X_train for this fold, transform X_train and X_test
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        input_w, bias_w, output_w = train_elm(X_train_scaled, y_train_one_hot, num_hidden_neurons, current_activation_func)

        if input_w is None: # ELM training failed (e.g. pseudo-inverse error)
            print(f"  Fold {fold + 1} ELM training failed. Recording 0 accuracy.")
            fold_accuracies.append(0.0)
            continue

        y_predictions = predict_elm(X_test_scaled, input_w, bias_w, output_w, current_activation_func)

        acc = accuracy_score(y_test_original_labels, y_predictions)
        fold_accuracies.append(acc)
        print(f"  Fold {fold + 1} Accuracy: {acc:.4f}")
    return fold_accuracies

# --- Main Orchestration ---
def main():
    """
    Main function to run ELM activation function comparison experiment.
    """
    np.random.seed(RANDOM_STATE) # Ensure reproducibility for ELM weights and KFold shuffling

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    X, y = load_data(DATASET_PATH)
    if X is None or y is None:
        print("Halting execution due to data loading failure.")
        return

    all_fold_results = {}
    for act_func_name, act_func_callable in activation_functions.items():
        print(f"\nRunning ELM experiment for '{act_func_name}' activation function...")
        accuracies = run_experiment_for_activation(
            X, y, N_HIDDEN_NEURONS, act_func_callable, N_CLASSES, RANDOM_STATE, KFOLD_SPLITS
        )
        all_fold_results[act_func_name] = accuracies

    # Plotting comparison of activation functions
    plt.figure(figsize=(12, 7))
    for name, accs_list in all_fold_results.items():
        if accs_list: # Check if list is not empty (e.g. if all folds failed)
            plt.plot(range(1, len(accs_list) + 1), accs_list, marker='o', linestyle='-',
                     label=f'{name} (Mean Acc: {np.mean(accs_list):.4f})')
        else:
            plt.plot([], [], marker='x', linestyle='-', label=f'{name} (No successful folds)')


    plt.title(f'Comparison of ELM Activation Functions ({KFOLD_SPLITS}-Fold CV)')
    plt.xlabel('Fold Number')
    plt.ylabel('Accuracy')
    plt.legend(loc='best') # Auto-choose best legend placement
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(range(1, KFOLD_SPLITS + 1))
    plt.ylim(0.0, 1.0) # Standard accuracy y-axis range

    plot_filename = "elm_activation_function_comparison.png"
    full_plot_path = os.path.join(OUTPUT_DIR, plot_filename)
    plt.savefig(full_plot_path)
    # plt.show() # Typically remove plt.show() for automated script runs, keep for interactive
    plt.close() # Ensure figure is closed
    print(f"Comparison plot saved to {full_plot_path}")

if __name__ == '__main__':
    main()

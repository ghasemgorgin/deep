import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import time
import os

# --- Global Constants ---
DATASET_PATH = 'Exam2/data/Dataset.csv'
OUTPUT_DIR = 'Exam2/outputs' # Defined for good practice, though not used for saving by this script
RANDOM_STATE = 42
N_HIDDEN_NEURONS_FAST_ELM = 2000
SAMPLE_SIZE_FAST_ELM = 20000  # Number of samples to draw from the dataset for evaluation
TEST_SPLIT_SIZE_FAST_ELM = 0.2 # Proportion of the sample to use as the test set
N_CLASSES_FAST_ELM = 6 # Number of classes in the dataset, used for one-hot encoding

# --- Activation Function ---
def tanh_activation(x: np.ndarray) -> np.ndarray:
    """
    Computes the hyperbolic tangent (tanh) activation function.

    Args:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Output array after applying tanh.
    """
    return np.tanh(x)

# --- ELM Core Functions ---
def train_elm_basic(X_train_data: np.ndarray, y_train_one_hot: np.ndarray, n_hidden_units: int) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Trains a basic Extreme Learning Machine (ELM) with Tanh activation.

    Args:
        X_train_data (np.ndarray): Scaled training features.
        y_train_one_hot (np.ndarray): One-hot encoded training target labels.
        n_hidden_units (int): Number of hidden neurons.

    Returns:
        tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
            Input weights, biases, output weights. Returns (None, None, None) if pseudo-inverse fails.
    """
    n_samples, n_features = X_train_data.shape
    # Initialize input weights and biases randomly
    # Weight initialization can impact performance; this is a common approach.
    input_weights = np.random.randn(n_features, n_hidden_units) * np.sqrt(2.0 / n_features)
    biases = np.random.randn(1, n_hidden_units)

    # Calculate hidden layer output matrix H
    H = tanh_activation(np.dot(X_train_data, input_weights) + biases)

    # Calculate output weights using Moore-Penrose pseudo-inverse
    try:
        H_pinv = np.linalg.pinv(H)
    except np.linalg.LinAlgError:
        print("ERROR: Pseudo-inverse calculation failed during ELM training. Matrix may be singular.")
        return None, None, None # Indicate failure
    output_weights = np.dot(H_pinv, y_train_one_hot)
    return input_weights, biases, output_weights

def predict_elm_basic(X_test_data: np.ndarray, input_weights: np.ndarray, biases: np.ndarray, output_weights: np.ndarray) -> np.ndarray:
    """
    Predicts class labels using trained basic ELM weights and Tanh activation.

    Args:
        X_test_data (np.ndarray): Scaled test features.
        input_weights (np.ndarray): Trained input weights from ELM.
        biases (np.ndarray): Trained biases from ELM.
        output_weights (np.ndarray): Trained output weights from ELM.

    Returns:
        np.ndarray: Predicted class labels (indices).
    """
    H_test = tanh_activation(np.dot(X_test_data, input_weights) + biases)
    y_pred_probabilities = np.dot(H_test, output_weights)
    return np.argmax(y_pred_probabilities, axis=1) # Return index of max probability for class label

# --- Data Handling ---
def load_data_for_fast_elm(file_path: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Loads feature (X) and target (y) data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        tuple[np.ndarray | None, np.ndarray | None]: (X, y) arrays or (None, None) if file not found.
    """
    try:
        data = pd.read_csv(file_path)
        X = data.iloc[:, :-1].values # All columns except the last are features
        y = data.iloc[:, -1].values   # The last column is the target
        print(f"Data loaded from {file_path}. X shape: {X.shape}, y shape: {y.shape}")
        return X, y
    except FileNotFoundError:
        print(f"ERROR: Dataset CSV file not found at {file_path}")
        return None, None

def sample_split_scale_data(
    X_full: np.ndarray, y_full: np.ndarray,
    sample_sz: int, test_sz: float,
    random_state_local: int, num_classes_local: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Samples the full dataset, splits into training and test sets,
    one-hot encodes the training target, and scales features.

    Args:
        X_full (np.ndarray): Full feature dataset.
        y_full (np.ndarray): Full target dataset.
        sample_sz (int): Number of samples to draw from the full dataset.
        test_sz (float): Proportion of the sample to use for the test set.
        random_state_local (int): Random state for reproducibility of sampling and splitting.
        num_classes_local (int): Number of unique classes for one-hot encoding y_train.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Scaled training features, scaled test features,
            one-hot encoded training labels, original test labels.
    """
    if X_full.shape[0] < sample_sz:
        print(f"Warning: Original dataset size ({X_full.shape[0]}) is less than requested sample size ({sample_sz}). Using full dataset for sampling.")
        sample_sz = X_full.shape[0] # Adjust sample size if dataset is smaller

    # Randomly sample from the full dataset without replacement
    indices = np.random.choice(X_full.shape[0], sample_sz, replace=False)
    X_sampled = X_full[indices]
    y_sampled = y_full[indices]

    # Split the sample into training and test sets
    X_train, X_test, y_train_labels, y_test_labels = train_test_split(
        X_sampled, y_sampled, test_size=test_sz, random_state=random_state_local, stratify=y_sampled
    )

    # One-hot encode training labels for ELM compatibility
    y_train_one_hot = np.eye(num_classes_local)[y_train_labels.astype(int)]

    # Scale features: fit scaler on X_train, then transform X_train and X_test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Data sampled (size={sample_sz}), split (test_size={test_sz}), and scaled.")
    return X_train_scaled, X_test_scaled, y_train_one_hot, y_test_labels

# --- Main Evaluation Logic for Fast ELM ---
def run_fast_elm_evaluation(
    X_full_data: np.ndarray, y_full_data: np.ndarray,
    n_hidden_units: int, sample_sz: int, test_sz: float,
    random_state_local: int, num_classes_local: int
) -> tuple[float, dict]:
    """
    Runs the fast ELM evaluation using sampling and hold-out validation.

    Args:
        X_full_data (np.ndarray): Full feature dataset.
        y_full_data (np.ndarray): Full target dataset.
        n_hidden_units (int): Number of hidden neurons for ELM.
        sample_sz (int): Size of the random sample to draw.
        test_sz (float): Proportion of the sample for testing.
        random_state_local (int): Random state for all random operations.
        num_classes_local (int): Number of classes for one-hot encoding.

    Returns:
        tuple[float, dict]: Overall accuracy on the test sample, and dictionary of per-class accuracies.
    """

    X_train_s, X_test_s, y_train_ohe, y_test_l = sample_split_scale_data(
        X_full_data, y_full_data, sample_sz, test_sz, random_state_local, num_classes_local
    )

    print(f"Training ELM with {n_hidden_units} hidden neurons on a sample of {X_train_s.shape[0]} records...")
    start_time = time.time()

    input_w, bias_w, output_w = train_elm_basic(X_train_s, y_train_ohe, n_hidden_units)

    if input_w is None: # Check if ELM training failed
        print("ELM training failed, aborting evaluation for this run.")
        return 0.0, {} # Return zero accuracy and empty class accuracies dictionary

    y_pred_labels = predict_elm_basic(X_test_s, input_w, bias_w, output_w)
    elapsed_time = time.time() - start_time

    main_accuracy = accuracy_score(y_test_l, y_pred_labels)
    print(f"Fast ELM training and prediction completed in {elapsed_time:.2f} seconds. Test Accuracy: {main_accuracy:.4f}")

    # Calculate per-class accuracies on the test sample
    per_class_accuracies = {}
    for cls_label in np.unique(y_test_l):
        cls_mask = (y_test_l == cls_label)
        # Ensure there are samples for this class in test set to avoid division by zero in accuracy_score if it were empty
        if np.sum(cls_mask) > 0:
            class_acc = accuracy_score(y_test_l[cls_mask], y_pred_labels[cls_mask])
            per_class_accuracies[cls_label] = class_acc
        else:
            per_class_accuracies[cls_label] = 0.0 # Or np.nan, or skip

    return main_accuracy, per_class_accuracies

# --- Main Script Runner ---
def main_fast_elm_runner():
    """
    Main function to orchestrate the fast ELM evaluation.
    This script provides a quick estimate of ELM performance by training on a sample of the data.
    """
    np.random.seed(RANDOM_STATE) # For reproducibility of sampling and ELM weights initialization

    # Create output directory (though not actively used for saving files by this script)
    if not os.path.exists(OUTPUT_DIR) and OUTPUT_DIR:
        os.makedirs(OUTPUT_DIR)
        print(f"Output directory ensured: {OUTPUT_DIR}")

    X_global, y_global = load_data_for_fast_elm(DATASET_PATH)

    if X_global is None or y_global is None:
        print("Halting: Data loading failed.")
        return

    # Run the fast ELM evaluation using defined constants
    estimated_accuracy, class_wise_accuracies = run_fast_elm_evaluation(
        X_global, y_global,
        n_hidden_units=N_HIDDEN_NEURONS_FAST_ELM,
        sample_sz=SAMPLE_SIZE_FAST_ELM,
        test_sz=TEST_SPLIT_SIZE_FAST_ELM,
        random_state_local=RANDOM_STATE,
        num_classes_local=N_CLASSES_FAST_ELM
    )

    # Report final results
    print(f"\nEstimated Overall Accuracy (based on sampled data): {estimated_accuracy:.4f}")
    print("\nClass-wise Accuracies on Test Sample (estimated):")

    # Example map for class labels to names, can be customized or loaded if available
    emotion_names_map = {
        0: 'Happiness', 1: 'Surprise', 2: 'Anger',
        3: 'Sadness', 4: 'Disgust', 5: 'Fear'
    }
    for cls_label, acc in class_wise_accuracies.items():
        class_name = emotion_names_map.get(cls_label, f"Class {cls_label}") # Fallback to "Class X"
        print(f"  {class_name}: {acc:.4f}")

if __name__ == '__main__':
    main_fast_elm_runner()
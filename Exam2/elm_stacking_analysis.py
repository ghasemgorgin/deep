import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from hpelm import ELM # Assuming hpelm is the correct library for ELM
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import time
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib # For matplotlib.use('Agg')
import warnings

# --- Global Constants ---
BASE_OUTPUT_DIR = "Exam2/outputs"
DATA_PATH = os.path.join("Exam2", "data", "Dataset.csv")
RANDOM_STATE = 42
NEURON_COUNTS = [500, 1000, 1500] # Configuration for ELM neurons in Stacking
TEST_SPLIT_SIZE = 0.2 # Proportion of data for the test set

# ELMWrapper default parameters (can be overridden)
ELM_DEFAULT_FUNC = 'sigm'
ELM_DEFAULT_BATCH = 1024
ELM_DEFAULT_L2 = 0.01

# Stacking Base Estimator Parameters
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
RF_MIN_SAMPLES_SPLIT = 5
LGBM_NUM_LEAVES = 31
LGBM_LEARNING_RATE = 0.05
LGBM_N_ESTIMATORS = 100
LGBM_SUBSAMPLE = 0.8
LGBM_COLSAMPLE_BYTREE = 0.8

# Stacking Classifier Parameters
STACKING_CV_FOLDS = 3 # CV folds for StackingClassifier's meta-feature generation
FINAL_ESTIMATOR_MAX_ITER = 1000
FINAL_ESTIMATOR_SOLVER = 'liblinear'


# --- Initial Setup ---
warnings.filterwarnings("ignore", category=UserWarning)
try:
    matplotlib.use('Agg')
except Exception as e:
    print(f"INFO: Could not set matplotlib backend to Agg: {e}.")


# --- ELMWrapper Class ---
class ELMWrapper(BaseEstimator, ClassifierMixin):
    """
    A scikit-learn compatible wrapper for Extreme Learning Machines (ELM) using the hpelm library.
    Allows ELM to be used as an estimator in scikit-learn pipelines and meta-estimators like StackingClassifier.
    """
    def __init__(self, n_neurons: int = 1000, func: str = ELM_DEFAULT_FUNC,
                 batch: int = ELM_DEFAULT_BATCH, l2: float = ELM_DEFAULT_L2):
        """
        Initialize the ELMWrapper.

        Args:
            n_neurons (int): Number of hidden neurons in the ELM.
            func (str): Activation function for hidden neurons (e.g., 'sigm', 'tanh', 'lin').
            batch (int): Batch size for ELM training (relevant for very large datasets).
            l2 (float): L2 regularization parameter for ELM.
        """
        self.n_neurons = n_neurons
        self.func = func
        self.batch = batch
        self.l2 = l2
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ELMWrapper':
        """
        Fit the ELM model to the training data.

        Args:
            X (pd.DataFrame or np.ndarray): Training features.
            y (pd.Series or np.ndarray): Training target values.

        Returns:
            ELMWrapper: The fitted estimator.
        """
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_one_hot = pd.get_dummies(y).values # ELM typically requires one-hot encoded targets for classification

        if X_np.shape[1] == 0:
            raise ValueError("Input X has 0 features. ELM cannot be initialized.")

        self.model = ELM(X_np.shape[1], y_one_hot.shape[1], classification="c", batch=self.batch)
        self.model.add_neurons(self.n_neurons, self.func)
        self.model.train(X_np, y_one_hot, "c", l2=self.l2) # 'c' for classification
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Args:
            X (pd.DataFrame or np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted class labels.
        """
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        predictions = self.model.predict(X_np)
        return np.argmax(predictions, axis=1) # Return class with highest probability

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator."""
        return {"n_neurons": self.n_neurons, "func": self.func, "batch": self.batch, "l2": self.l2}

    def set_params(self, **parameters) -> 'ELMWrapper':
        """Set the parameters of this estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

# --- Data Handling Function ---
def load_and_preprocess_data(csv_path: str, random_state_local: int, test_size_local: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list] | tuple[None]*5 :
    """
    Loads data from CSV, shuffles, splits into training/testing sets, and scales features.

    Args:
        csv_path (str): Path to the dataset CSV file.
        random_state_local (int): Random state for shuffling and splitting.
        test_size_local (float): Proportion of data for the test set.

    Returns:
        tuple: (X_train_scaled_df, X_test_scaled_df, y_train_series, y_test_series, feature_column_names)
               or (None, None, None, None, None) if data loading fails.
    """
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: Dataset not found at {csv_path}.")
        return None, None, None, None, None

    print(f"Dataset original shape: {data.shape}, Columns: {data.columns.tolist()}")
    # Shuffle data to ensure randomness before splitting
    data = data.sample(frac=1, random_state=random_state_local).reset_index(drop=True)

    X_raw = data.iloc[:, :-1].values.astype(float) # Assuming all features are numeric or become numeric
    y_raw = data.iloc[:, -1].values # Target column

    # Create generic feature names if not available or to ensure consistency
    feature_column_names = [f'feature_{i}' for i in range(X_raw.shape[1])]
    X_df = pd.DataFrame(X_raw, columns=feature_column_names)

    # Split data into training and testing sets
    X_train_df, X_test_df, y_train_series, y_test_series = train_test_split(
        X_df, y_raw, test_size=test_size_local, random_state=random_state_local, stratify=y_raw
    )

    # Scale features: fit scaler on training data, then transform both train and test sets
    scaler = StandardScaler()
    X_train_scaled_np = scaler.fit_transform(X_train_df)
    X_test_scaled_np = scaler.transform(X_test_df)

    # Convert scaled NumPy arrays back to DataFrames with column names
    X_train_scaled_df = pd.DataFrame(X_train_scaled_np, columns=X_train_df.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled_np, columns=X_test_df.columns)

    print("Data loaded, preprocessed (shuffled, split, scaled).")
    return X_train_scaled_df, X_test_scaled_df, y_train_series, y_test_series, feature_column_names

# --- Plotting Function ---
def plot_and_save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, title: str, base_dir: str, filename: str) -> None:
    """
    Plots and saves a confusion matrix.

    Args:
        y_true (pd.Series or np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        title (str): Title for the plot.
        base_dir (str): Base directory to save the plot.
        filename (str): Filename for the saved plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues') # 'd' for integer format
    plt.title(title); plt.ylabel('True Label'); plt.xlabel('Predicted Label')

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    full_path = os.path.join(base_dir, filename)
    plt.savefig(full_path)
    plt.close() # Close figure to free memory
    print(f"Confusion matrix saved: {full_path}")

# --- Core Evaluation Function ---
def evaluate_stacking_configuration(
    n_hidden_elm: int, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,
    base_output_dir_local: str, random_state_local: int
) -> dict:
    """
    Defines, trains, and evaluates a stacking classifier configuration with ELM.
    Includes LightGBM GPU fallback.

    Args:
        n_hidden_elm (int): Number of hidden neurons for the ELM base model.
        X_train (pd.DataFrame): Scaled training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Scaled test features.
        y_test (pd.Series): Test target.
        base_output_dir_local (str): Directory for saving plots.
        random_state_local (int): Random state for reproducibility.

    Returns:
        dict: Dictionary containing performance metrics and file paths.
    """
    start_time = time.time()

    # Define ELM base model
    elm_model = ELMWrapper(n_neurons=n_hidden_elm, func=ELM_DEFAULT_FUNC, batch=ELM_DEFAULT_BATCH, l2=ELM_DEFAULT_L2)

    # Define RandomForest base model
    rf_model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS, random_state=random_state_local, n_jobs=-1,
        max_depth=RF_MAX_DEPTH, min_samples_split=RF_MIN_SAMPLES_SPLIT
    )

    # Define LightGBM base model with GPU attempt and CPU fallback
    lgbm_params = {
        'num_leaves': LGBM_NUM_LEAVES, 'learning_rate': LGBM_LEARNING_RATE,
        'n_estimators': LGBM_N_ESTIMATORS, 'random_state': random_state_local,
        'subsample': LGBM_SUBSAMPLE, 'colsample_bytree': LGBM_COLSAMPLE_BYTREE
    }
    try:
        lgbm_model = LGBMClassifier(**lgbm_params, device="gpu")
        print("INFO: Attempting LightGBM with GPU.")
        # Note: A quick test fit could be done here, but LightGBM usually errors out on fit if GPU is problematic.
    except Exception as e:
        print(f"INFO: LightGBM GPU initialization/support failed ({e}), falling back to CPU.")
        lgbm_model = LGBMClassifier(**lgbm_params)

    estimators = [('elm', elm_model), ('rf', rf_model), ('lgbm', lgbm_model)]

    # Define Stacking Classifier
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=FINAL_ESTIMATOR_MAX_ITER, solver=FINAL_ESTIMATOR_SOLVER, random_state=random_state_local),
        cv=STACKING_CV_FOLDS,
        n_jobs=1 # n_jobs > 1 can cause issues with some estimators (like hpelm) if not properly picklable
    )

    print(f"Fitting StackingClassifier with ELM ({n_hidden_elm} neurons)...")
    stacking_clf.fit(X_train, y_train)
    y_pred = stacking_clf.predict(X_test)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    class_report = classification_report(y_test, y_pred, zero_division=0)

    # Save confusion matrix plot
    cm_filename = f"stacking_cm_elm_neurons_{n_hidden_elm}.png"
    cm_title = f"Stacking CM (ELM Neurons: {n_hidden_elm}, Test Set)"
    plot_and_save_confusion_matrix(y_test, y_pred, cm_title, base_output_dir_local, cm_filename)

    elapsed_time = time.time() - start_time

    metrics_summary = {
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1,
        "class_report": class_report, "time_taken_seconds": elapsed_time,
        "confusion_matrix_file": os.path.join(base_output_dir_local, cm_filename)
    }
    return metrics_summary

# --- Main Orchestration Function ---
def main():
    """Main function to orchestrate the ELM Stacking Classifier analysis."""
    # Ensure base output directory exists
    if not os.path.exists(BASE_OUTPUT_DIR):
        os.makedirs(BASE_OUTPUT_DIR)
        print(f"Created base output directory: {BASE_OUTPUT_DIR}")

    # Load and preprocess data
    # The `_` is used for feature_cols as it's not directly used in this main flow after data loading
    X_train_scaled_df, X_test_scaled_df, y_train_series, y_test_series, _ = \
        load_and_preprocess_data(DATA_PATH, RANDOM_STATE, TEST_SPLIT_SIZE)

    if X_train_scaled_df is None: # Check if data loading failed
        print("Halting execution due to data loading error.")
        return

    all_run_results = []
    for num_neurons in NEURON_COUNTS:
        print(f"\n--- Evaluating Stacking Classifier: ELM with {num_neurons} Hidden Neurons ---")

        current_run_metrics = evaluate_stacking_configuration(
            num_neurons, X_train_scaled_df, y_train_series, X_test_scaled_df, y_test_series,
            BASE_OUTPUT_DIR, RANDOM_STATE
        )
        current_run_metrics["elm_neurons"] = num_neurons # Add neuron count for easy reference in results
        all_run_results.append(current_run_metrics)

        # Print results for the current configuration
        print(f"Results for ELM with {num_neurons} neurons:")
        for metric_name, metric_value in current_run_metrics.items():
            if metric_name == "class_report":
                print(f"  Classification Report:\n{metric_value}")
            else:
                # Format floats to 4 decimal places, keep strings as is
                value_str = f"{metric_value:.4f}" if isinstance(metric_value, float) else str(metric_value)
                print(f"  {metric_name.replace('_', ' ').title()}: {value_str}")

    # Identify and print the best configuration based on accuracy
    if all_run_results:
        best_overall_run = max(all_run_results, key=lambda x: x["accuracy"])
        print("\n--- Overall Best Stacking Configuration (based on Accuracy) ---")
        for metric_name, metric_value in best_overall_run.items():
            if metric_name == "class_report":
                print(f"  Classification Report (Best Run):\n{metric_value}")
            else:
                value_str = f"{metric_value:.4f}" if isinstance(metric_value, float) else str(metric_value)
                print(f"  {metric_name.replace('_', ' ').title()}: {value_str}")
    else:
        print("No configurations were evaluated.")

if __name__ == '__main__':
    main()

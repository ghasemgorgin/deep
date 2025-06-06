import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pyswarms as ps
import time
import os # Added for os.path.join if any output saving is added later

# --- Global Constants ---
DATASET_PATH = 'Exam2/data/Dataset.csv'
OUTPUT_DIR = 'Exam2/outputs' # Defined for good practice, though not used for saving files yet
RANDOM_STATE = 42
N_CLASSES = 6  # Number of classes in the dataset
N_HIDDEN_NEURONS_PSO = 100  # Number of hidden neurons for PSO-ELM
PSO_ITERATIONS = 50 # Number of iterations for PSO
PSO_PARTICLES = 15   # Number of particles for PSO
KFOLD_SPLITS = 5

# --- ELM Core Functions ---
def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def predict_elm(X_input, input_weights, biases, output_weights):
    """Predicts outputs using trained ELM weights."""
    H = sigmoid(np.dot(X_input, input_weights) + biases)
    return np.dot(H, output_weights)

# --- PSO-ELM Specific Functions ---
def fitness_function(params, X_train_pso, y_train_pso, n_features_pso, n_hidden_pso, num_classes_pso):
    """
    Fitness function for PSO.
    Calculates negative accuracy for minimization.
    `params` contains the weights and biases for each particle.
    """
    n_particles = params.shape[0]
    losses = []
    # One-hot encode y_train_pso for ELM output layer calculation
    y_one_hot_pso = np.eye(num_classes_pso)[y_train_pso]

    for i in range(n_particles):
        particle_params = params[i]
        # Reshape particle parameters to input weights (iw) and biases (b)
        iw = particle_params[:n_features_pso * n_hidden_pso].reshape(n_features_pso, n_hidden_pso)
        b = particle_params[n_features_pso * n_hidden_pso:].reshape(1, n_hidden_pso)

        # Calculate hidden layer output (H)
        H = sigmoid(np.dot(X_train_pso, iw) + b)

        # Calculate output weights (beta) using pseudo-inverse
        # This is the core of ELM training: solving H * beta = Y
        try:
            H_pinv = np.linalg.pinv(H)
        except np.linalg.LinAlgError:
            # Handle cases where pseudo-inverse might fail (e.g. singular matrix)
            # This might happen if H is ill-conditioned.
            # Assign a very high loss to penalize this particle.
            losses.append(1.0) # Max loss (0% accuracy)
            print("Warning: Pseudo-inverse calculation failed in fitness function.")
            continue

        beta = np.dot(H_pinv, y_one_hot_pso)

        # Predict on training data to evaluate this particle's parameters
        y_pred_pso_train = np.dot(H, beta)

        # Calculate accuracy
        acc = accuracy_score(np.argmax(y_one_hot_pso, axis=1), np.argmax(y_pred_pso_train, axis=1))
        losses.append(-acc)  # PSO minimizes, so use negative accuracy
    return np.array(losses)

def train_elm_with_pso(X_train_data, y_train_data, n_hidden_units, num_classes_pso,
                       n_iterations=PSO_ITERATIONS, n_particles=PSO_PARTICLES):
    """
    Trains an ELM using PSO to optimize input weights and biases.
    """
    n_samples, n_features = X_train_data.shape
    # Dimensions for PSO: (n_features * n_hidden) weights + n_hidden biases
    dimensions = n_features * n_hidden_units + n_hidden_units

    # PSO options: c1 (cognitive), c2 (social), w (inertia)
    # These are common starting points for PSO parameters.
    pso_options = {'c1': 1.5, 'c2': 1.5, 'w': 0.6}
    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=dimensions, options=pso_options)

    # Optimize to find the best input weights and biases
    # The fitness_function requires additional arguments, passed via kwargs for `optimizer.optimize`
    cost, best_params = optimizer.optimize(
        fitness_function, iters=n_iterations, verbose=True,
        # Additional args for fitness_function:
        X_train_pso=X_train_data, y_train_pso=y_train_data,
        n_features_pso=n_features, n_hidden_pso=n_hidden_units,
        num_classes_pso=num_classes_pso
    )

    # Extract optimized input weights and biases
    best_iw = best_params[:n_features * n_hidden_units].reshape(n_features, n_hidden_units)
    best_b = best_params[n_features * n_hidden_units:].reshape(1, n_hidden_units)

    # Calculate final output weights using the optimized iw and b
    H_optimized = sigmoid(np.dot(X_train_data, best_iw) + best_b)
    y_one_hot_train = np.eye(num_classes_pso)[y_train_data]

    try:
        H_opt_pinv = np.linalg.pinv(H_optimized)
    except np.linalg.LinAlgError:
        print("ERROR: Pseudo-inverse failed for final output weight calculation after PSO.")
        # This would be a critical failure. Might return None or raise an exception.
        return None, None, None

    final_output_weights = np.dot(H_opt_pinv, y_one_hot_train)

    return best_iw, best_b, final_output_weights

# --- Data Loading and Main Experiment Logic ---
def load_data(file_path):
    """Loads data from CSV file."""
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values.astype(np.float32)
    y = data.iloc[:, -1].values.astype(int)
    return X, y

def run_pso_elm_experiment(X_data, y_data, num_hidden, num_classes_exp, k_splits, random_state_exp):
    """
    Runs the PSO-ELM experiment using K-Fold cross-validation.
    """
    kf = KFold(n_splits=k_splits, shuffle=True, random_state=random_state_exp)
    fold_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_data)):
        print(f"Fold {fold + 1}/{k_splits} ...")
        start_time_fold = time.time()

        X_train_fold, X_test_fold = X_data[train_idx], X_data[test_idx]
        y_train_fold, y_test_fold = y_data[train_idx], y_data[test_idx]

        # Scale data: fit on training data for this fold, transform both train and test
        scaler = StandardScaler()
        X_train_scaled_fold = scaler.fit_transform(X_train_fold).astype(np.float32)
        X_test_scaled_fold = scaler.transform(X_test_fold).astype(np.float32)

        # Train ELM using PSO
        # Pass num_classes_exp to train_elm_with_pso for one-hot encoding
        input_w, biases_w, output_w = train_elm_with_pso(
            X_train_scaled_fold, y_train_fold, num_hidden, num_classes_exp
        )

        if input_w is None: # Check if PSO training failed
            print(f"Fold {fold+1} failed during PSO training. Skipping.")
            fold_accuracies.append(0.0) # Record as 0 accuracy for this fold
            continue

        # Predict on the test set for this fold
        y_pred_probabilities = predict_elm(X_test_scaled_fold, input_w, biases_w, output_w)
        y_pred_labels = np.argmax(y_pred_probabilities, axis=1)

        acc = accuracy_score(y_test_fold, y_pred_labels)
        fold_accuracies.append(acc)

        elapsed_time_fold = time.time() - start_time_fold
        print(f"  Fold {fold + 1} Accuracy: {acc:.4f} - Time: {elapsed_time_fold:.2f}s")

    return fold_accuracies

def main():
    """Main function to run the PSO-ELM experiment."""
    np.random.seed(RANDOM_STATE) # Ensure reproducibility for PSO and KFold shuffling

    # Create output directory if it doesn't exist (though not used for saving files yet)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Output directory created (or already exists): {OUTPUT_DIR}")

    X_main, y_main = load_data(DATASET_PATH)

    print(f"\nStarting PSO-ELM experiment with {N_HIDDEN_NEURONS_PSO} hidden neurons...")
    accuracies = run_pso_elm_experiment(
        X_main, y_main,
        num_hidden=N_HIDDEN_NEURONS_PSO,
        num_classes_exp=N_CLASSES,
        k_splits=KFOLD_SPLITS,
        random_state_exp=RANDOM_STATE
    )

    if accuracies: # Check if any folds completed successfully
        print(f"\n✅ Average Accuracy with PSO-ELM ({N_HIDDEN_NEURONS_PSO} neurons): {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    else:
        print("\n❌ PSO-ELM experiment completed with no successful folds.")

if __name__ == '__main__':
    main()

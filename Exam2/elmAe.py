import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pyswarms as ps
import time

np.random.seed(42)
n_classes = 6  # تعداد کلاس‌ها


def sigmoid(x): return 1 / (1 + np.exp(-x))


def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values.astype(np.float32)
    y = data.iloc[:, -1].values.astype(int)
    return X, y


def preprocess_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X).astype(np.float32)


def predict_elm(X, input_weights, biases, output_weights):
    H = sigmoid(np.dot(X, input_weights) + biases)
    return np.dot(H, output_weights)


def fitness_function(params, X_train, y_train, n_features, n_hidden):
    n_particles = params.shape[0]
    losses = []
    y_one_hot = np.eye(n_classes)[y_train]

    for i in range(n_particles):
        particle = params[i]
        iw = particle[:n_features * n_hidden].reshape(n_features, n_hidden)
        b = particle[n_features * n_hidden:].reshape(1, n_hidden)
        H = sigmoid(np.dot(X_train, iw) + b)
        H_pinv = np.linalg.pinv(H)
        beta = np.dot(H_pinv, y_one_hot)
        y_pred = np.dot(H, beta)
        acc = accuracy_score(np.argmax(y_one_hot, axis=1), np.argmax(y_pred, axis=1))
        losses.append(-acc)  # منفی چون باید کمینه شود
    return np.array(losses)


def train_elm_pso(X_train, y_train, n_hidden, n_iter=50, n_particles=15):
    n_samples, n_features = X_train.shape
    dim = n_features * n_hidden + n_hidden

    options = {'c1': 1.5, 'c2': 1.5, 'w': 0.6}
    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=dim, options=options)

    cost, best_params = optimizer.optimize(
        fitness_function, iters=n_iter,
        X_train=X_train, y_train=y_train,
        n_features=n_features, n_hidden=n_hidden
    )

    iw = best_params[:n_features * n_hidden].reshape(n_features, n_hidden)
    b = best_params[n_features * n_hidden:].reshape(1, n_hidden)
    H = sigmoid(np.dot(X_train, iw) + b)
    H_pinv = np.linalg.pinv(H)
    ow = np.dot(H_pinv, np.eye(n_classes)[y_train])
    return iw, b, ow


def run_elm_pso(X, y, n_hidden):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}/5 ...")
        start_time = time.time()

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_scaled = preprocess_data(X_train)
        X_test_scaled = preprocess_data(X_test)

        iw, b, ow = train_elm_pso(X_train_scaled, y_train, n_hidden=n_hidden)
        y_pred = predict_elm(X_test_scaled, iw, b, ow)
        y_pred_labels = np.argmax(y_pred, axis=1)

        acc = accuracy_score(y_test, y_pred_labels)
        accuracies.append(acc)

        print(f"  Accuracy: {acc:.4f} - Time: {time.time() - start_time:.2f}s")

    return accuracies


# استفاده
file_path = 'Dataset.csv'  # مسیر فایل CSV خود را جایگزین کن
X, y = load_data(file_path)
results = run_elm_pso(X, y, n_hidden=100)
print(f"\n✅ Average Accuracy with PSO-ELM: {np.mean(results):.4f}")

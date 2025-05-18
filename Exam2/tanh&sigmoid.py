import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time

# تنظیمات اولیه
np.random.seed(42)
n_hidden = 2000
rbf_width = 0.08
batch_size = 10000
n_classes = 6

# توابع فعال‌سازی
def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def hardlim(x):
    return np.where(x >= 0, 1, 0)

def gaussian(x, width=rbf_width):
    return np.exp(-(x**2) / (2 * width**2))

activation_functions = {
    'tanh': tanh,
    'sigmoid': sigmoid,
    'hardlim': hardlim,
    'gaussian': gaussian
}

# بارگذاری و نرمال‌سازی داده‌ها
def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

def preprocess_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

# آموزش ELM
def train_elm(X_train, y_train, n_hidden, activation_func):
    n_samples, n_features = X_train.shape
    input_weights = np.random.randn(n_features, n_hidden) * np.sqrt(2.0 / n_features)
    biases = np.random.randn(1, n_hidden)
    H = np.dot(X_train, input_weights) + biases
    H = activation_func(H)
    H_pinv = np.linalg.pinv(H)
    output_weights = np.dot(H_pinv, y_train)
    return input_weights, biases, output_weights

# پیش‌بینی ELM
def predict_elm(X_test, input_weights, biases, output_weights, activation_func):
    H = np.dot(X_test, input_weights) + biases
    H = activation_func(H)
    y_pred = np.dot(H, output_weights)
    return np.argmax(y_pred, axis=1)

# اجرای ELM با cross-validation
def run_elm(X, y, n_hidden, activation_func):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    y_one_hot = np.eye(n_classes)[y.astype(int)]

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_one_hot[train_idx], y[test_idx]
        X_train = preprocess_data(X_train)
        X_test = preprocess_data(X_test)

        input_weights, biases, output_weights = train_elm(X_train, y_train, n_hidden, activation_func)
        y_pred = predict_elm(X_test, input_weights, biases, output_weights, activation_func)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")
    return accuracies

# مسیر فایل CSV خود را اینجا وارد کنید
file_path = '/content/danger/Dataset.csv'
X, y = load_data(file_path)

# اجرای مدل با تمام توابع فعال‌سازی و ذخیره نتایج
results = {}
for name, func in activation_functions.items():
    print(f"\nRunning ELM with {name} activation function...")
    accs = run_elm(X, y, n_hidden, func)
    results[name] = accs

# رسم نمودار مقایسه‌ای
plt.figure(figsize=(10, 6))
for name, accs in results.items():
    plt.plot(accs, marker='o', label=f'{name} (Mean: {np.mean(accs):.4f})')

plt.title('Comparison of Activation Functions in ELM')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('activation_function_comparison.png')
plt.show()

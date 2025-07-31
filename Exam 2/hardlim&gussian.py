import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import time

# تنظیمات
np.random.seed(42)
n_hidden = 2000  # کاهش برای جلوگیری از کرش
rbf_width = 0.08

# اکتیویشن فانکشن‌ها
def tanh(x): return np.tanh(x)
def sigmoid(x): return 1 / (1 + np.exp(-x))
def hardlim(x): return np.where(x >= 0, 1.0, 0.0)
def gaussian(x): return np.exp(-x**2)

# بارگذاری و نرمال‌سازی داده
def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values.astype(np.float32)  # استفاده از float32
    y = data.iloc[:, -1].values
    return X, y

def preprocess_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X).astype(np.float32)

# آموزش ELM
def train_elm(X_train, y_train, n_hidden, activation_func):
    n_samples, n_features = X_train.shape
    input_weights = np.random.randn(n_features, n_hidden).astype(np.float32) * np.sqrt(2.0 / n_features)
    biases = np.random.randn(1, n_hidden).astype(np.float32)
    H = activation_func(np.dot(X_train, input_weights) + biases)
    H_pinv = np.linalg.pinv(H)
    output_weights = np.dot(H_pinv, y_train)
    return input_weights, biases, output_weights

def predict_elm(X_test, input_weights, biases, output_weights, activation_func):
    H_test = activation_func(np.dot(X_test, input_weights) + biases)
    y_pred = np.dot(H_test, output_weights)
    return np.argmax(y_pred, axis=1)

# اجرای ELM با اعتبارسنجی K-Fold
def run_elm(X, y, n_hidden, activation_func, name=''):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []

    y_one_hot = np.eye(6)[y.astype(int)]

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"[{name}] Fold {fold + 1}/5 ...")
        start_time = time.time()

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_one_hot[train_idx], y[test_idx]

        X_train_scaled = preprocess_data(X_train)
        X_test_scaled = preprocess_data(X_test)

        input_weights, biases, output_weights = train_elm(X_train_scaled, y_train, n_hidden, activation_func)
        y_pred = predict_elm(X_test_scaled, input_weights, biases, output_weights, activation_func)

        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

        print(f"  Accuracy: {acc:.4f} - Time: {time.time() - start_time:.2f}s")

    return accuracies

# مسیر داده‌ها
file_path = '/content/danger/Dataset.csv'
X, y = load_data(file_path)

# اجرای ELM برای دو تابع باقی‌مانده
results = {}
results['hardlim'] = run_elm(X, y, n_hidden, hardlim, name='hardlim')
results['gaussian'] = run_elm(X, y, n_hidden, gaussian, name='gaussian')

# رسم نمودار مقایسه‌ای
plt.figure(figsize=(10, 6))
for act_name, accs in results.items():
    plt.plot(range(1, 6), accs, marker='o', label=f'{act_name} (mean={np.mean(accs):.4f})')

plt.title("Comparison of Activation Functions on ELM")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("activation_function_comparison.png")
plt.show()

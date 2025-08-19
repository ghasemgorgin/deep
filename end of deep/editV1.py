import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from lightgbm import LGBMClassifier
from hpelm import ELM

# --- Wrapper برای ELM ---
class ELMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neurons=500, func='sigm'):
        self.n_neurons = n_neurons
        self.func = func
        self.model = None

    def fit(self, X, y):
        y_one_hot = pd.get_dummies(y).values
        self.model = ELM(X.shape[1], y_one_hot.shape[1], classification="c")
        self.model.add_neurons(self.n_neurons, self.func)
        self.model.train(X, y_one_hot, "c")
        return self

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=1)

# --- بارگذاری داده ---
data = pd.read_csv("../Exam 2/Dataset.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# --- تقسیم و نرمال‌سازی ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- مقادیر مختلف نورون‌ها برای بررسی ---
neuron_counts = [500, 1000]
results = []

for n in neuron_counts:
    print(f"\n⏳ Testing with {n} neurons...")
    start = time.time()

    elm = ELMWrapper(n_neurons=n)
    estimators = [
        ('elm', elm),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('lgbm', LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42))
    ]
    model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000))
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    t = time.time() - start

    print(f"✅ Neurons: {n} | Accuracy: {acc:.4f} | Time: {t:.2f} s")
    results.append((n, acc, t, y_pred, model))

# --- انتخاب بهترین مدل ---
best_n, best_acc, best_time, best_pred, best_model = max(results, key=lambda x: x[1])
print(f"\n🏆 Best Configuration: {best_n} neurons | Accuracy: {best_acc:.4f} | Time: {best_time:.2f} s")

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix (Best Model - {best_n} Neurons)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix99.png")
plt.close()

# --- نمودار مقایسه دقت ---
neuron_list, acc_list, time_list, _, _ = zip(*results)
plt.figure()
plt.plot(neuron_list, acc_list, marker='o')
plt.title("Accuracy vs Number of Neurons")
plt.xlabel("Number of Neurons in ELM")
plt.ylabel("Test Accuracy")
plt.grid()
plt.savefig("accuracy_vs_neurons.png")
plt.close()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from hpelm import ELM
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


import time




# تعریف Wrapper ساده برای ELM
class ELMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neurons=1000, func='sigm'):
        self.n_neurons = n_neurons
        self.func = func
        self.model = None

    def fit(self, X, y):
        # تبدیل برچسب‌ها به one-hot برای ELM
        y_one_hot = pd.get_dummies(y).values
        self.model = ELM(X.shape[1], y_one_hot.shape[1], classification="c")
        self.model.add_neurons(self.n_neurons, self.func)
        self.model.train(X, y_one_hot, "c")
        return self


    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=1)

    def get_params(self, deep=True):
        return {"n_neurons": self.n_neurons, "func": self.func}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

# بارگذاری دیتاست
data = pd.read_csv("dataset.csv")  # جایگزین با مسیر دیتاست شما
X = data.iloc[:, :-1].values  # 10 ستون ویژگی
y = data.iloc[:, -1].values  # ستون کلاس

# تقسیم داده‌ها به آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# استانداردسازی داده‌ها
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# تابع برای آزمایش تعداد نورون‌های مختلف
def evaluate_stacking(n_hidden):
    start_time = time.time()

    # تعریف مدل ELM با Wrapper
    elm = ELMWrapper(n_neurons=n_hidden, func="sigm")

    # تعریف مدل‌های پایه
    estimators = [
        ('elm', elm),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('lgbm', LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=100, random_state=42))
    ]

    # تعریف Stacking
    stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000))
    stacking.fit(X_train_scaled, y_train)
    y_pred = stacking.predict(X_test_scaled)

    # محاسبه دقت و زمان
    accuracy = accuracy_score(y_test, y_pred)
    elapsed_time = time.time() - start_time
    return accuracy, elapsed_time

# آزمایش تعداد نورون‌های مختلف
neuron_counts = [500, 1000, 1500, 2000]
results = []

for n in neuron_counts:
    acc, t = evaluate_stacking(n)
    results.append((n, acc, t))
    print(f"Neurons: {n}, Accuracy: {acc:.4f}, Time: {t:.2f} seconds")

# انتخاب بهترین تعداد نورون
best_result = max(results, key=lambda x: x[1])  # بر اساس دقت
best_neurons, best_accuracy, best_time = best_result
print(f"\nBest Configuration: Neurons={best_neurons}, Accuracy={best_accuracy:.4f}, Time={best_time:.2f} seconds")
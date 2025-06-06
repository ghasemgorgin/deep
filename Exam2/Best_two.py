import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from hpelm import ELM
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import time

# تعریف Wrapper ساده برای ELM
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

    def get_params(self, deep=True):
        return {"n_neurons": self.n_neurons, "func": self.func}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

# بارگذاری دیتاست
data = pd.read_csv("dataset.csv")  # جایگزین با مسیر دیتاست شما
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# تقسیم داده‌ها به آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# استانداردسازی داده‌ها
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# تعریف مدل‌های پایه
estimators = [
    ('elm', ELMWrapper(n_neurons=500, func='sigm')),
    ('rf', RandomForestClassifier(random_state=42)),
    ('lgbm', LGBMClassifier(random_state=42)),
    ('svm', SVC(probability=True, random_state=42))
]

# تعریف StackingClassifier
stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000)
)

# تعریف هایپرپارامترها برای RandomizedSearch
param_grid = {
    'rf__n_estimators': [50, 100, 200],
    'rf__max_depth': [10, 20, None],
    'lgbm__learning_rate': [0.01, 0.05, 0.1],
    'lgbm__num_leaves': [15, 31, 63],
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['rbf'],
    'final_estimator__C': [0.1, 1, 10]
}

# اجرای RandomizedSearchCV
start_time = time.time()
grid_search = RandomizedSearchCV(
    estimator=stacking,
    param_distributions=param_grid,
    n_iter=20,  # 20 ترکیب تصادفی
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

# بهترین مدل و پیش‌بینی
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
elapsed_time = time.time() - start_time

# چاپ نتایج
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Time: {elapsed_time:.2f} seconds")
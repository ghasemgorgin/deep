import estimators
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from hpelm import ELM
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import StackingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import time
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# ELM Wrapper
class ELMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neurons=2000, func='tanh', l2=1.0):
        self.n_neurons = n_neurons
        self.func = func
        self.l2 = l2
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        self.classes_ = np.unique(y)
        y_one_hot = pd.get_dummies(y).values
        self.model = ELM(X.shape[1], y_one_hot.shape[1], classification="c", batch=2000)
        self.model.add_neurons(self.n_neurons, self.func)
        self.model.train(X, y_one_hot, "c", l2=self.l2)
        return self

    def predict(self, X):
        X = np.asarray(X)
        return np.argmax(self.model.predict(X), axis=1)

    def predict_proba(self, X):
        X = np.asarray(X)
        proba = self.model.predict(X)
        proba = np.clip(proba, 0, None)
        proba_sum = proba.sum(axis=1, keepdims=True)
        proba_sum = np.where(proba_sum == 0, 1, proba_sum)
        return proba / proba_sum

    def get_params(self, deep=True):
        return {"n_neurons": self.n_neurons, "func": self.func, "l2": self.l2}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

# Load dataset
data = pd.read_csv("dataset.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1].values

# Select top 3 features for interaction terms
selector = SelectKBest(score_func=mutual_info_classif, k=3)
X_top = selector.fit_transform(X, y)
top_indices = selector.get_support(indices=True)
X_top = pd.DataFrame(X_top, columns=[f"feature_{i}" for i in top_indices])

# Add interaction terms for top 3 features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_top_poly = poly.fit_transform(X_top)
X_top_poly = pd.DataFrame(X_top_poly, columns=poly.get_feature_names_out())

# Combine original and interaction features
X = pd.concat([X, X_top_poly], axis=1)
selector = SelectKBest(score_func=mutual_info_classif, k=12)
X = selector.fit_transform(X, y)
X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Robust scaling
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

# Compute sample weights to reduce outlier impact
def compute_sample_weights(X):
    median = np.median(X, axis=0)
    mad = stats.median_abs_deviation(X, axis=0)
    mad = np.where(mad == 0, 1, mad)  # Avoid division by zero
    distances = np.abs(X - median) / mad
    distances = np.mean(distances, axis=1)
    weights = 1 / (1 + distances)  # Lower weights for outliers
    return np.clip(weights, 0.1, 1.0)

sample_weights = compute_sample_weights(X_train_scaled.values)

# Evaluate model with Stacking
def evaluate_stacking(n_hidden):
    start_time = time.time()

    # Base models
    estimators = [
        ('elm', ELMWrapper(n_neurons=n_hidden, func='tanh', l2=1.0)),
        ('xgb', XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, random_state=42)),
        ('lgbm', LGBMClassifier(n_estimators=300, num_leaves=50, learning_rate=0.1, random_state=42)),
        ('cat', CatBoostClassifier(iterations=300, depth=8, learning_rate=0.1, random_state=42, verbose=0))
    ]

    # Stacking
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=XGBClassifier(n_estimators=200, max_depth=7, learning_rate=0.1, random_state=42),
        cv=5,
        n_jobs=1
    )

    # Train ELM separately (no sample weights)
    elm = ELMWrapper(n_neurons=n_hidden, func='tanh', l2=1.0)
    elm.fit(X_train_scaled, y_train)

    # Train other models with sample weights
    for name, est in estimators:
        if name != 'elm':
            est.fit(X_train_scaled, y_train, sample_weight=sample_weights)

    # Train stacking (without sample weights for ELM compatibility)
    stacking.fit(X_train_scaled, y_train)

    # Test predictions
    y_pred = stacking.predict(X_test_scaled)

    # Second stage for problematic classes (0 and 1)
    problem_classes = [0, 1]
    problem_indices = np.isin(y_pred, problem_classes)
    X_test_problem = X_test_scaled[problem_indices]
    y_test_problem = y_test[problem_indices]
    probas_problem = stacking.predict_proba(X_test_scaled)[problem_indices]
    if len(np.unique(y_test_problem)) > 1:
        cat_second = CatBoostClassifier(iterations=300, depth=7, learning_rate=0.1, random_state=42, verbose=0)
        X_second = np.hstack([X_test_problem, probas_problem])
        X_train_second = np.hstack([X_train_scaled, stacking.predict_proba(X_train_scaled)])
        y_train_second = y_train
        train_problem_indices = np.isin(y_train, problem_classes)
        X_train_second = X_train_second[train_problem_indices]
        y_train_second = y_train[train_problem_indices]
        sample_weights_second = sample_weights[train_problem_indices]
        cat_second.fit(X_train_second, y_train_second, sample_weight=sample_weights_second)
        y_pred_second = cat_second.predict(X_second)
        y_pred[problem_indices] = y_pred_second.flatten()

    test_accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    elapsed_time = time.time() - start_time
    return test_accuracy, elapsed_time, cm, report

# Test and evaluate
neuron_counts = [2000, 3000]
results = []

for n in neuron_counts:
    acc, t, cm, report = evaluate_stacking(n)
    results.append((n, acc, t, cm, report))
    print(f"\nNeurons: {n}, Test Accuracy: {acc:.4f}, Time: {t:.2f} seconds")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Classification Report:\n{report}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix (Neurons={n})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'confusion_matrix_{n}.png')
    plt.close()

# Select best configuration
best_result = max(results, key=lambda x: x[1])
best_neurons, best_accuracy, best_time, best_cm, best_report = best_result
print(f"\nBest Configuration: Neurons={best_neurons}, Test Accuracy={best_accuracy:.4f}, Time={best_time:.2f} seconds")
print(f"Best Confusion Matrix:\n{best_cm}")
print(f"Best Classification Report:\n{best_report}")

plt.figure(figsize=(8, 6))
sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f'Best Confusion Matrix (Neurons={best_neurons})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('best_confusion_matrix.png')
plt.close()

# Check training accuracy to monitor overfitting
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_accuracies = []
for train_idx, val_idx in kf.split(X_train_scaled):
    X_tr = X_train_scaled.iloc[train_idx]
    y_tr = y_train[train_idx]
    X_val = X_train_scaled.iloc[val_idx]
    y_val = y_train[val_idx]
    sample_weights_tr = sample_weights[train_idx]
    # Train ELM separately
    elm = ELMWrapper(n_neurons=best_neurons, func='tanh', l2=1.0)
    elm.fit(X_tr, y_tr)
    # Train other models with sample weights
    for name, est in estimators:
        if name != 'elm':
            est.fit(X_tr, y_tr, sample_weight=sample_weights_tr)
    # Train stacking
    stacking.fit(X_tr, y_tr)
    y_val_pred = stacking.predict(X_val)
    train_acc = accuracy_score(y_val, y_val_pred)
    train_accuracies.append(train_acc)
print(f"Training Accuracy (mean): {np.mean(train_accuracies):.4f}")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from hpelm import ELM
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import time
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ایجاد پوشه برای ذخیره تصاویر
if not os.path.exists("resultstwo"):
    os.makedirs("resultstwo")


# تعریف Wrapper ساده برای ELM
class ELMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neurons=1000, func='sigm'):
        self.n_neurons = n_neurons
        self.func = func
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        try:
            # تبدیل DataFrame به آرایه NumPy
            if isinstance(X, pd.DataFrame):
                X = X.values
            y_one_hot = pd.get_dummies(y).values
            self.model = ELM(X.shape[1], y_one_hot.shape[1], classification="c")
            self.model.add_neurons(self.n_neurons, self.func)
            self.model.train(X, y_one_hot, "c")
            self.classes_ = np.unique(y)
            return self
        except Exception as e:
            raise ValueError(f"Error in ELMWrapper.fit: {str(e)}")

    def predict(self, X):
        try:
            if self.model is None:
                raise ValueError("Model is not fitted yet.")
            # تبدیل DataFrame به آرایه NumPy
            if isinstance(X, pd.DataFrame):
                X = X.values
            return np.argmax(self.model.predict(X), axis=1)
        except Exception as e:
            raise ValueError(f"Error in ELMWrapper.predict: {str(e)}")

    def predict_proba(self, X):
        try:
            if self.model is None:
                raise ValueError("Model is not fitted yet.")
            # تبدیل DataFrame به آرایه NumPy
            if isinstance(X, pd.DataFrame):
                X = X.values
            return self.model.predict(X)
        except Exception as e:
            raise ValueError(f"Error in ELMWrapper.predict_proba: {str(e)}")

    def get_params(self, deep=True):
        return {"n_neurons": self.n_neurons, "func": self.func}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


# بارگذاری دیتاست
try:
    data = pd.read_csv("../Exam 2/Dataset.csv")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
except FileNotFoundError:
    print("Dataset.csv not found. Please provide the correct path.")
    exit()

# بررسی توزیع کلاس‌ها
print("توزیع کلاس‌ها:")
print(pd.Series(y).value_counts())

# تقسیم داده‌ها به آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# استانداردسازی داده‌ها
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)


# تابع برای رسم و ذخیره ماتریس درهم‌ریختگی
def plot_confusion_matrix(cm, classes, title, filename):
    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title, fontsize=12)
    plt.ylabel('True Label', fontsize=10)
    plt.xlabel('Predicted Label', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join("resultstwo", filename), bbox_inches='tight')
    plt.close()


# تابع برای رسم و ذخیره جدول گزارش طبقه‌بندی
def plot_classification_report(report, title, filename):
    metrics = ['precision', 'recall', 'f1-score']
    classes = [str(i) for i in range(len(report) - 3)]
    data = [[report[c][m] for m in metrics] for c in classes]
    data = np.array(data).T

    fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=np.round(data, 4), rowLabels=metrics,
                     colLabels=[f"Class {i} (Emotion {int(i) + 1})" for i in classes],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.title(title, fontsize=12)
    plt.savefig(os.path.join("resultstwo", filename), bbox_inches='tight')
    plt.close()


# تابع برای رسم و ذخیره نمودار میله‌ای معیارها
def plot_metrics_bar(report, title, filename):
    metrics = ['precision', 'recall', 'f1-score']
    classes = [str(i) for i in range(len(report) - 3)]
    data = [[report[c][m] for c in classes] for m in metrics]

    x = np.arange(len(classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, data[i], width, label=metric.capitalize(), color=colors[i])
    ax.set_xlabel('Classes', fontsize=10)
    ax.set_ylabel('Score', fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"Class {i} (Emotion {int(i) + 1})" for i in classes], rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("resultstwo", filename), bbox_inches='tight')
    plt.close()


# تابع برای آزمایش تعداد نورون‌های مختلف
def evaluate_stacking(n_hidden):
    start_time = time.time()

    # تعریف مدل ELM با Wrapper
    elm = ELMWrapper(n_neurons=n_hidden, func="sigm")

    # تعریف مدل‌های پایه
    estimators = [
        ('elm', elm),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('lgbm', LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=100, random_state=42, verbose=-1))
    ]

    # تعریف Stacking
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        stack_method='predict'
    )

    # اعتبارسنجی متقاطع
    cv_scores = cross_val_score(stacking, X_train_scaled, y_train, cv=5, scoring='accuracy', error_score='raise')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    # آموزش مدل روی کل داده‌های آموزشی
    stacking.fit(X_train_scaled, y_train)
    y_pred = stacking.predict(X_test_scaled)

    # محاسبه دقت، گزارش طبقه‌بندی، و ماتریس درهم‌ریختگی
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    elapsed_time = time.time() - start_time

    # ذخیره گزارش‌ها به‌صورت عکس
    plot_confusion_matrix(conf_matrix, classes=[f"Emotion {i + 1}" for i in range(6)],
                          title=f'Confusion Matrix (Neurons={n_hidden})',
                          filename=f'confusion_matrix_{n_hidden}.png')
    plot_classification_report(report, title=f'Classification Report (Neurons={n_hidden})',
                               filename=f'classification_report_{n_hidden}.png')
    plot_metrics_bar(report, title=f'Metrics Comparison (Neurons={n_hidden})',
                     filename=f'metrics_bar_{n_hidden}.png')

    return accuracy, elapsed_time, cv_mean, cv_std, report, conf_matrix


# آزمایش تعداد نورون‌های مختلف
neuron_counts = [300,500]
resultstwo = []

for n in neuron_counts:
    try:
        acc, t, cv_mean, cv_std, report, conf_matrix = evaluate_stacking(n)
        resultstwo.append((n, acc, t, cv_mean, cv_std, report, conf_matrix))
        print(f"\nNeurons: {n}")
        print(f"Test Accuracy: {acc:.4f}")
        print(f"Cross-Validation Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"Time: {t:.2f} seconds")
        print("Classification Report:")
        for label in report:
            if label.isdigit():
                print(f"Class {label} (Emotion {int(label) + 1}):")
                print(f"  Precision: {report[label]['precision']:.4f}")
                print(f"  Recall: {report[label]['recall']:.4f}")
                print(f"  F1-Score: {report[label]['f1-score']:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)
    except Exception as e:
        print(f"Error evaluating with {n} neurons: {str(e)}")

# انتخاب بهترین تعداد نورون
best_result = max(resultstwo, key=lambda x: x[1], default=(0, 0.0, 0.0, 0.0, 0.0, {}, np.zeros((6, 6))))
best_neurons, best_accuracy, best_time, best_cv_mean, best_cv_std, best_report, best_conf_matrix = best_result
print("\n" + "=" * 50)
print(f"Best Configuration: Neurons={best_neurons}")
print(f"Test Accuracy: {best_accuracy:.4f}")
print(f"Cross-Validation Accuracy: {best_cv_mean:.4f} ± {best_cv_std:.4f}")
print(f"Time: {best_time:.2f} seconds")
print("\nDetailed Classification Report for the Best Model:")
for label in best_report:
    if label.isdigit():
        print(f"Class {label} (Emotion {int(label) + 1}):")
        print(f"  Precision: {best_report[label]['precision']:.4f}")
        print(f"  Recall: {best_report[label]['recall']:.4f}")
        print(f"  F1-Score: {best_report[label]['f1-score']:.4f}")
print("\nConfusion Matrix for the Best Model:")
print(best_conf_matrix)
print(f"\nMacro Avg Precision: {best_report.get('macro avg', {}).get('precision', 0.0):.4f}")
print(f"Weighted Avg Precision: {best_report.get('weighted avg', {}).get('precision', 0.0):.4f}")

# ذخیره نتایج عددی در فایل CSV
results_df = pd.DataFrame([
    {
        'Neurons': n,
        'Test_Accuracy': acc,
        'CV_Accuracy': cv_mean,
        'CV_Std': cv_std,
        'Time': t
    } for n, acc, t, cv_mean, cv_std, _, _ in resultstwo
])
results_df.to_csv(os.path.join("resultstwo", "resultstwo.csv"), index=False)
print("\nResults saved to 'resultstwo/resultstwo.csv' and images saved to 'resultstwo/' folder.")
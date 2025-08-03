# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
#
# from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.ensemble import RandomForestClassifier, StackingClassifier
# from sklearn.linear_model import LogisticRegression
# from hpelm import ELM
# from lightgbm import LGBMClassifier
# from sklearn.base import BaseEstimator, ClassifierMixin
# import time
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
#
#
# # تعریف Wrapper ساده برای ELM
# class ELMWrapper(BaseEstimator, ClassifierMixin):
#     def __init__(self, n_neurons=1000, func='sigm'):
#         self.n_neurons = n_neurons
#         self.func = func
#         self.model = None
#
#     def fit(self, X, y):
#         # تبدیل برچسب‌ها به one-hot برای ELM
#         y_one_hot = pd.get_dummies(y).values
#         self.model = ELM(X.shape[1], y_one_hot.shape[1], classification="c")
#         self.model.add_neurons(self.n_neurons, self.func)
#         self.model.train(X, y_one_hot, "c")
#         return self
#
#     def predict(self, X):
#         return np.argmax(self.model.predict(X), axis=1)
#
#     def get_params(self, deep=True):
#         return {"n_neurons": self.n_neurons, "func": self.func}
#
#     def set_params(self, **parameters):
#         for parameter, value in parameters.items():
#             setattr(self, parameter, value)
#         return self
#
#
# # بارگذاری دیتاست
#
# data = pd.read_csv("Dataset.csv")  # جایگزین با مسیر دیتاست شما
# X = data.iloc[:, :-1].values  # 10 ستون ویژگی
# y = data.iloc[:, -1].values  # ستون کلاس
#
# # تقسیم داده‌ها به آموزش و تست
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#
# # استانداردسازی داده‌ها
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
#
# # تابع برای آزمایش تعداد نورون‌های مختلف
# def evaluate_stacking(n_hidden):
#     start_time = time.time()
#
#     # تعریف مدل ELM با Wrapper
#     elm = ELMWrapper(n_neurons=n_hidden, func="sigm")
#
#     # تعریف مدل‌های پایه
#     estimators = [
#         ('elm', elm),
#         ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
#         ('lgbm', LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=100, random_state=42))
#     ]
#
#     # تعریف Stacking
#     stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000))
#     stacking.fit(X_train_scaled, y_train)
#     y_pred = stacking.predict(X_test_scaled)
#
#     # محاسبه دقت، زمان و ماتریس درهم‌ریختگی
#     accuracy = accuracy_score(y_test, y_pred)
#     cm = confusion_matrix(y_test, y_pred)
#     elapsed_time = time.time() - start_time
#     return accuracy, elapsed_time, cm
#
#
# # آزمایش تعداد نورون‌های مختلف
# neuron_counts = [500]
# results = []
#
# for n in neuron_counts:
#     acc, t, cm = evaluate_stacking(n)
#     results.append((n, acc, t, cm))
#     print(f"Neurons: {n}, Accuracy: {acc:.4f}, Time: {t:.2f} seconds")
#
#     # نمایش ماتریس درهم‌ریختگی
#     print(f"\nConfusion Matrix for {n} Neurons:\n{cm}")
#
#     # رسم نقشه گرمایی ماتریس درهم‌ریختگی
#     plt.figure(figsize=(6, 4))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
#     plt.title(f'Confusion Matrix (Neurons={n})')
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.show()
#
# # انتخاب بهترین تعداد نورون
# best_result = max(results, key=lambda x: x[1])  # بر اساس دقت
# best_neurons, best_accuracy, best_time, best_cm = best_result
# print(f"\nBest Configuration: Neurons={best_neurons}, Accuracy={best_accuracy:.4f}, Time={best_time:.2f} seconds")
# print(f"\nBest Confusion Matrix:\n{best_cm}")
#
# # رسم نقشه گرمایی برای بهترین پیکربندی
# plt.figure(figsize=(6, 4))
# sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
# plt.title(f'Best Confusion Matrix (Neurons={best_neurons})')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()

# second

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.ensemble import RandomForestClassifier, StackingClassifier
# from sklearn.linear_model import LogisticRegression
# from hpelm import ELM
# from lightgbm import LGBMClassifier
# from sklearn.base import BaseEstimator, ClassifierMixin
# import time
# import seaborn as sns
# import matplotlib
#
# # تنظیم backend برای جلوگیری از خطا
# matplotlib.use('Agg')  # یا 'TkAgg' بسته به محیط شما
# import matplotlib.pyplot as plt
#
#
# # تعریف Wrapper ساده برای ELM
# class ELMWrapper(BaseEstimator, ClassifierMixin):
#     def __init__(self, n_neurons=1000, func='sigm'):
#         self.n_neurons = n_neurons
#         self.func = func
#         self.model = None
#
#     def fit(self, X, y):
#         # تبدیل X به آرایه NumPy برای سازگاری با hpelm
#         X = np.asarray(X)
#         y_one_hot = pd.get_dummies(y).values
#         self.model = ELM(X.shape[1], y_one_hot.shape[1], classification="c")
#         self.model.add_neurons(self.n_neurons, self.func)
#         self.model.train(X, y_one_hot, "c")
#         return self
#
#     def predict(self, X):
#         # تبدیل X به آرایه NumPy برای پیش‌بینی
#         X = np.asarray(X)
#         return np.argmax(self.model.predict(X), axis=1)
#
#     def get_params(self, deep=True):
#         return {"n_neurons": self.n_neurons, "func": self.func}
#
#     def set_params(self, **parameters):
#         for parameter, value in parameters.items():
#             setattr(self, parameter, value)
#         return self
#
#
# # بارگذاری دیتاست
# data = pd.read_csv("Dataset.csv")  # جایگزین با مسیر دیتاست شما
# X = data.iloc[:, :-1]  # نگه داشتن DataFrame برای حفظ نام ستون‌ها
# y = data.iloc[:, -1].values  # ستون کلاس
#
# # تقسیم داده‌ها به آموزش و تست
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#
# # استانداردسازی داده‌ها
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# # تبدیل به DataFrame برای حفظ نام ستون‌ها برای LightGBM
# X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
# X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
#
#
# # تابع برای آزمایش تعداد نورون‌های مختلف
# def evaluate_stacking(n_hidden):
#     start_time = time.time()
#     elm = ELMWrapper(n_neurons=n_hidden, func="sigm")
#     estimators = [
#         ('elm', elm),
#         ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
#         ('lgbm', LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=100, random_state=42))
#     ]
#     stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000))
#     stacking.fit(X_train_scaled, y_train)
#     y_pred = stacking.predict(X_test_scaled)
#     accuracy = accuracy_score(y_test, y_pred)
#     cm = confusion_matrix(y_test, y_pred)
#     report = classification_report(y_test, y_pred)
#     elapsed_time = time.time() - start_time
#     return accuracy, elapsed_time, cm, report
#
#
# # آزمایش تعداد نورون‌های مختلف
# neuron_counts = [500]
# results = []
# class_labels = np.unique(y)  # برچسب‌های کلاس‌ها برای نمایش در نقشه گرمایی
#
# for n in neuron_counts:
#     acc, t, cm, report = evaluate_stacking(n)
#     results.append((n, acc, t, cm, report))
#     print(f"Neurons: {n}, Accuracy: {acc:.4f}, Time: {t:.2f} seconds")
#     print(f"\nConfusion Matrix for {n} Neurons:\n{cm}")
#     print(f"\nClassification Report for {n} Neurons:\n{report}")
#
#     # رسم نقشه گرمایی ماتریس درهم‌ریختگی
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
#                 xticklabels=class_labels, yticklabels=class_labels)
#     plt.title(f'Confusion Matrix (Neurons={n})')
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.savefig(f'confusion_matrix_{n}.png')  # ذخیره تصویر به جای نمایش مستقیم
#     plt.close()
#
# # انتخاب بهترین تعداد نورون
# best_result = max(results, key=lambda x: x[1])  # بر اساس دقت
# best_neurons, best_accuracy, best_time, best_cm, best_report = best_result
# print(f"\nBest Configuration: Neurons={best_neurons}, Accuracy={best_accuracy:.4f}, Time={best_time:.2f} seconds")
# print(f"\nBest Confusion Matrix:\n{best_cm}")
# print(f"\nBest Classification Report:\n{best_report}")
#
# # رسم نقشه گرمایی برای بهترین پیکربندی
# plt.figure(figsize=(8, 6))
# sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', cbar=False,
#             xticklabels=class_labels, yticklabels=class_labels)
# plt.title(f'Best Confusion Matrix (Neurons={best_neurons})')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.savefig('best_confusion_matrix.png')  # ذخیره تصویر
# plt.close()

# best code for now up
# resualt
# Confusion Matrix for 500 Neurons:
# [[6086   98  111   48   83   58]
#  [  96 5948   76   46   75  111]
#  [  96   84 5914   75   25   59]
#  [  80   49   75 5911   15   37]
#  [  62   63   21   39 6210   60]
#  [  63   84   58   32  117 6129]]
#
# Classification Report for 500 Neurons:
#               precision    recall  f1-score   support
#
#            0       0.94      0.94      0.94      6484
#            1       0.94      0.94      0.94      6352
#            2       0.95      0.95      0.95      6253
#            3       0.96      0.96      0.96      6167
#            4       0.95      0.96      0.96      6455
#            5       0.95      0.95      0.95      6483
#
#     accuracy                           0.95     38194
#    macro avg       0.95      0.95      0.95     38194
# weighted avg       0.95      0.95      0.95     38194
#
#
# Best Configuration: Neurons=500, Accuracy=0.9477, Time=285.23 seconds
#
# Best Confusion Matrix:
# [[6086   98  111   48   83   58]
#  [  96 5948   76   46   75  111]
#  [  96   84 5914   75   25   59]
#  [  80   49   75 5911   15   37]
#  [  62   63   21   39 6210   60]
#  [  63   84   58   32  117 6129]]
#
# Best Classification Report:
#               precision    recall  f1-score   support
#
#            0       0.94      0.94      0.94      6484
#            1       0.94      0.94      0.94      6352
#            2       0.95      0.95      0.95      6253
#            3       0.96      0.96      0.96      6167
#            4       0.95      0.96      0.96      6455
#            5       0.95      0.95      0.95      6483
#
#     accuracy                           0.95     38194
#    macro avg       0.95      0.95      0.95     38194
# weighted avg       0.95      0.95      0.95     38194


#third
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from hpelm import ELM
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import time
import seaborn as sns
import matplotlib

matplotlib.use('Agg')  # برای جلوگیری از خطاهای نمایش
import matplotlib.pyplot as plt


# تعریف Wrapper برای ELM
class ELMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neurons=1000, func='sigm'):
        self.n_neurons = n_neurons
        self.func = func
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        self.classes_ = np.unique(y)  # تنظیم ویژگی classes_
        y_one_hot = pd.get_dummies(y).values
        self.model = ELM(X.shape[1], y_one_hot.shape[1], classification="c", batch=1000)
        self.model.add_neurons(self.n_neurons, self.func)
        self.model.train(X, y_one_hot, "c")
        return self

    def predict(self, X):
        X = np.asarray(X)
        predictions = self.model.predict(X)
        return np.argmax(predictions, axis=1)

    def predict_proba(self, X):
        X = np.asarray(X)
        proba = self.model.predict(X)
        # نرمال‌سازی احتمالات
        proba = np.clip(proba, 0, None)  # اطمینان از غیرمنفی بودن
        proba_sum = proba.sum(axis=1, keepdims=True)
        proba_sum = np.where(proba_sum == 0, 1, proba_sum)  # جلوگیری از تقسیم بر صفر
        return proba / proba_sum

    def get_params(self, deep=True):
        return {"n_neurons": self.n_neurons, "func": self.func}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


# بارگذاری دیتاست
data = pd.read_csv("Dataset.csv")  # جایگزین با مسیر دیتاست شما
X = data.iloc[:, :-1]  # نگه داشتن DataFrame برای حفظ نام ستون‌ها
y = data.iloc[:, -1].values  # ستون کلاس

# تقسیم داده‌ها به آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# استانداردسازی داده‌ها
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)


# تابع برای آزمایش مدل با طبقه‌بندی دومرحله‌ای
def evaluate_stacking_with_second_stage(n_hidden):
    start_time = time.time()

    # مرحله اول: مدل انباشته اصلی
    elm = ELMWrapper(n_neurons=n_hidden, func="sigm")
    estimators = [
        ('elm', elm),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('lgbm', LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=100, random_state=42))
    ]
    stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000), n_jobs=1)
    stacking.fit(X_train_scaled, y_train)
    y_pred = stacking.predict(X_test_scaled)
    y_pred_proba = stacking.predict_proba(X_test_scaled)  # احتمالات برای مرحله دوم

    # شناسایی نمونه‌های مشکل‌ساز (کلاس‌های 1, 4, 5)
    problem_classes = [1, 4, 5]
    problem_indices = np.isin(y_pred, problem_classes)
    X_test_problem = X_test_scaled[problem_indices]
    y_test_problem = y_test[problem_indices]
    y_pred_problem = y_pred[problem_indices]
    probas_problem = y_pred_proba[problem_indices]  # احتمالات برای نمونه‌های مشکل‌ساز

    # بررسی توزیع کلاس‌ها در داده‌های مرحله دوم
    print("Class distribution in second stage training data:", np.bincount(y_train[np.isin(y_train, problem_classes)]))

    # مرحله دوم: آموزش ELM برای کلاس‌های 1, 4, 5
    if len(np.unique(y_test_problem)) > 1:  # بررسی وجود حداقل دو کلاس
        elm_second = ELMWrapper(n_neurons=1000, func="sigm")  # افزایش نورون‌ها
        # استفاده از ویژگی‌های اصلی + احتمالات مدل‌های پایه
        X_second = np.hstack([X_test_problem, probas_problem])
        X_train_second = np.hstack([X_train_scaled, stacking.predict_proba(X_train_scaled)])
        y_train_second = y_train
        # فیلتر کردن نمونه‌های آموزشی برای کلاس‌های 1, 4, 5
        train_problem_indices = np.isin(y_train, problem_classes)
        X_train_second = X_train_second[train_problem_indices]
        y_train_second = y_train[train_problem_indices]
        # نگاشت برچسب‌ها به مقادیر 0, 1, 2 برای کلاس‌های 1, 4, 5
        class_mapping = {1: 0, 4: 1, 5: 2}
        y_train_second_mapped = np.array([class_mapping[cls] for cls in y_train_second])
        # اعتبارسنجی متقاطع برای مدل دوم
        scores = cross_val_score(elm_second, X_train_second, y_train_second_mapped, cv=5, scoring='accuracy')
        print(f"Second Stage CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
        elm_second.fit(X_train_second, y_train_second_mapped)
        y_pred_second_mapped = elm_second.predict(X_second)
        # بازگرداندن به برچسب‌های اصلی
        reverse_mapping = {0: 1, 1: 4, 2: 5}
        y_pred_second = np.array([reverse_mapping[cls] for cls in y_pred_second_mapped])

        # به‌روزرسانی پیش‌بینی‌ها برای نمونه‌های مشکل‌ساز
        y_pred[problem_indices] = y_pred_second

    # محاسبه معیارها
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    elapsed_time = time.time() - start_time
    return accuracy, elapsed_time, cm, report


# آزمایش تعداد نورون‌های مختلف
neuron_counts = [700]
results = []
class_labels = np.unique(y)  # برچسب‌های کلاس‌ها

for n in neuron_counts:
    acc, t, cm, report = evaluate_stacking_with_second_stage(n)
    results.append((n, acc, t, cm, report))
    print(f"Neurons: {n}, Accuracy: {acc:.4f}, Time: {t:.2f} seconds")
    print(f"\nConfusion Matrix for {n} Neurons:\n{cm}")
    print(f"\nClassification Report for {n} Neurons:\n{report}")

    # رسم نقشه گرمایی
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Confusion Matrix (Neurons={n})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'confusion_matrix_{n}.png')
    plt.close()

# انتخاب بهترین پیکربندی
best_result = max(results, key=lambda x: x[1])  # بر اساس دقت
best_neurons, best_accuracy, best_time, best_cm, best_report = best_result
print(f"\nBest Configuration: Neurons={best_neurons}, Accuracy={best_accuracy:.4f}, Time={best_time:.2f} seconds")
print(f"\nBest Confusion Matrix:\n{best_cm}")
print(f"\nBest Classification Report:\n{best_report}")

# رسم نقشه گرمایی برای بهترین پیکربندی
plt.figure(figsize=(8, 6))
sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=class_labels, yticklabels=class_labels)
plt.title(f'Best Confusion Matrix (Neurons={best_neurons})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('best_confusion_matrix.png')
plt.close()
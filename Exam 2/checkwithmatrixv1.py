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
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split, cross_val_score
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
# matplotlib.use('Agg')  # برای جلوگیری از خطاهای نمایش
# import matplotlib.pyplot as plt
#
#
# # تعریف Wrapper برای ELM
# class ELMWrapper(BaseEstimator, ClassifierMixin):
#     def __init__(self, n_neurons=1000, func='sigm'):
#         self.n_neurons = n_neurons
#         self.func = func
#         self.model = None
#         self.classes_ = None
#
#     def fit(self, X, y):
#         X = np.asarray(X)
#         self.classes_ = np.unique(y)  # تنظیم ویژگی classes_
#         y_one_hot = pd.get_dummies(y).values
#         self.model = ELM(X.shape[1], y_one_hot.shape[1], classification="c", batch=1000)
#         self.model.add_neurons(self.n_neurons, self.func)
#         self.model.train(X, y_one_hot, "c")
#         return self
#
#     def predict(self, X):
#         X = np.asarray(X)
#         predictions = self.model.predict(X)
#         return np.argmax(predictions, axis=1)
#
#     def predict_proba(self, X):
#         X = np.asarray(X)
#         proba = self.model.predict(X)
#         # نرمال‌سازی احتمالات
#         proba = np.clip(proba, 0, None)  # اطمینان از غیرمنفی بودن
#         proba_sum = proba.sum(axis=1, keepdims=True)
#         proba_sum = np.where(proba_sum == 0, 1, proba_sum)  # جلوگیری از تقسیم بر صفر
#         return proba / proba_sum
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
# X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
# X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
#
#
# # تابع برای آزمایش مدل با طبقه‌بندی دومرحله‌ای
# def evaluate_stacking_with_second_stage(n_hidden):
#     start_time = time.time()
#
#     # مرحله اول: مدل انباشته اصلی
#     elm = ELMWrapper(n_neurons=n_hidden, func="sigm")
#     estimators = [
#         ('elm', elm),
#         ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
#         ('lgbm', LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=100, random_state=42))
#     ]
#     stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000), n_jobs=1)
#     stacking.fit(X_train_scaled, y_train)
#     y_pred = stacking.predict(X_test_scaled)
#     y_pred_proba = stacking.predict_proba(X_test_scaled)  # احتمالات برای مرحله دوم
#
#     # شناسایی نمونه‌های مشکل‌ساز (کلاس‌های 1, 4, 5)
#     problem_classes = [1, 4, 5]
#     problem_indices = np.isin(y_pred, problem_classes)
#     X_test_problem = X_test_scaled[problem_indices]
#     y_test_problem = y_test[problem_indices]
#     y_pred_problem = y_pred[problem_indices]
#     probas_problem = y_pred_proba[problem_indices]  # احتمالات برای نمونه‌های مشکل‌ساز
#
#     # بررسی توزیع کلاس‌ها در داده‌های مرحله دوم
#     print("Class distribution in second stage training data:", np.bincount(y_train[np.isin(y_train, problem_classes)]))
#
#     # مرحله دوم: آموزش ELM برای کلاس‌های 1, 4, 5
#     if len(np.unique(y_test_problem)) > 1:  # بررسی وجود حداقل دو کلاس
#         elm_second = ELMWrapper(n_neurons=1000, func="sigm")  # افزایش نورون‌ها
#         # استفاده از ویژگی‌های اصلی + احتمالات مدل‌های پایه
#         X_second = np.hstack([X_test_problem, probas_problem])
#         X_train_second = np.hstack([X_train_scaled, stacking.predict_proba(X_train_scaled)])
#         y_train_second = y_train
#         # فیلتر کردن نمونه‌های آموزشی برای کلاس‌های 1, 4, 5
#         train_problem_indices = np.isin(y_train, problem_classes)
#         X_train_second = X_train_second[train_problem_indices]
#         y_train_second = y_train[train_problem_indices]
#         # نگاشت برچسب‌ها به مقادیر 0, 1, 2 برای کلاس‌های 1, 4, 5
#         class_mapping = {1: 0, 4: 1, 5: 2}
#         y_train_second_mapped = np.array([class_mapping[cls] for cls in y_train_second])
#         # اعتبارسنجی متقاطع برای مدل دوم
#         scores = cross_val_score(elm_second, X_train_second, y_train_second_mapped, cv=5, scoring='accuracy')
#         print(f"Second Stage CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
#         elm_second.fit(X_train_second, y_train_second_mapped)
#         y_pred_second_mapped = elm_second.predict(X_second)
#         # بازگرداندن به برچسب‌های اصلی
#         reverse_mapping = {0: 1, 1: 4, 2: 5}
#         y_pred_second = np.array([reverse_mapping[cls] for cls in y_pred_second_mapped])
#
#         # به‌روزرسانی پیش‌بینی‌ها برای نمونه‌های مشکل‌ساز
#         y_pred[problem_indices] = y_pred_second
#
#     # محاسبه معیارها
#     accuracy = accuracy_score(y_test, y_pred)
#     cm = confusion_matrix(y_test, y_pred)
#     report = classification_report(y_test, y_pred, zero_division=0)
#     elapsed_time = time.time() - start_time
#     return accuracy, elapsed_time, cm, report
#
#
# # آزمایش تعداد نورون‌های مختلف
# neuron_counts = [500]
# results = []
# class_labels = np.unique(y)  # برچسب‌های کلاس‌ها
#
# for n in neuron_counts:
#     acc, t, cm, report = evaluate_stacking_with_second_stage(n)
#     results.append((n, acc, t, cm, report))
#     print(f"Neurons: {n}, Accuracy: {acc:.4f}, Time: {t:.2f} seconds")
#     print(f"\nConfusion Matrix for {n} Neurons:\n{cm}")
#     print(f"\nClassification Report for {n} Neurons:\n{report}")
#
#     # رسم نقشه گرمایی
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
#                 xticklabels=class_labels, yticklabels=class_labels)
#     plt.title(f'Confusion Matrix (Neurons={n})')
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.savefig(f'confusion_matrix_{n}.png')
#     plt.close()
#
# # انتخاب بهترین پیکربندی
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
# plt.savefig('best_confusion_matrix.png')
# plt.close()

#res
# Class distribution in second stage training data: [    0 25407     0     0 25820 25933]
# Second Stage CV Accuracy: 0.9981 ± 0.0002
# Neurons: 700, Accuracy: 0.9465, Time: 400.64 seconds
#
# Confusion Matrix for 700 Neurons:
# [[6092   96  106   49   79   62]
#  [  97 5936   78   45   75  121]
#  [  96   83 5907   80   22   65]
#  [  86   48   73 5907   14   39]
#  [  65   62   22   38 6177   91]
#  [  69   77   57   31  116 6133]]
#
# Classification Report for 700 Neurons:
#               precision    recall  f1-score   support
#
#            0       0.94      0.94      0.94      6484
#            1       0.94      0.93      0.94      6352
#            2       0.95      0.94      0.95      6253
#            3       0.96      0.96      0.96      6167
#            4       0.95      0.96      0.95      6455
#            5       0.94      0.95      0.94      6483
#
#     accuracy                           0.95     38194
#    macro avg       0.95      0.95      0.95     38194
# weighted avg       0.95      0.95      0.95     38194
#
#
# Best Configuration: Neurons=700, Accuracy=0.9465, Time=400.64 seconds
#
# Best Confusion Matrix:
# [[6092   96  106   49   79   62]
#  [  97 5936   78   45   75  121]
#  [  96   83 5907   80   22   65]
#  [  86   48   73 5907   14   39]
#  [  65   62   22   38 6177   91]
#  [  69   77   57   31  116 6133]]
#
# Best Classification Report:
#               precision    recall  f1-score   support
#
#            0       0.94      0.94      0.94      6484
#            1       0.94      0.93      0.94      6352
#            2       0.95      0.94      0.95      6253
#            3       0.96      0.96      0.96      6167
#            4       0.95      0.96      0.95      6455
#            5       0.94      0.95      0.94      6483
#
#     accuracy                           0.95     38194
#    macro avg       0.95      0.95      0.95     38194
# weighted avg       0.95      0.95      0.95     38194

#fourth
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ELM Wrapper
class ELMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neurons=1000, func='sigm'):
        self.n_neurons = n_neurons
        self.func = func
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        self.classes_ = np.unique(y)
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
        proba = np.clip(proba, 0, None)
        proba_sum = proba.sum(axis=1, keepdims=True)
        proba_sum = np.where(proba_sum == 0, 1, proba_sum)
        return proba / proba_sum

    def get_params(self, deep=True):
        return {"n_neurons": self.n_neurons, "func": self.func}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

# Load dataset
data = pd.read_csv("Dataset.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Evaluate model with 5-fold CV and second-stage classifier
def evaluate_stacking_with_second_stage_cv(n_hidden):
    start_time = time.time()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_accuracies = []
    val_accuracies = []
    test_accuracy = None
    best_cm = None
    best_report = None
    class_labels = np.unique(y)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled, y_train)):
        print(f"\nFold {fold + 1}")
        X_tr = X_train_scaled.iloc[train_idx]
        y_tr = y_train[train_idx]
        X_val = X_train_scaled.iloc[val_idx]
        y_val = y_train[val_idx]

        # Stage 1: Stacking classifier
        elm = ELMWrapper(n_neurons=n_hidden, func="sigm")
        estimators = [
            ('elm', elm),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('lgbm', LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=100, random_state=42))
        ]
        stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000), n_jobs=1)
        stacking.fit(X_tr, y_tr)

        # Training and validation predictions
        y_train_pred = stacking.predict(X_tr)
        y_val_pred = stacking.predict(X_val)
        train_acc = accuracy_score(y_tr, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        print(f"Fold {fold + 1} - Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}")

        # Stage 2: ELM for problematic classes
        problem_classes = [1, 4, 5]
        problem_indices = np.isin(y_val_pred, problem_classes)
        X_val_problem = X_val[problem_indices]
        y_val_problem = y_val[problem_indices]
        y_pred_problem = y_val_pred[problem_indices]
        probas_problem = stacking.predict_proba(X_val)[problem_indices]

        if len(np.unique(y_val_problem)) > 1:
            elm_second = ELMWrapper(n_neurons=1000, func="sigm")
            X_second = np.hstack([X_val_problem, probas_problem])
            X_train_second = np.hstack([X_tr, stacking.predict_proba(X_tr)])
            y_train_second = y_tr
            train_problem_indices = np.isin(y_tr, problem_classes)
            X_train_second = X_train_second[train_problem_indices]
            y_train_second = y_tr[train_problem_indices]
            class_mapping = {1: 0, 4: 1, 5: 2}
            y_train_second_mapped = np.array([class_mapping[cls] for cls in y_train_second])
            elm_second.fit(X_train_second, y_train_second_mapped)
            y_pred_second_mapped = elm_second.predict(X_second)
            reverse_mapping = {0: 1, 1: 4, 2: 5}
            y_pred_second = np.array([reverse_mapping[cls] for cls in y_pred_second_mapped])
            y_val_pred[problem_indices] = y_pred_second
            val_acc = accuracy_score(y_val, y_val_pred)
            print(f"Fold {fold + 1} - Updated Validation Accuracy after Stage 2: {val_acc:.4f}")
            val_accuracies[-1] = val_acc

        # For the last fold, compute test set metrics
        if fold == 4:
            y_pred = stacking.predict(X_test_scaled)
            y_pred_proba = stacking.predict_proba(X_test_scaled)
            problem_indices = np.isin(y_pred, problem_classes)
            X_test_problem = X_test_scaled[problem_indices]
            y_test_problem = y_test[problem_indices]
            probas_problem = y_pred_proba[problem_indices]
            if len(np.unique(y_test_problem)) > 1:
                X_second = np.hstack([X_test_problem, probas_problem])
                y_pred_second_mapped = elm_second.predict(X_second)
                y_pred_second = np.array([reverse_mapping[cls] for cls in y_pred_second_mapped])
                y_pred[problem_indices] = y_pred_second
            test_accuracy = accuracy_score(y_test, y_pred)
            best_cm = confusion_matrix(y_test, y_pred)
            best_report = classification_report(y_test, y_pred, zero_division=0)

    # Overfitting check
    mean_train_acc = np.mean(train_accuracies)
    mean_val_acc = np.mean(val_accuracies)
    print(f"\nMean Train Accuracy: {mean_train_acc:.4f}")
    print(f"Mean Validation Accuracy: {mean_val_acc:.4f}")
    if mean_train_acc - mean_val_acc > 0.1:
        print("Warning: Potential overfitting detected (Train-Val gap > 0.1)")
    else:
        print("No significant overfitting detected")

    elapsed_time = time.time() - start_time
    return test_accuracy, elapsed_time, best_cm, best_report, class_labels

# Test different neuron counts
neuron_counts = [500]
results = []

for n in neuron_counts:
    acc, t, cm, report, class_labels = evaluate_stacking_with_second_stage_cv(n)
    results.append((n, acc, t, cm, report))
    print(f"\nNeurons: {n}, Test Accuracy: {acc:.4f}, Time: {t:.2f} seconds")
    print(f"Confusion Matrix for {n} Neurons:\n{cm}")
    print(f"Classification Report for {n} Neurons:\n{report}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_labels, yticklabels=class_labels)
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

# Plot best confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=class_labels, yticklabels=class_labels)
plt.title(f'Best Confusion Matrix (Neurons={best_neurons})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('best_confusion_matrix.png')
plt.close()
#res


#Fold 1 - Train Accuracy: 0.9988, Validation Accuracy: 0.9415
# Fold 1 - Updated Validation Accuracy after Stage 2: 0.9411

#Fold 2 - Train Accuracy: 0.9988, Validation Accuracy: 0.9432
# Fold 2 - Updated Validation Accuracy after Stage 2: 0.9430

# Fold 3 - Train Accuracy: 0.9987, Validation Accuracy: 0.9430
# Fold 3 - Updated Validation Accuracy after Stage 2: 0.9425





# Fold 4 - Train Accuracy: 0.9989, Validation Accuracy: 0.9440
# Fold 4 - Updated Validation Accuracy after Stage 2: 0.9437

# Fold 5 - Train Accuracy: 0.9988, Validation Accuracy: 0.9464
# Fold 5 - Updated Validation Accuracy after Stage 2: 0.9461
#
# Mean Train Accuracy: 0.9988
# Mean Validation Accuracy: 0.9433
# No significant overfitting detected
#
# Neurons: 500, Test Accuracy: 0.9435, Time: 1606.78 seconds
# Confusion Matrix for 500 Neurons:
# [[6069  104  108   58   86   59]
#  [  94 5917   81   53   77  130]
#  [ 106   82 5883   89   26   67]
#  [  78   60   82 5898   16   33]
#  [  63   91   33   37 6147   84]
#  [  62   95   63   28  112 6123]]
# Classification Report for 500 Neurons:
#               precision    recall  f1-score   support
#
#            0       0.94      0.94      0.94      6484
#            1       0.93      0.93      0.93      6352
#            2       0.94      0.94      0.94      6253
#            3       0.96      0.96      0.96      6167
#            4       0.95      0.95      0.95      6455
#            5       0.94      0.94      0.94      6483
#
#     accuracy                           0.94     38194
#    macro avg       0.94      0.94      0.94     38194
# weighted avg       0.94      0.94      0.94     38194
#
#
# Best Configuration: Neurons=500, Test Accuracy=0.9435, Time=1606.78 seconds
# Best Confusion Matrix:
# [[6069  104  108   58   86   59]
#  [  94 5917   81   53   77  130]
#  [ 106   82 5883   89   26   67]
#  [  78   60   82 5898   16   33]
#  [  63   91   33   37 6147   84]
#  [  62   95   63   28  112 6123]]
# Best Classification Report:
#               precision    recall  f1-score   support
#
#            0       0.94      0.94      0.94      6484
#            1       0.93      0.93      0.93      6352
#            2       0.94      0.94      0.94      6253
#            3       0.96      0.96      0.96      6167
#            4       0.95      0.95      0.95      6455
#            5       0.94      0.94      0.94      6483
#
#     accuracy                           0.94     38194
#    macro avg       0.94      0.94      0.94     38194
# weighted avg       0.94      0.94      0.94     38194

#fifth
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split, KFold
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
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
#
# # ELM Wrapper
# class ELMWrapper(BaseEstimator, ClassifierMixin):
#     def __init__(self, n_neurons=1000, func='sigm', kernel=None, rbf_width=None):
#         self.n_neurons = n_neurons
#         self.func = func
#         self.kernel = kernel
#         self.rbf_width = rbf_width
#         self.model = None
#         self.classes_ = None
#
#     def fit(self, X, y):
#         X = np.asarray(X)
#         self.classes_ = np.unique(y)
#         y_one_hot = pd.get_dummies(y).values
#         if self.kernel == 'rbf':
#             self.model = ELM(X.shape[1], y_one_hot.shape[1], classification="wc", batch=1000, wfunc=self.rbf_width)
#         else:
#             self.model = ELM(X.shape[1], y_one_hot.shape[1], classification="c", batch=1000)
#         self.model.add_neurons(self.n_neurons, self.func)
#         self.model.train(X, y_one_hot, "c")
#         return self
#
#     def predict(self, X):
#         X = np.asarray(X)
#         predictions = self.model.predict(X)
#         return np.argmax(predictions, axis=1)
#
#     def predict_proba(self, X):
#         X = np.asarray(X)
#         proba = self.model.predict(X)
#         proba = np.clip(proba, 0, None)
#         proba_sum = proba.sum(axis=1, keepdims=True)
#         proba_sum = np.where(proba_sum == 0, 1, proba_sum)
#         return proba / proba_sum
#
#     def get_params(self, deep=True):
#         return {"n_neurons": self.n_neurons, "func": self.func, "kernel": self.kernel, "rbf_width": self.rbf_width}
#
#     def set_params(self, **parameters):
#         for parameter, value in parameters.items():
#             setattr(self, parameter, value)
#         return self
#
# # Load dataset
# data = pd.read_csv("Dataset.csv")
# X = data.iloc[:, :-1]
# y = data.iloc[:, -1].values
#
# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#
# # Standardize data
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
# X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
#
# # Evaluate model with 5-fold CV and second-stage classifier
# def evaluate_stacking_with_second_stage_cv(n_neurons, func, kernel=None, rbf_width=None):  # Changed n_hidden to n_neurons
#     start_time = time.time()
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     train_accuracies = []
#     val_accuracies = []
#     test_accuracy = None
#     best_cm = None
#     best_report = None
#     class_labels = np.unique(y)
#
#     for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled, y_train)):
#         print(f"\nFold {fold + 1}")
#         X_tr = X_train_scaled.iloc[train_idx]
#         y_tr = y_train[train_idx]
#         X_val = X_train_scaled.iloc[val_idx]
#         y_val = y_train[val_idx]
#
#         # Stage 1: Stacking classifier
#         elm = ELMWrapper(n_neurons=n_neurons, func=func, kernel=kernel, rbf_width=rbf_width)  # Changed n_hidden to n_neurons
#         estimators = [
#             ('elm', elm),
#             ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
#             ('lgbm', LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=100, random_state=42))
#         ]
#         stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000), n_jobs=1)
#         stacking.fit(X_tr, y_tr)
#
#         # Training and validation predictions
#         y_train_pred = stacking.predict(X_tr)
#         y_val_pred = stacking.predict(X_val)
#         train_acc = accuracy_score(y_tr, y_train_pred)
#         val_acc = accuracy_score(y_val, y_val_pred)
#         train_accuracies.append(train_acc)
#         val_accuracies.append(val_acc)
#         print(f"Fold {fold + 1} - Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}")
#
#         # Stage 2: ELM for problematic classes
#         problem_classes = [1, 4, 5]
#         problem_indices = np.isin(y_val_pred, problem_classes)
#         X_val_problem = X_val[problem_indices]
#         y_val_problem = y_val[problem_indices]
#         probas_problem = stacking.predict_proba(X_val)[problem_indices]
#
#         if len(np.unique(y_val_problem)) > 1:
#             elm_second = ELMWrapper(n_neurons=1000, func=func, kernel=kernel, rbf_width=rbf_width)
#             X_second = np.hstack([X_val_problem, probas_problem])
#             X_train_second = np.hstack([X_tr, stacking.predict_proba(X_tr)])
#             y_train_second = y_tr
#             train_problem_indices = np.isin(y_tr, problem_classes)
#             X_train_second = X_train_second[train_problem_indices]
#             y_train_second = y_tr[train_problem_indices]
#             class_mapping = {1: 0, 4: 1, 5: 2}
#             y_train_second_mapped = np.array([class_mapping[cls] for cls in y_train_second])
#             elm_second.fit(X_train_second, y_train_second_mapped)
#             y_pred_second_mapped = elm_second.predict(X_second)
#             reverse_mapping = {0: 1, 1: 4, 2: 5}
#             y_pred_second = np.array([reverse_mapping[cls] for cls in y_pred_second_mapped])
#             y_val_pred[problem_indices] = y_pred_second
#             val_acc = accuracy_score(y_val, y_val_pred)
#             print(f"Fold {fold + 1} - Updated Validation Accuracy after Stage 2: {val_acc:.4f}")
#             val_accuracies[-1] = val_acc
#
#         # For the last fold, compute test set metrics
#         if fold == 4:
#             y_pred = stacking.predict(X_test_scaled)
#             y_pred_proba = stacking.predict_proba(X_test_scaled)
#             problem_indices = np.isin(y_pred, problem_classes)
#             X_test_problem = X_test_scaled[problem_indices]
#             y_test_problem = y_test[problem_indices]
#             probas_problem = y_pred_proba[problem_indices]
#             if len(np.unique(y_test_problem)) > 1:
#                 X_second = np.hstack([X_test_problem, probas_problem])
#                 y_pred_second_mapped = elm_second.predict(X_second)
#                 y_pred_second = np.array([reverse_mapping[cls] for cls in y_pred_second_mapped])
#                 y_pred[problem_indices] = y_pred_second
#             test_accuracy = accuracy_score(y_test, y_pred)
#             best_cm = confusion_matrix(y_test, y_pred)
#             best_report = classification_report(y_test, y_pred, zero_division=0)
#
#     # Overfitting check
#     mean_train_acc = np.mean(train_accuracies)
#     mean_val_acc = np.mean(val_accuracies)
#     print(f"\nMean Train Accuracy: {mean_train_acc:.4f}")
#     print(f"Mean Validation Accuracy: {mean_val_acc:.4f}")
#     if mean_train_acc - mean_val_acc > 0.1:
#         print("Warning: Potential overfitting detected (Train-Val gap > 0.1)")
#     else:
#         print("No significant overfitting detected")
#
#     elapsed_time = time.time() - start_time
#     return test_accuracy, elapsed_time, best_cm, best_report, class_labels
#
# # Test different configurations
# configs = [
#     {'n_neurons': 500, 'func': 'sigm', 'kernel': None, 'rbf_width': None},
#     # {'n_neurons': 1000, 'func': 'sigm', 'kernel': None, 'rbf_width': None},
#     # {'n_neurons': 2000, 'func': 'sigm', 'kernel': None, 'rbf_width': None},
#     {'n_neurons': 500, 'func': 'tanh', 'kernel': None, 'rbf_width': None},
#     # {'n_neurons': 1000, 'func': 'tanh', 'kernel': None, 'rbf_width': None},
#     # {'n_neurons': 2000, 'func': 'tanh', 'kernel': None, 'rbf_width': None},
#     {'n_neurons': 500, 'func': 'gaus', 'kernel': None, 'rbf_width': None},
#     # {'n_neurons': 1000, 'func': 'gaus', 'kernel': None, 'rbf_width': None},
#     # {'n_neurons': 2000, 'func': 'gaus', 'kernel': None, 'rbf_width': None},
#     {'n_neurons': 500, 'func': 'sigm', 'kernel': 'rbf', 'rbf_width': 0.01},
#     # {'n_neurons': 1000, 'func': 'sigm', 'kernel': 'rbf', 'rbf_width': 0.01},
#     # {'n_neurons': 2000, 'func': 'sigm', 'kernel': 'rbf', 'rbf_width': 0.01},
#     {'n_neurons': 500, 'func': 'sigm', 'kernel': 'rbf', 'rbf_width': 0.08}
#     # {'n_neurons': 1000, 'func': 'sigm', 'kernel': 'rbf', 'rbf_width': 0.08},
#     # {'n_neurons': 2000, 'func': 'sigm', 'kernel': 'rbf', 'rbf_width': 0.08}
# ]
# results = []
#
# for config in configs:
#     acc, t, cm, report, class_labels = evaluate_stacking_with_second_stage_cv(**config)
#     results.append((config, acc, t, cm, report))
#     config_str = f"Neurons={config['n_neurons']}_Func={config['func']}"
#     if config['kernel'] == 'rbf':
#         config_str += f"_RBFWidth={config['rbf_width']}"
#     print(f"\nConfig: {config_str}, Test Accuracy: {acc:.4f}, Time: {t:.2f} seconds")
#     print(f"Confusion Matrix for {config_str}:\n{cm}")
#     print(f"Classification Report for {config_str}:\n{report}")
#
#     # Plot confusion matrix
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
#                 xticklabels=class_labels, yticklabels=class_labels)
#     plt.title(f'Confusion Matrix ({config_str})')
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.savefig(f'confusion_matrix_{config_str}.png')
#     plt.close()
#
# # Select best configuration
# best_result = max(results, key=lambda x: x[1])
# best_config, best_accuracy, best_time, best_cm, best_report = best_result
# best_config_str = f"Neurons={best_config['n_neurons']}_Func={best_config['func']}"
# if best_config['kernel'] == 'rbf':
#     best_config_str += f"_RBFWidth={best_config['rbf_width']}"
# print(f"\nBest Configuration: {best_config_str}, Test Accuracy: {best_accuracy:.4f}, Time: {best_time:.2f} seconds")
# print(f"Best Confusion Matrix:\n{best_cm}")
# print(f"Best Classification Report:\n{best_report}")
#
# # Plot best confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', cbar=False,
#             xticklabels=class_labels, yticklabels=class_labels)
# plt.title(f'Best Confusion Matrix ({best_config_str})')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.savefig('best_confusion_matrix.png')
# plt.close()
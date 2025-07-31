#lighbm and ELM
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score
# from hpelm import ELM
# from lightgbm import LGBMClassifier
# import time
#
# # بارگذاری دیتاست
# data = pd.read_csv('Dataset.csv')  # مسیر فایل
# X = data.iloc[:, :-1].values  # 10 ستون ویژگی
# y = data.iloc[:, -1].values  # ستون کلاس
#
# # تبدیل برچسب‌ها به one-hot encoding
# y_one_hot = pd.get_dummies(y).values  # تبدیل به one-hot
#
# # تقسیم داده‌ها به آموزش و تست
# X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42, stratify=y_one_hot)
# y_train_labels = np.argmax(y_train, axis=1)  # برچسب‌های اصلی برای LightGBM
# y_test_labels = np.argmax(y_test, axis=1)  # برچسب‌های اصلی برای ارزیابی
#
# # استانداردسازی داده‌ها
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
#
# # تابع برای آزمایش تعداد نورون‌های مختلف
# def evaluate_elm_lgbm(n_hidden: int) -> tuple[float, float]:
#     start_time = time.time()
#
#     # آموزش ELM (روی CPU)
#     elm = ELM(X_train_scaled.shape[1], 6, classification="c")  # 6 کلاس
#     elm.add_neurons(n_hidden, "sigm")  # تابع فعال‌سازی sigmoid
#     elm.train(X_train_scaled, y_train, "c")  # استفاده از y_train one-hot
#     elm_features_train = elm.project(X_train_scaled)  # ویژگی‌های استخراج‌شده
#     elm_features_test = elm.project(X_test_scaled)
#
#     # آموزش LightGBM (روی GPU)
#     lgbm = LGBMClassifier(
#         num_leaves=31,
#         learning_rate=0.05,
#         n_estimators=100,
#         random_state=42,
#         device="gpu",  # فعال‌سازی GPU
#         gpu_platform_id=0,  # پلتفرم GPU (اختیاری)
#         gpu_device_id=0,  # دستگاه GPU (اختیاری)
#         subsample=0.8,  # کاهش نمونه برای سرعت بیشتر
#         colsample_bytree=0.8  # کاهش ویژگی‌ها برای سرعت بیشتر
#     )
#     lgbm.fit(elm_features_train, y_train_labels)
#     y_pred = lgbm.predict(elm_features_test)
#
#     # محاسبه دقت و زمان
#     accuracy = accuracy_score(y_test_labels, y_pred)
#     elapsed_time = time.time() - start_time
#     return accuracy, elapsed_time
#
#
# # آزمایش تعداد نورون‌های مختلف (بهینه‌سازی ساده برای سرعت)
# neuron_counts = [500, 1000, 1500]  # کاهش تعداد برای سرعت بیشتر
# results = []
#
# for n in neuron_counts:
#     acc, t = evaluate_elm_lgbm(n)
#     results.append((n, acc, t))
#     print(f"Neurons: {n}, Accuracy: {acc:.4f}, Time: {t:.2f} seconds")
#
# # انتخاب بهترین تعداد نورون
# best_result = max(results, key=lambda x: x[1])  # بر اساس دقت
# best_neurons, best_accuracy, best_time = best_result
# print(f"\nBest Configuration: Neurons={best_neurons}, Accuracy={best_accuracy:.4f}, Time={best_time:.2f} seconds")


# stacking + ELM #بهترین

import numpy as np




#f1 score بالایی

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
#     classification_report
# from sklearn.ensemble import RandomForestClassifier, StackingClassifier
# from sklearn.linear_model import LogisticRegression
# from hpelm import ELM
# from lightgbm import LGBMClassifier
# from sklearn.base import BaseEstimator, ClassifierMixin
# import time
# import os
# import seaborn as sns
# import matplotlib.pyplot as plt
# import matplotlib
# import warnings
#
# # غیرفعال کردن هشدارهای matplotlib
# warnings.filterwarnings("ignore", category=UserWarning)
#
# # تنظیم backend به Agg برای ذخیره‌سازی تصاویر (بدون نیاز به نمایش مستقیم)
# matplotlib.use('Agg')
#
# # تنظیم برای استفاده از تمام هسته‌های CPU
# os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
#
#
# # تعریف Wrapper برای ELM با بهینه‌سازی
# class ELMWrapper(BaseEstimator, ClassifierMixin):
#     def __init__(self, n_neurons=1000, func='sigm', batch=1024, l2=0.01):
#         self.n_neurons = n_neurons
#         self.func = func
#         self.batch = batch
#         self.l2 = l2
#         self.model = None
#
#     def fit(self, X, y):
#         # تبدیل X به آرایه NumPy اگر DataFrame باشد
#         X = X.values if isinstance(X, pd.DataFrame) else X
#         y_one_hot = pd.get_dummies(y).values
#         self.model = ELM(X.shape[1], y_one_hot.shape[1], classification="c", batch=self.batch)
#         self.model.add_neurons(self.n_neurons, self.func)
#         self.model.train(X, y_one_hot, "c", l2=self.l2)
#         return self
#
#     def predict(self, X):
#         # تبدیل X به آرایه NumPy اگر DataFrame باشد
#         X = X.values if isinstance(X, pd.DataFrame) else X
#         return np.argmax(self.model.predict(X), axis=1)
#
#     def get_params(self, deep=True):
#         return {"n_neurons": self.n_neurons, "func": self.func, "batch": self.batch, "l2": self.l2}
#
#     def set_params(self, **parameters):
#         for parameter, value in parameters.items():
#             setattr(self, parameter, value)
#         return self
#
#
# # بارگذاری دیتاست
# data = pd.read_csv("dataset.csv")  # جایگزین با مسیر دیتاست شما
#
# # بررسی دیتاست
# print("Shape of dataset:", data.shape)
# print("Columns:", data.columns)
# print("Missing values:\n", data.isnull().sum())
#
# # مخلوط کردن داده‌ها برای رفع مشکل مرتب بودن
# data = data.sample(frac=1, random_state=42).reset_index(drop=True)
#
# # استخراج ویژگی‌ها و برچسب‌ها
# X = data.iloc[:, :-1].values.astype(float)  # تبدیل به float برای اطمینان
# y = data.iloc[:, -1].values
#
# # تبدیل X به DataFrame برای حفظ نام ویژگی‌ها (رفع هشدار LGBMClassifier)
# X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
#
# # بررسی عددی بودن داده‌ها
# assert X.apply(
#     lambda x: pd.to_numeric(x, errors='coerce').notnull().all()).all(), "Non-numeric values detected in features"
#
# # تقسیم داده‌ها به آموزش و تست
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#
# # استانداردسازی داده‌ها
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# # تبدیل به DataFrame برای حفظ نام ویژگی‌ها
# X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
# X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
#
#
# # تابع برای نمایش و ذخیره ماتریس درهم‌ریختگی
# def plot_confusion_matrix(y_true, y_pred, title, filename):
#     cm = confusion_matrix(y_true, y_pred)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     plt.title(title)
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.savefig(filename)  # ذخیره تصویر
#     plt.close()  # بستن شکل برای جلوگیری از مشکلات حافظه
#
#
# # تابع برای آزمایش تعداد نورون‌های مختلف
# def evaluate_stacking(n_hidden):
#     start_time = time.time()
#
#     # تعریف مدل ELM با Wrapper
#     elm = ELMWrapper(n_neurons=n_hidden, func="sigm", batch=1024, l2=0.01)
#
#     # تعریف مدل‌های پایه
#     estimators = [
#         ('elm', elm),
#         ('rf', RandomForestClassifier(
#             n_estimators=100,
#             random_state=42,
#             n_jobs=-1,
#             max_depth=10,
#             min_samples_split=5
#         )),
#         ('lgbm', LGBMClassifier(
#             num_leaves=31,
#             learning_rate=0.05,
#             n_estimators=100,
#             random_state=42,
#             device="gpu",
#             subsample=0.8,
#             colsample_bytree=0.8
#         ))
#     ]
#
#     # تعریف Stacking بدون موازی‌سازی
#     stacking = StackingClassifier(
#         estimators=estimators,
#         final_estimator=LogisticRegression(max_iter=1000, n_jobs=-1),
#         n_jobs=1  # غیرفعال کردن موازی‌سازی برای جلوگیری از خطای سریال‌سازی
#     )
#
#     # اعتبارسنجی متقاطع
#     cv_scores = cross_val_score(stacking, X_train_scaled, y_train, cv=5, scoring='accuracy', n_jobs=-1)
#
#     # آموزش مدل
#     stacking.fit(X_train_scaled, y_train)
#     y_pred = stacking.predict(X_test_scaled)
#
#     # محاسبه معیارهای ارزیابی
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred, average='weighted')
#     recall = recall_score(y_test, y_pred, average='weighted')
#     f1 = f1_score(y_test, y_pred, average='weighted')
#     cv_mean = cv_scores.mean()
#     cv_std = cv_scores.std()
#
#     # گزارش کلاسیفیکیشن
#     class_report = classification_report(y_test, y_pred)
#
#     # ذخیره ماتریس درهم‌ریختگی
#     filename = f"confusion_matrix_neurons_{n_hidden}.png"
#     plot_confusion_matrix(y_test, y_pred, f"Confusion Matrix (Neurons={n_hidden})", filename)
#     print(f"Confusion matrix saved as {filename}")
#
#     elapsed_time = time.time() - start_time
#     return accuracy, precision, recall, f1, cv_mean, cv_std, class_report, elapsed_time
#
#
# # آزمایش تعداد نورون‌های مختلف
# neuron_counts = [500, 1000, 1500]
# results = []
#
# for n in neuron_counts:
#     acc, prec, rec, f1, cv_mean, cv_std, class_report, t = evaluate_stacking(n)
#     results.append((n, acc, prec, rec, f1, cv_mean, cv_std, class_report, t))
#     print(f"\nNeurons: {n}")
#     print(f"Accuracy: {acc:.4f}")
#     print(f"Precision: {prec:.4f}")
#     print(f"Recall: {rec:.4f}")
#     print(f"F1-Score: {f1:.4f}")
#     print(f"Cross-Validation Mean: {cv_mean:.4f}")
#     print(f"Cross-Validation Std: {cv_std:.4f}")
#     print(f"Time: {t:.2f} seconds")
#     print("\nClassification Report:")
#     print(class_report)
#
# # انتخاب بهترین نتیجه بر اساس دقت
# best_result = max(results, key=lambda x: x[1])
# best_neurons, best_accuracy, best_precision, best_recall, best_f1, best_cv_mean, best_cv_std, best_class_report, best_time = best_result
# print(f"\nBest Configuration:")
# print(f"Neurons: {best_neurons}")
# print(f"Accuracy: {best_accuracy:.4f}")
# print(f"Precision: {best_precision:.4f}")
# print(f"Recall: {best_recall:.4f}")
# print(f"F1-Score: {best_f1:.4f}")
# print(f"Cross-Validation Mean: {best_cv_mean:.4f}")
# print(f"Cross-Validation Std: {best_cv_std:.4f}")
# print(f"Time: {best_time:.2f} seconds")
# print("\nBest Classification Report:")
# print(best_class_report)


# import numpy as np
# import pandas as pd
# from sklearn.model_selection import KFold
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler
# import time
#
# # تنظیمات اولیه
# np.random.seed(42)  # برای تکرارپذیری نتایج
# n_hidden = 1000  # تعداد نورون‌های مخفی (طبق مقاله)
# rbf_width = 0.08  # عرض RBF (بهینه‌شده در مقاله)
# batch_size = 10000  # اندازه دسته برای پردازش داده‌های بزرگ
#
#
# # تابع فعال‌سازی tanh
# def tanh(x):
#     return np.tanh(x)
#
#
# # تابع ELM
# def train_elm(X_train, y_train, n_hidden, rbf_width):
#     n_samples, n_features = X_train.shape
#
#     # مقداردهی تصادفی وزن‌های ورودی و بایاس
#     input_weights = np.random.randn(n_features, n_hidden) * np.sqrt(2.0 / n_features)
#     biases = np.random.randn(1, n_hidden)
#
#     # محاسبه ماتریس مخفی (H) برای دسته‌ها
#     H = np.dot(X_train, input_weights) + biases
#     H = tanh(H)  # اعمال تابع فعال‌سازی
#
#     # محاسبه وزن‌های خروجی با استفاده از معکوس‌سازی شبه‌ماتریس
#     H_pinv = np.linalg.pinv(H)
#     output_weights = np.dot(H_pinv, y_train)
#
#     return input_weights, biases, output_weights
#
#
# def predict_elm(X_test, input_weights, biases, output_weights):
#     # محاسبه ماتریس مخفی برای داده‌های تست
#     H_test = np.dot(X_test, input_weights) + biases
#     H_test = tanh(H_test)  # اعمال تابع فعال‌سازی
#
#     # پیش‌بینی خروجی
#     y_pred = np.dot(H_test, output_weights)
#     return np.argmax(y_pred, axis=1)
#
#
# # بارگذاری داده‌ها از فایل CSV
# def load_data(file_path):
#     data = pd.read_csv(file_path)
#     X = data.iloc[:, :-1].values  # همه ستون‌ها به جز آخرین ستون (ویژگی‌ها)
#     y = data.iloc[:, -1].values  # آخرین ستون به عنوان برچسب
#     return X, y
#
#
# # نرمال‌سازی داده‌ها
# def preprocess_data(X):
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     return X_scaled
#
#
# # پردازش داده‌ها به صورت دسته‌ای
# def process_in_batches(X, y, batch_size, func, *args):
#     n_samples = X.shape[0]
#     results = []
#     for start_idx in range(0, n_samples, batch_size):
#         end_idx = min(start_idx + batch_size, n_samples)
#         X_batch = X[start_idx:end_idx]
#         y_batch = y[start_idx:end_idx]
#         results.append(func(X_batch, y_batch, *args))
#     return np.vstack(results) if results[0].ndim > 1 else np.concatenate(results)
#
#
# # اجرای ELM با اعتبارسنجی 5-fold
# def run_elm(X, y, n_hidden, rbf_width):
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     accuracies = []
#     class_accuracies = {i: [] for i in range(6)}  # برای ذخیره دقت هر کلاس
#
#     # تبدیل برچسب‌ها به one-hot
#     y_one_hot = np.eye(6)[y.astype(int)]  # فرض می‌کنیم برچسب‌ها از 0 تا 5 هستند
#
#     for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
#         print(f"Fold {fold + 1}/5 started...")
#         start_time = time.time()
#
#         X_train, X_test = X[train_idx], X[test_idx]
#         y_train, y_test = y_one_hot[train_idx], y[train_idx]  # y_test برای مقایسه
#
#         # نرمال‌سازی داده‌های هر فلود
#         X_train_scaled = preprocess_data(X_train)
#         X_test_scaled = preprocess_data(X_test)
#
#         # آموزش ELM
#         input_weights, biases, output_weights = train_elm(X_train_scaled, y_train, n_hidden, rbf_width)
#
#         # پیش‌بینی
#         y_pred = predict_elm(X_test_scaled, input_weights, biases, output_weights)
#
#         # محاسبه دقت
#         accuracy = accuracy_score(y_test, y_pred)
#         accuracies.append(accuracy)
#
#         # محاسبه دقت برای هر کلاس
#         for cls in range(6):
#             cls_mask = y_test == cls
#             if np.sum(cls_mask) > 0:
#                 cls_accuracy = accuracy_score(y_test[cls_mask], y_pred[cls_mask])
#                 class_accuracies[cls].append(cls_accuracy)
#
#         print(f"Fold {fold + 1} completed in {time.time() - start_time:.2f} seconds, Accuracy: {accuracy:.4f}")
#
#     return accuracies, class_accuracies
#
#
# # مسیر فایل داده‌ها
# file_path = 'Dataset.csv'  # جایگزین با مسیر فایل واقعی شما
# X, y = load_data(file_path)
#
# # اجرای مدل
# accuracies, class_accuracies = run_elm(X, y, n_hidden, rbf_width)
#
# # گزارش نتایج
# print(f"\nMean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
# print("\nClass-wise Accuracy:")
# emotion_names = ['Happiness', 'Surprise', 'Anger', 'Sadness', 'Disgust', 'Fear']
# for cls in range(6):
#     mean_cls_acc = np.mean(class_accuracies[cls]) if class_accuracies[cls] else 0.0
#     print(f"{emotion_names[cls]}: {mean_cls_acc:.4f}")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import time

# تنظیمات اولیه
np.random.seed(42)  # برای تکرارپذیری نتایج
n_hidden = 2000  # تعداد نورون‌های مخفی (کاهش برای سرعت)
rbf_width = 0.08  # عرض RBF (بهینه‌شده در مقاله)
sample_size = 20000  # اندازه نمونه تصادفی برای تخمین سریع
test_size = 0.2  # 20% برای تست

# تابع فعال‌سازی tanh
def tanh(x):
    return np.tanh(x)

# تابع ELM
def train_elm(X_train, y_train, n_hidden, rbf_width):
    n_samples, n_features = X_train.shape

    # مقداردهی تصادفی وزن‌های ورودی و بایاس
    input_weights = np.random.randn(n_features, n_hidden) * np.sqrt(2.0 / n_features)
    biases = np.random.randn(1, n_hidden)

    # محاسبه ماتریس مخفی (H)
    H = np.dot(X_train, input_weights) + biases
    H = tanh(H)  # اعمال تابع فعال‌سازی

    # محاسبه وزن‌های خروجی با استفاده از معکوس‌سازی شبه‌ماتریس
    H_pinv = np.linalg.pinv(H)
    output_weights = np.dot(H_pinv, y_train)

    return input_weights, biases, output_weights

def predict_elm(X_test, input_weights, biases, output_weights):
    # محاسبه ماتریس مخفی برای داده‌های تست
    H_test = np.dot(X_test, input_weights) + biases
    H_test = tanh(H_test)  # اعمال تابع فعال‌سازی

    # پیش‌بینی خروجی
    y_pred = np.dot(H_test, output_weights)
    return np.argmax(y_pred, axis=1)

# بارگذاری داده‌ها از فایل CSV
def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values  # همه ستون‌ها به جز آخرین ستون (ویژگی‌ها)
    y = data.iloc[:, -1].values   # آخرین ستون به عنوان برچسب
    return X, y

# نرمال‌سازی داده‌ها
def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# اجرای ELM با Hold-out Validation
def run_elm_fast(X, y, n_hidden, rbf_width, sample_size, test_size):
    # نمونه‌گیری تصادفی از داده‌ها
    indices = np.random.choice(X.shape[0], sample_size, replace=False)
    X_sampled = X[indices]
    y_sampled = y[indices]

    # تقسیم داده‌ها به آموزش و تست
    X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=test_size, random_state=42)

    # تبدیل برچسب‌های آموزشی به one-hot
    y_train_one_hot = np.eye(6)[y_train.astype(int)]

    # نرمال‌سازی
    X_train_scaled = preprocess_data(X_train)
    X_test_scaled = preprocess_data(X_test)

    print("Training started...")
    start_time = time.time()

    # آموزش ELM
    input_weights, biases, output_weights = train_elm(X_train_scaled, y_train_one_hot, n_hidden, rbf_width)

    # پیش‌بینی
    y_pred = predict_elm(X_test_scaled, input_weights, biases, output_weights)

    # محاسبه دقت
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Training completed in {time.time() - start_time:.2f} seconds, Accuracy: {accuracy:.4f}")

    # محاسبه دقت برای هر کلاس
    class_accuracies = {}
    for cls in range(6):
        cls_mask = y_test == cls
        if np.sum(cls_mask) > 0:
            cls_accuracy = accuracy_score(y_test[cls_mask], y_pred[cls_mask])
            class_accuracies[cls] = cls_accuracy
        else:
            class_accuracies[cls] = 0.0

    return accuracy, class_accuracies

# مسیر فایل داده‌ها
file_path = 'Dataset.csv'  # جایگزین با مسیر فایل واقعی شما
X, y = load_data(file_path)

# اجرای مدل با سرعت بیشتر
accuracy, class_accuracies = run_elm_fast(X, y, n_hidden, rbf_width, sample_size, test_size)

# گزارش نتایج
print(f"\nEstimated Mean Accuracy (based on sampled data): {accuracy:.4f}")
print("\nClass-wise Accuracy (estimated):")
emotion_names = ['Happiness', 'Surprise', 'Anger', 'Sadness', 'Disgust', 'Fear']
for cls in range(6):
    print(f"{emotion_names[cls]}: {class_accuracies[cls]:.4f}")
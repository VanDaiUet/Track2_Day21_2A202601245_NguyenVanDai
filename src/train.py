import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

EVAL_THRESHOLD = 0.70


def transform_features(X):
    """
    Hàm biến đổi 12 đặc trưng ban đầu thành 19 đặc trưng bằng Feature Engineering.
    Hoạt động với cả numpy array và pandas DataFrame.
    """
    if hasattr(X, "values"):
        X = X.values
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    total_acidity = (X[:, 0] + X[:, 1] + X[:, 2]).reshape(-1, 1)
    bound_so2 = (X[:, 6] - X[:, 5]).reshape(-1, 1)
    so2_ratio = (X[:, 5] / (X[:, 6] + 1e-5)).reshape(-1, 1)
    sugar_alcohol = (X[:, 3] / (X[:, 10] + 1e-5)).reshape(-1, 1)
    alcohol_density = (X[:, 10] / (X[:, 7] + 1e-5)).reshape(-1, 1)
    acid_ph = (X[:, 0] / (X[:, 8] + 1e-5)).reshape(-1, 1)
    sulphate_alcohol = (X[:, 9] * X[:, 10]).reshape(-1, 1)
    
    return np.hstack([X, total_acidity, bound_so2, so2_ratio, sugar_alcohol, alcohol_density, acid_ph, sulphate_alcohol])


def get_model(params: dict):
    """
    Khởi tạo mô hình dựa trên tham số model_type (Bonus 2) và use_feature_engineering.
    Mặc định sử dụng random_forest nếu không chỉ định model_type.
    """
    use_fe = params.get("use_feature_engineering", False)
    model_type = params.get("model_type", "random_forest")
    model_params = {k: v for k, v in params.items() if k not in ["model_type", "use_feature_engineering"]}

    model_params.setdefault("random_state", 42)

    if model_type == "random_forest":
        base_model = RandomForestClassifier(**model_params)
    elif model_type == "extra_trees":
        base_model = ExtraTreesClassifier(**model_params)
    elif model_type == "hist_gradient_boosting":
        base_model = HistGradientBoostingClassifier(**model_params)
    elif model_type == "gradient_boosting":
        base_model = GradientBoostingClassifier(**model_params)
    elif model_type == "logistic_regression":
        model_params.setdefault("max_iter", 1000)
        base_model = LogisticRegression(**model_params)
    else:
        raise ValueError(f"Khong ho tro model_type: {model_type}")

    if use_fe:
        from sklearn.preprocessing import QuantileTransformer
        return Pipeline([
            ("fe", FunctionTransformer(transform_features)),
            ("scaler", QuantileTransformer(output_distribution="normal", random_state=42)),
            ("clf", base_model)
        ])
    return base_model




def check_data_drift_and_distribution(y_train: pd.Series) -> dict:
    """
    Bonus 5: Kiểm tra tỷ lệ phân phối nhãn trong tập huấn luyện.
    Cảnh báo nếu có lớp chiếm dưới 10% tổng số mẫu.
    """
    total = len(y_train)
    counts = y_train.value_counts().to_dict()
    distribution = {}
    for cls in [0, 1, 2]:
        cls_count = counts.get(cls, 0)
        ratio = cls_count / total if total > 0 else 0
        distribution[str(cls)] = round(ratio, 4)
        if ratio < 0.10:
            print(f"[CANH BAO LECH DULIEU] Lop {cls} chiem {ratio*100:.2f}% (< 10% tong mau)")
    return distribution


def generate_detailed_report(y_eval, preds, output_path: str = "outputs/report.txt"):
    """
    Bonus 3: Tạo báo cáo hiệu suất chi tiết (Confusion Matrix & Classification Report).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cm = confusion_matrix(y_eval, preds, labels=[0, 1, 2])
    report_str = classification_report(
        y_eval,
        preds,
        labels=[0, 1, 2],
        target_names=["thap (0)", "trung_binh (1)", "cao (2)"],
        zero_division=0,
    )

    report_content = "=== BAO CAO HIEU SUAT MO HINH ===\n\n"
    report_content += "1. Confusion Matrix (Ma tran nham lan):\n"
    report_content += f"{cm}\n\n"
    report_content += "2. Classification Report (Precision, Recall, F1-Score):\n"
    report_content += f"{report_str}\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print("\n" + report_content)


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huấn luyện mô hình và ghi nhận kết quả vào MLflow.

    Tham số:
        params     : dict chứa các siêu tham số cho mô hình.
        data_path  : đường dẫn đến file dữ liệu huấn luyện.
        eval_path  : đường dẫn đến file dữ liệu đánh giá.

    Trả về:
        accuracy (float): độ chính xác trên tập đánh giá.
    """
    # 1. Đọc dữ liệu huấn luyện và đánh giá
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # 2. Tách đặc trưng (X) và nhãn (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # Bonus 5: Kiểm tra phân phối dữ liệu
    class_distribution = check_data_drift_and_distribution(y_train)

    # 4. Khởi tạo và huấn luyện mô hình
    model = get_model(params)
    model.fit(X_train, y_train)

    # 5. Dự đoán trên tập đánh giá và tính chỉ số
    preds = model.predict(X_eval)
    acc = float(accuracy_score(y_eval, preds))
    f1 = float(f1_score(y_eval, preds, average="weighted"))

    # Log vào MLflow một cách an toàn
    try:
        if "MLFLOW_TRACKING_URI" not in os.environ:
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("Wine_Quality_Classification")
        
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            for cls, ratio in class_distribution.items():
                mlflow.log_metric(f"class_ratio_{cls}", ratio)
            mlflow.sklearn.log_model(model, "model")
    except Exception as e:
        print(f"[MLflow Warning] Khong the ghi log vao MLflow: {e}")


    # 7. In kết quả ra màn hình
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

    # Bonus 3: Tạo báo cáo chi tiết outputs/report.txt
    generate_detailed_report(y_eval, preds, output_path="outputs/report.txt")

    # 8. Lưu metrics ra file outputs/metrics.json
    os.makedirs("outputs", exist_ok=True)
    metrics_data = {
        "accuracy": acc,
        "f1_score": f1,
        "class_distribution": class_distribution,
    }
    with open("outputs/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    # 9. Lưu mô hình ra file models/model.pkl
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    # 10. Trả về accuracy
    return acc



if __name__ == "__main__":
    with open("params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    train(params)


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

EVAL_THRESHOLD = 0.70


def get_model(params: dict):
    """
    Khởi tạo mô hình dựa trên tham số model_type (Bonus 2).
    Mặc định sử dụng random_forest nếu không chỉ định model_type.
    """
    model_type = params.get("model_type", "random_forest")
    model_params = {k: v for k, v in params.items() if k != "model_type"}

    if model_type == "random_forest":
        return RandomForestClassifier(**model_params, random_state=42)
    elif model_type == "extra_trees":
        return ExtraTreesClassifier(**model_params, random_state=42)
    elif model_type == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(**model_params, random_state=42)
    elif model_type == "gradient_boosting":
        return GradientBoostingClassifier(**model_params, random_state=42)
    elif model_type == "logistic_regression":
        return LogisticRegression(**model_params, random_state=42, max_iter=1000)
    else:
        raise ValueError(f"Khong ho tro model_type: {model_type}")



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

    with mlflow.start_run():
        # 3. Ghi nhận các siêu tham số vào MLflow
        mlflow.log_params(params)

        # 4. Khởi tạo và huấn luyện mô hình
        model = get_model(params)
        model.fit(X_train, y_train)

        # 5. Dự đoán trên tập đánh giá và tính chỉ số
        preds = model.predict(X_eval)
        acc = float(accuracy_score(y_eval, preds))
        f1 = float(f1_score(y_eval, preds, average="weighted"))

        # 6. Ghi nhận chỉ số vào MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        for cls, ratio in class_distribution.items():
            mlflow.log_metric(f"class_ratio_{cls}", ratio)
        
        mlflow.sklearn.log_model(model, "model")

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


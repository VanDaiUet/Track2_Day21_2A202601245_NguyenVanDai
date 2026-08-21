from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os
import sys
import numpy as np


def transform_features(X):
    """
    Hàm biến đổi 12 đặc trưng ban đầu thành 19 đặc trưng bằng Feature Engineering.
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


# Gán vào module hiện tại và __main__ để pickle/unpickle tìm thấy hàm
sys.modules[__name__].transform_features = transform_features
if "__main__" in sys.modules:
    sys.modules["__main__"].transform_features = transform_features


app = FastAPI(title="Wine Quality Inference API")

GCS_BUCKET = os.environ.get("GCS_BUCKET", "")
GCS_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")


def download_model():
    """
    Tải file model.pkl từ GCS về máy khi server khởi động.
    """
    if not GCS_BUCKET:
        print("CANH BAO: GCS_BUCKET chua duoc dat trong bien moi truong.")
        return

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(GCS_MODEL_KEY)
        blob.download_to_filename(MODEL_PATH)
        print(f"Model da duoc tai thanh cong tu gs://{GCS_BUCKET}/{GCS_MODEL_KEY} ve {MODEL_PATH}")
    except Exception as e:
        print(f"Loi khi tai model tu GCS: {e}")


# Tải mô hình khi khởi động server
download_model()
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    print(f"File model chua ton tai tai {MODEL_PATH}")


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    """
    Endpoint kiểm tra sức khỏe server.
    GitHub Actions gọi endpoint này sau khi deploy để xác nhận server đang chạy.
    """
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luận chính.

    Đầu vào : JSON {"features": [f1, f2, ..., f12]}
    Đầu ra  : JSON {"prediction": <0|1|2>, "label": <"thap"|"trung_binh"|"cao">}
    """
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        else:
            raise HTTPException(status_code=503, detail="Model is not loaded yet")

    # 1. Kiểm tra số lượng đặc trưng
    if len(req.features) != 12:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 12 features (wine quality), but got {len(req.features)}",
        )

    # 2. Gọi model.predict
    pred = int(model.predict([req.features])[0])

    # 3. Ánh xạ nhãn: 0 -> "thap", 1 -> "trung_binh", 2 -> "cao"
    labels_map = {0: "thap", 1: "trung_binh", 2: "cao"}
    label = labels_map.get(pred, "khong_xac_dinh")

    return {"prediction": pred, "label": label}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


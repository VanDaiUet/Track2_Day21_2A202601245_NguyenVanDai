from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os

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


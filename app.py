import os
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# 1. MLflow 설정 및 모델 로드
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)

app = FastAPI(title="Iris Classifier API", version="1.0")

# 서버 시작 시 모델을 미리 로드합니다.
try:
    model = mlflow.sklearn.load_model("models:/iris_classifier@production")
    print("✅ Production 모델 로드 완료!")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")

# 2. 데이터 구조 정의
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

iris_classes = ["setosa", "versicolor", "virginica"]

# 3. 엔드포인트 설정
@app.get("/")
def health_check():
    return {"status": "ok", "model": "iris_classifier@production"}

@app.post("/predict")
def predict(data: IrisInput):
    # 입력 데이터를 numpy 배열로 변환
    features = np.array([[
        data.sepal_length, 
        data.sepal_width,
        data.petal_length, 
        data.petal_width
    ]])

    # 모델 예측
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0].tolist()

    # 결과 반환 (이 return문이 반드시 있어야 null이 안 나옵니다!)
    return {
        "prediction": int(prediction),
        "class_name": iris_classes[prediction],
        "probability": probability
    }
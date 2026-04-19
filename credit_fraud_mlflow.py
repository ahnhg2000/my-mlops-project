"""
신용카드 사기 탐지 - Scikit-learn 파이프라인 + MLflow 통합
======================================================
RandomForest 파라미터 5가지 조합 비교 및 MLflow Tracking/Registry 적용
"""

import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")
import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# MLflow 연결 설정 ────────────────────────────────────────────────
# 환경변수 MLFLOW_TRACKING_URI가 있으면 사용, 없으면 로컬 서버 주소 사용
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)

# Basic Auth 설정 (DagsHub 등 사용 시)
if "MLFLOW_TRACKING_USERNAME" in os.environ:
    os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("MLFLOW_TRACKING_USERNAME")
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("MLFLOW_TRACKING_PASSWORD")

# 실험(Experiment) 이름 설정
experiment_name = "credit_fraud_detection"
mlflow.set_experiment(experiment_name)

model_dir = "model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"'{model_dir}' 폴더가 생성되었습니다.")
else:
    print(f"'{model_dir}' 폴더가 이미 존재합니다.")

# ── 1. 데이터 로드 ──────────────────────────────────────────────────────────
try:
    df = pd.read_csv("data/credit_fraud_dataset.csv")
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    print(f"✅ 데이터 로드 완료 | 크기: {X.shape} | 사기 비율: {y.mean():.1%}")
except FileNotFoundError:
    print("❌ 데이터를 찾을 수 없습니다. 경로를 확인하세요.")
    exit(1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ── 2. 컬럼 분류 ────────────────────────────────────────────────────────────
numeric_features     = ["amount", "hour", "transaction_count_1h", "distance_from_home_km", "age"]
categorical_features = ["merchant_category", "card_type", "country"]

# ── 3. 전처리 서브-파이프라인 ────────────────────────────────────────────────
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer,     numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# ── 4. RandomForest 파라미터 조합 5가지 ────────────────────────────────────
param_list = [
    {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "class_weight": "balanced"},
    {"n_estimators": 200, "max_depth": 10,   "min_samples_split": 5, "class_weight": "balanced"},
    {"n_estimators": 50,  "max_depth": 5,    "min_samples_split": 10, "class_weight": None},
    {"n_estimators": 300, "max_depth": 20,   "min_samples_split": 2, "class_weight": "balanced_subsample"},
    {"n_estimators": 150, "max_depth": 8,    "min_samples_split": 4, "class_weight": "balanced"},
]

# ── 5. 파이프라인 생성 및 학습/평가 (MLflow Tracking 적용) ──────────────────
print("\n" + "="*85)
print(f"  {'run_name':<35} {'test_acc':>9} {'AUC':>9}  {'status'}")
print("="*85)

run_results = []

for params in param_list:
    # MLflow Run 이름 생성
    run_name = f"n{params['n_estimators']}_d{params['max_depth']}_min{params['min_samples_split']}"
    
    with mlflow.start_run(run_name=run_name):
        pipe = Pipeline([
            ("preprocessor", preprocessor), 
            ("clf", RandomForestClassifier(**params, random_state=42))
        ])
        pipe.fit(X_train, y_train)

        # 예측 및 메트릭 계산
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        # MLflow 로깅
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)
        
        # 모델을 MLflow Artifact로 저장
        model_info = mlflow.sklearn.log_model(pipe, artifact_path="model")
        
        # 기존 로컬 저장 로직 유지
        local_model_path = f"{model_dir}/pipeline_{run_name}.pkl"
        joblib.dump(pipe, local_model_path)
        
        run_results.append({
            "run_name": run_name,
            "accuracy": acc,
            "roc_auc": auc,
            "model_uri": model_info.model_uri,
            "local_path": local_model_path
        })

        print(f"  {run_name:<35} {acc:>9.4f} {auc:>9.4f}  Logged to MLflow")


# ── 6. 가장 좋은 모델 자동 선택 ─────────────────────────────────────────────
# Accuracy 기준으로 최적 모델 선택 (필요시 roc_auc로 변경 가능)
best = max(run_results, key=lambda x: x["accuracy"])
print(f"\n🏆 최고 모델: {best['run_name']} | Accuracy: {best['accuracy']:.4f}")

# ── 7. 최고 모델 Model Registry 등록 및 Alias 설정 ──────────────────────────
# MLflow Registry 등록
model_name = "credit_fraud_classifier"
registered = mlflow.register_model(model_uri=best["model_uri"], name=model_name)

# Production 별칭 부여
client = MlflowClient()
client.set_registered_model_alias(
    name=model_name,
    alias="production",
    version=registered.version
)
print(f"✅ Registry 등록 완료: {model_name} (Version: {registered.version})")
print(f"🚀 Production Alias 설정 완료")

# ── 8. 최고 모델 불러오기 및 상세 평가 (Registry 모델 사용 가능) ───────────
# 여기서는 방금 로드한 pipeline 객체나 로컬 저장본을 사용
best_pipeline_model = joblib.load(best["local_path"])

y_pred_best  = best_pipeline_model.predict(X_test)
y_proba_best = best_pipeline_model.predict_proba(X_test)[:, 1]

print("\n" + "="*65)
print(f"최종 모델 분류 리포트 — {best['run_name']}")
print("="*65)
print(classification_report(y_test, y_pred_best, target_names=["정상(0)", "사기(1)"]))
print(f"최종 ROC-AUC: {roc_auc_score(y_test, y_proba_best):.4f}")

# ── 9. 예제 데이터 예측 ──────────────────────────────────────────────────────
example_data = pd.DataFrame({
    "amount":                [500.0,    12.5],
    "hour":                  [3,        14  ],
    "transaction_count_1h":  [8,        1   ],
    "distance_from_home_km": [300.0,    2.0 ],
    "age":                   [25.0,     45.0],
    "merchant_category":     ["online", "grocery" ],
    "card_type":             ["credit", "debit"   ],
    "country":               ["foreign","domestic"],
})

print("\n--- 예제 데이터 예측 ---")
predictions = best_pipeline_model.predict(example_data)
print(predictions)
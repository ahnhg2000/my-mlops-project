import pandas as pd                                          # 데이터프레임 처리용
from sklearn.model_selection import train_test_split         # 데이터를 train/test로 분리
from sklearn.preprocessing import StandardScaler             # 특성값 정규화 (평균0, 표준편차1)
from sklearn.ensemble import RandomForestClassifier          # 앙상블 기반 분류 모델
from sklearn.pipeline import Pipeline                        # 전처리+모델을 하나로 묶는 파이프라인
from sklearn.metrics import accuracy_score                   # 예측 정확도 계산
import mlflow                                                # 실험 추적 및 모델 관리
import mlflow.sklearn                                        # sklearn 모델 전용 MLflow 기능
import os                                                    # 환경변수 읽기용
from mlflow.tracking import MlflowClient                     # MLflow 서버 직접 조작용 클라이언트
 
# ── MLflow 연결 설정 ──────────────────────────────────────────
# 환경변수 MLFLOW_TRACKING_URI가 있으면 사용, 없으면 로컬 서버 주소로 대체
# GitHub Actions에서는 Secrets에 등록된 값이 자동으로 환경변수로 주입됨
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)                        # MLflow가 이 주소의 서버로 연결
 
# Basic Auth가 필요한 MLflow 서버(예: DagsHub)일 경우 인증 정보 설정
# 환경변수에 USERNAME이 있을 때만 실행 → 로컬에서는 건너뜀
if "MLFLOW_TRACKING_USERNAME" in os.environ:
    os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("MLFLOW_TRACKING_USERNAME")
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("MLFLOW_TRACKING_PASSWORD")
 
experiment_name = "iris_classification"                      # 실험 묶음 이름 (MLflow UI에서 폴더처럼 표시됨)
mlflow.set_experiment(experiment_name)                       # 없으면 자동 생성, 있으면 해당 실험에 기록
 
# ── 1. 데이터 로드 및 전처리 ──────────────────────────────────
try:
    df = pd.read_csv("data/iris_data.csv")                   # DVC로 관리되는 데이터 파일 로드
    df = df.select_dtypes(include=['number']).dropna()        # 수치형 컬럼만 선택 + 결측치 행 제거
    X = df.drop('target', axis=1)                            # 입력 특성 (꽃받침/꽃잎 길이·너비)
    y = df['target']                                         # 정답 레이블 (0=setosa, 1=versicolor, 2=virginica)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("✅ 데이터 로드 및 전처리 완료")
except FileNotFoundError:
    print("❌ 데이터를 찾을 수 없음. dvc pull 확인 필요.")
    exit(1)                                                   # 데이터 없으면 실험 자체가 의미 없으므로 종료
 
# ── 2. 실험 파라미터 목록 정의 ────────────────────────────────
run_results = []                                             # 각 run의 결과를 담을 리스트 (나중에 최고 모델 선택용)
param_list = [
    {"n_estimators": 50,  "max_depth": 2},                  # 트리 50개, 최대 깊이 2 (단순한 모델)
    {"n_estimators": 100, "max_depth": 3},                  # 트리 100개, 최대 깊이 3
    {"n_estimators": 200, "max_depth": 5},                  # 트리 200개, 최대 깊이 5
    {"n_estimators": 300, "max_depth": 4},                  # 트리 300개, 최대 깊이 4 (복잡한 모델)
]
 
# ── 3. 파라미터별 실험 실행 ───────────────────────────────────
for params in param_list:
    run_name = f"n{params['n_estimators']}_d{params['max_depth']}"  # ex) "n100_d3" → MLflow UI에서 식별용
    with mlflow.start_run(run_name=run_name):                # 이 블록 안의 모든 기록이 하나의 run으로 묶임
 
        # 전처리(StandardScaler) + 모델을 하나의 파이프라인으로 구성
        # Pipeline 사용 시 predict()만 해도 자동으로 스케일링 후 예측
        pipe = Pipeline([
            ("scaler", StandardScaler()),                    # 특성값을 평균0, 표준편차1로 정규화
            ("clf", RandomForestClassifier(**params, random_state=42))  # 랜덤포레스트 (random_state로 재현성 보장)
        ])
        pipe.fit(X_train, y_train)                           # 학습: 스케일러 fit + 모델 학습 한 번에 처리
 
        acc = accuracy_score(y_test, pipe.predict(X_test))   # 테스트셋으로 정확도 평가
 
        mlflow.log_params(params)                            # n_estimators, max_depth를 MLflow에 기록
        mlflow.log_metric("accuracy", acc)                   # 정확도를 MLflow에 기록 (나중에 비교 가능)
 
        # 모델을 MLflow artifact로 저장 → name="model"이 predict.py의 URI 조합과 일치해야 함
        model_info = mlflow.sklearn.log_model(pipe, name="model")
 
        # 이 run의 결과를 리스트에 저장 (for문 끝난 후 최고 모델 선택에 사용)
        run_results.append({
            "run_name": run_name,
            "accuracy": acc,
            "model_uri": model_info.model_uri                # ex) runs:/abc123/model
        })
        print(f"  {run_name}: {acc:.4f} | uri: {model_info.model_uri}")
 
# ── 4. 최고 모델 선택 ─────────────────────────────────────────
# accuracy 기준으로 가장 높은 결과를 가진 딕셔너리 반환
best = max(run_results, key=lambda x: x["accuracy"])
print(f"🏆 최고 모델: {best['run_name']} | accuracy: {best['accuracy']:.4f}")
 
# ── 5. Model Registry에 등록 ──────────────────────────────────
# run URI → Registry로 복사 (버전 자동 부여됨)
registered = mlflow.register_model(
    model_uri=best["model_uri"],                             # 최고 모델의 run URI
    name="iris_classifier"                                   # Registry에서 사용할 모델 이름
)
print(f"✅ 등록 완료! Version: {registered.version}")
 
# ── 6. Production alias 설정 ──────────────────────────────────
client = MlflowClient()                                      # Registry 조작을 위한 클라이언트
client.set_registered_model_alias(
    name="iris_classifier",                                  # 대상 모델 이름
    alias="production",                                      # 붙일 별명 (predict.py에서 @production으로 참조)
    version=registered.version                               # 방금 등록된 버전에 alias 연결
)
print(f"🚀 production alias → Version {registered.version}")
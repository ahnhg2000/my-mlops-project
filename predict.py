import mlflow                                        # MLflow 실험 추적 및 모델 관리 라이브러리
import mlflow.sklearn                                  # sklearn 모델 전용 MLflow 기능 (저장/로드)
from sklearn.datasets import load_iris                 # 테스트용 iris 데이터셋
from sklearn.model_selection import train_test_split   # 데이터를 train/test로 분리
from sklearn.metrics import accuracy_score             # 예측 정확도 계산
import os                                              # 환경변수 읽기용
from mlflow.tracking import MlflowClient               # MLflow 서버와 직접 통신하는 클라이언트
 
# ── MLflow 연결 설정 ──────────────────────────────────────────
# 환경변수 MLFLOW_TRACKING_URI가 있으면 그 값 사용, 없으면 로컬 서버 주소로 대체
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)                  # MLflow가 이 주소로 서버에 연결하도록 설정
 
client = MlflowClient()                                # API로 실험/모델 정보를 직접 조회할 클라이언트 생성
 
# ── 1. Production 모델 로드 & 정확도 확인 ─────────────────────
# "models:/모델명@alias" 형식 → 버전 번호 몰라도 alias로 최신 production 모델 참조 가능
model_uri = "models:/iris_classifier@production"
loaded_model = mlflow.sklearn.load_model(model_uri)    # Registry에서 production alias 모델 로드
print("✅ Production 모델 로드 완료!")
 
iris = load_iris()                                     # iris 데이터셋 로드 (특성 4개, 클래스 3개)
X, y = iris.data, iris.target                          # X: 입력 특성, y: 정답 레이블
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2)
 
predictions = loaded_model.predict(X_test)             # 테스트 데이터로 예측 수행
acc = accuracy_score(y_test, predictions)              # 정답과 예측값 비교해서 정확도 계산
print(f"🎯 Production 모델 정확도: {acc:.4f}")
 
# ── 2. 실험 내 모든 Run 가져오기 ──────────────────────────────
# 실험 이름으로 실험 객체 조회 (experiment_id를 하드코딩하지 않기 위함)
experiment = client.get_experiment_by_name("iris_classification")
if experiment is None:                                 # 실험이 존재하지 않으면 즉시 종료
    print("❌ 실험 'iris_classification'을 찾을 수 없습니다.")
    exit(1)
 
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],         # 위에서 찾은 실험 ID로 필터링
    order_by=["metrics.accuracy DESC"]                 # 정확도 높은 순으로 정렬 → [0]=1위, [1]=2위
)
 
# ── 3. run_results 구성 (model_uri 포함) ──────────────────────
run_results = []                                       # 유효한 run 정보를 담을 빈 리스트
for run in runs:
    accuracy = run.data.metrics.get("accuracy", None)  # 해당 run의 accuracy 메트릭 조회
    if accuracy is None:                               # accuracy가 기록되지 않은 run은 스킵
        continue
 
    # train.py에서 mlflow.sklearn.log_model(pipe, name="model") 로 저장했으므로
    # artifact 경로가 항상 "model"임 → URI 직접 조합
    model_uri_run = f"runs:/{run.info.run_id}/model"
    run_results.append({
        "run_id": run.info.run_id,                     # run 고유 ID (Registry 버전 매칭에 사용)
        "run_name": run.info.run_name,                 # run 이름 (ex. n100_d3)
        "accuracy": accuracy,                          # 정확도 수치
        "model_uri": model_uri_run                     # 모델 로드/등록에 사용할 URI
    })
 
print(f"\n📊 총 {len(run_results)}개 Run 발견")
for r in run_results:                                  # 정렬된 순서대로 전체 결과 출력
    print(f"  {r['run_name']} | accuracy: {r['accuracy']:.4f} | run_id: {r['run_id']}")
 
# ── 4. 2위 모델 Registry 등록 ─────────────────────────────────
if len(run_results) < 2:                               # run이 1개 이하면 2위를 뽑을 수 없음
    print("⚠️ 비교할 Run이 2개 미만입니다. 2위 모델 등록을 건너뜁니다.")
    exit(0)
 
# order_by DESC로 이미 정렬됐으므로 index 1 = 정확도 2위 모델
second_best = run_results[1]
print(f"\n🥈 2위 모델: {second_best['run_name']} | accuracy: {second_best['accuracy']:.4f}")
 
registered = mlflow.register_model(
    model_uri=second_best["model_uri"],                # 2위 모델의 run URI → Registry에 등록
    name="iris_classifier"                             # Registry에서 사용할 모델 이름
)
second_version = registered.version                    # 등록 후 자동 부여된 버전 번호 저장
print(f"✅ 2위 모델 등록 완료! Version: {second_version}")
 
# ── 5. 2위 모델에 production alias 설정 ───────────────────────
# alias를 바꾸는 것만으로 배포 전환 가능 → 코드 변경 없이 모델 교체
client.set_registered_model_alias("iris_classifier", "production", second_version)
print(f"🚀 Version {second_version} → production!")
 
# alias가 실제로 바뀌었는지 확인
current = client.get_model_version_by_alias("iris_classifier", "production")
print(f"현재 production: Version {current.version}")
 
# ── 6. 롤백 시나리오 ─────────────────────────────────────────
print(f"\n⚠️  Version {second_version}에서 문제 발생! 롤백합니다...")
 
# order_by DESC 정렬이므로 index 0 = 정확도 1위 (롤백 대상)
best = run_results[0]
# Registry에 등록된 모든 버전 목록 조회
all_versions = client.search_model_versions("name='iris_classifier'")
 
best_version = None
for v in all_versions:
    if v.run_id == best["run_id"]:                     # run_id로 1위 모델의 Registry 버전 찾기
        best_version = v.version
        break                                          # 찾으면 즉시 루프 종료
 
if best_version is None:                               # 1위 모델이 Registry에 없으면 롤백 불가
    print("⚠️ 1위 모델이 Registry에 없습니다. 먼저 등록이 필요합니다.")
    exit(1)
 
# 모델 파일은 그대로, alias만 변경 → 즉시 롤백 (다운타임 없음)
client.set_registered_model_alias("iris_classifier", "production", best_version)
print(f"✅ Version {best_version} 으로 롤백 완료!")
 
# 롤백 후 현재 production 버전 최종 확인
current = client.get_model_version_by_alias("iris_classifier", "production")
print(f"현재 production: Version {current.version}")
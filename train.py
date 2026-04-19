import pandas as pd
#from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline  # 전처리와 모델 파이프라인 연결
from sklearn.metrics import accuracy_score
import joblib
import os
model_dir = "model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"'{model_dir}' 폴더가 생성되었습니다.")
else:
    print(f"'{model_dir}' 폴더가 이미 존재합니다.")

try:
    df = pd.read_csv("data/iris_data.csv")
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
    print("✅ 데이터 로드 및 전처리 완료")
except FileNotFoundError:
    print("❌ 데이터를 찾을 수 없음. ")
  
run_results=[]  
param_list = [
    {"n_estimators": 50,  "max_depth": 2},
    {"n_estimators": 100, "max_depth": 3},
    {"n_estimators": 200, "max_depth": 5},
    {"n_estimators": 300, "max_depth": 4},
]

for params in param_list:
    run_name = f"n{params['n_estimators']}_d{params['max_depth']}"
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(**params, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    train_acc = accuracy_score(y_train, pipe.predict(X_train))
    joblib.dump(pipe, f"model/pipeline_n{params['n_estimators']}_d{params['max_depth']}.pkl")   
    run_results.append({
        "run_name": run_name,
        "accuracy": acc,
        "train_acc": train_acc,
        "model_uri": f"model/pipeline_n{params['n_estimators']}_d{params['max_depth']}.pkl"
    }) 
    
    #결과 출력 
    print(f"  {run_name}: {acc:.4f}/ {train_acc:.4f} |  pipeline_n{params['n_estimators']}_d{params['max_depth']}.pkl")
# 실험 후 — 가장 좋은 모델 자동 선택
best = max(run_results, key=lambda x: x["accuracy"])
print(f"🏆 최고 모델: {best['run_name']} | accuracy: {best['accuracy']:.4f}")

# 이전 실험에서 가장 좋은 모델 불러오기
# 여기서는 파이프라인 모델 'pipeline_{'n_estimators': 50, 'max_depth': 2}.pkl'을 사용합니다.
# (파이프라인이 스케일링을 자동으로 처리하므로 더 간편합니다)
model_path = best[ "model_uri"]
best_pipeline_model = joblib.load(model_path)

# 예제 데이터 준비 (X_test의 일부 사용)
example_data = X_test.head(5)

print("--- 예제 데이터 ---")
print(example_data)

# 예측 수행
predictions = best_pipeline_model.predict(example_data)

print("\n--- 예측 결과 ---")
print(predictions)

# 실제 값 (비교를 위해 y_test에서 해당 부분 가져오기)
actual_labels = y_test.head(5)
print("\n--- 실제 레이블 ---")
print(actual_labels)

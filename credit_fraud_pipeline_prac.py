"""
신용카드 사기 탐지 - Scikit-learn 파이프라인
==============================================
RandomForest 파라미터 5가지 조합 비교 (train/test split만 사용)
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

model_dir = "model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"'{model_dir}' 폴더가 생성되었습니다.")
else:
    print(f"'{model_dir}' 폴더가 이미 존재합니다.")

# ── 1. 데이터 로드 ──────────────────────────────────────────────────────────
df = pd.read_csv("data/credit_fraud_dataset.csv")
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

print(f"데이터 크기: {X.shape}  |  사기 비율: {y.mean():.1%}")

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

# ── 5. 파이프라인 생성 및 학습/평가 ─────────────────────────────────────────
print("\n" + "="*65)
print(f"  {'run_name':<30} {'test_acc':>9} {'train_acc':>10}  {'model_uri'}")
print("="*65)

run_results = []

for params in param_list:
    run_name = f"n{params['n_estimators']}_d{params['max_depth']}_min{params['min_samples_split']}"
    
    pipe = Pipeline([
        ("preprocessor", preprocessor), 
        ("clf", RandomForestClassifier(**params, random_state=42))
    ])
    pipe.fit(X_train, y_train)

    acc       = accuracy_score(y_test,  pipe.predict(X_test))
    train_acc = accuracy_score(y_train, pipe.predict(X_train))
    
    # model_dir 변수를 활용하여 경로를 일치
    model_uri = f"{model_dir}/pipeline_{run_name}.pkl"
    
    joblib.dump(pipe, model_uri)
    
    run_results.append({
        "run_name": run_name,
        "accuracy": acc,
        "train_acc": train_acc,
        "model_uri": model_uri
    })

    print(f"  {run_name:<30} {acc:>9.4f} {train_acc:>10.4f}  {model_uri}")


# ── 6. 가장 좋은 모델 자동 선택 ─────────────────────────────────────────────
best = max(run_results, key=lambda x: x["accuracy"])
print(f"\n🏆 최고 모델: {best['run_name']} | accuracy: {best['accuracy']:.4f}")

# ── 7. 최고 모델 불러오기 및 상세 평가 ──────────────────────────────────────
model_path = best["model_uri"]
best_pipeline_model = joblib.load(model_path)

y_pred  = best_pipeline_model.predict(X_test)
y_proba = best_pipeline_model.predict_proba(X_test)[:, 1]

print("\n" + "="*65)
print(f"테스트셋 분류 리포트 — {best['run_name']}")
print("="*65)
print(classification_report(y_test, y_pred, target_names=["정상(0)", "사기(1)"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# ── 8. 예제 데이터 예측 ──────────────────────────────────────────────────────
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

print("\n--- 예제 데이터 ---")
print(example_data)

predictions = best_pipeline_model.predict(example_data)
print("\n--- 예측 결과 ---")
print(predictions)
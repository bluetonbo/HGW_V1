import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.optimize import differential_evolution

# 🔹 1️⃣ 엑셀 데이터 로드
file_path = "data.xlsx"  # 파일 경로를 적절히 수정하세요
df = pd.read_excel(file_path, header=0)  # 필요 시 header=1로 변경

df.columns = df.columns.str.strip()  # 공백 제거

# 🔹 2️⃣ 컬럼명 변경
rename_dict = {
    '온도': '가열온도',
    '냉각속도': '냉각',
    '거리(mm)': '거리',
    '라이너 두께': '라이너'
}
df.rename(columns=rename_dict, inplace=True)

# 🔹 3️⃣ 입력 변수(X)와 출력 변수(y) 설정
try:
    X = df[['가열온도', '냉각', '거리', '라이너']]
    y = df['강성']  # 강성 컬럼이 존재하는지 확인 필요
except KeyError:
    print("❌ 컬럼명이 맞지 않습니다. 아래 출력된 컬럼명을 확인하세요.")
    print(df.columns)
    exit()

# 🔹 4️⃣ 데이터 전처리 및 모델 학습
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gradient Boosting Regressor 모델 사용
model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42))
])
model.fit(X_train, y_train)

# 🔹 5️⃣ 최적 강성 값 찾기 (Differential Evolution 사용)
def objective(params):
    params = np.array(params).reshape(1, -1)
    return -model.predict(params)[0]  # 강성 최대화 → 음수 부호 사용

# 변수 범위 설정 (데이터의 최소~최대값)
bounds = [(X[col].min(), X[col].max()) for col in X.columns]

# 최적의 조건 탐색 (DE 알고리즘 적용)
result = differential_evolution(objective, bounds, strategy='best1bin', maxiter=1000, popsize=15, tol=1e-6)

# 🔹 6️⃣ 최적 조건 및 예측값 출력
optimal_params = result.x
predicted_max_stiffness = -result.fun  # 음수 부호를 다시 되돌림

# 결과 저장
output_file = "optimization_results.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("✅ 강성을 최대로 하는 조건:\n")
    for i, col in enumerate(X.columns):
        f.write(f"   {col}: {optimal_params[i]:.2f}\n")
    
    f.write(f"\n🎯 예측된 최대 강성 값: {predicted_max_stiffness:.2f}\n")

print(f"결과가 {output_file}에 저장되었습니다.")

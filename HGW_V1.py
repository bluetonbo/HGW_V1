import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.optimize import differential_evolution

# -------------------------------------------------------------------------
# 1. 스트림릿 웹 페이지 헤더 및 테마 설정
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="HGW 강성 최적화 시스템", 
    layout="centered", 
    initial_sidebar_state="expanded"
)

st.title("HGW 강성 최대화 공정 최적화 시스템")
st.caption("공정 데이터(온도, 냉각, 거리, 라이너)를 바탕으로 AI 모델이 최대 강성을 내는 최적 제어 조건을 찾아냅니다.")
st.markdown("---")

# -------------------------------------------------------------------------
# 2. 사이드바: 웹 전용 엑셀 파일 업로더 구성
# -------------------------------------------------------------------------
st.sidebar.header("데이터 업로드 및 관리")
uploaded_file = st.sidebar.file_uploader(
    "HGW 공정 데이터 엑셀 파일(.xlsx)을 선택하세요.", 
    type=["xlsx"]
)

# -------------------------------------------------------------------------
# 3. 메인 웹 애플리케이션 로직 가동
# -------------------------------------------------------------------------
if uploaded_file is not None:
    try:
        # [1] 업로드된 파일로부터 엑셀 데이터 로드 (openpyxl 엔진 사용)
        df = pd.read_excel(uploaded_file, header=0)
        df.columns = df.columns.str.strip()  # 컬럼명 양끝 공백 제거

        # [2] 사용자 편의를 위한 예외 처리용 컬럼명 변경 딕셔너리
        rename_dict = {
            '온도': '가열온도',
            '냉각속도': '냉각',
            '거리(mm)': '거리',
            '라이너 두께': '라이너'
        }
        df.rename(columns=rename_dict, inplace=True)

        # [3] 입력 변수(X)와 출력 target 변수(y) 검증 및 설정
        required_cols = ['가열온도', '냉각', '거리', '라이너']
        
        # 필수 컬럼이나 '강성' 레이블이 데이터셀에 유실되었는지 검사
        if not all(col in df.columns for col in required_cols) or '강성' not in df.columns:
            st.error("엑셀 파일의 필수 컬럼명이 구조와 일치하지 않습니다. 헤더 명칭을 확인해 주세요.")
            st.write("현재 파일에서 인식된 컬럼 목록:", list(df.columns))
            st.stop()  # 일치하지 않으면 프로세스를 안전하게 일시 중단
            
        X = df[required_cols]
        y = df['강성']

        # 업로드된 원본 데이터의 형상을 확인할 수 있도록 대시보드 확장 탭 제공
        with st.expander("업로드된 데이터셋 미리보기 (상위 5개 행)"):
            st.dataframe(df.head(), use_container_width=True)

        # [4] 데이터 분할 및 머신러닝(Gradient Boosting) 파이프라인 학습
        with st.spinner("AI 예측 엔진 최적화 학습 진행 중..."):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # 데이터 스케일링과 그라디언트 부스팅 회귀 알고리즘 결합
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42))
            ])
            model.fit(X_train, y_train)
            
        st.success("머신러닝 AI 모델 학습이 성공적으로 완료되었습니다.")
        st.markdown("---")

        # [5] 초정밀 유전자 알고리즘(DE) 가동 영역
        st.header("최적 공정 마진 시뮬레이션")
        st.info("아래 버튼을 클릭하면 Differential Evolution 최적화 알고리즘이 가동되어 강성을 극대화하는 수치적 교차점 조합을 탐색합니다.")

        # 최적화 연산 트리거 버튼
        if st.button("AI 초정밀 최적화 시뮬레이션 가동", type="primary", use_container_width=True):
            with st.spinner("유전자 알고리즘 기반 파라미터 조합 탐색 중... (수초 소요)"):
                
                # 목적 함수 정의 (강성 최대화를 유도하기 위해 음수 부호 처리)
                def objective(params):
                    params = np.array(params).reshape(1, -1)
                    return -model.predict(params)[0]

                # 탐색 범위 바운더리 설정 (과적합 방지를 위해 입력 데이터의 min ~ max 한계치로 제한)
                bounds = [(X[col].min(), X[col].max()) for col in X.columns]

                # Differential Evolution 최적화 구동
                result = differential_evolution(
                    objective, bounds, strategy='best1bin', 
                    maxiter=1000, popsize=15, tol=1e-6
                )

                optimal_params = result.x
                predicted_max_stiffness = -result.fun  # 다시 양수 부호로 변환

            # [6] 최적 조건 도출 및 시각화 리포트 화면 구성 (풍선 효과 삭제)
            st.markdown("### 예측된 최대 마진 결과")
            # 스코어 카드로 시인성 확보
            st.metric(label="AI 모델 기준 최적 예측 최대 강성", value=f"{predicted_max_stiffness:.3f}")
            
            st.markdown("### 최대 강성을 위한 최적 공정 제어 매개변수")
            
            # 레이아웃을 바둑판 배열로 나누어 가독성 상향
            col_left, col_right = st.columns(2)
            with col_left:
                st.info(f"**가열온도 제어값**: {optimal_params[0]:.2f}")
                st.info(f"**냉각 속도 제어값**: {optimal_params[1]:.2f}")
            with col_right:
                st.info(f"**공정 거리 제어값**: {optimal_params[2]:.2f}")
                st.info(f"**라이너 두께 제어값**: {optimal_params[3]:.2f}")
                
            # 안내 문구 한 줄 정렬
            st.caption(f"제안된 최적화 결과는 데이터 내 한계 범위 안에서 도출되었습니다. (온도: {X['가열온도'].min():.1f}~{X['가열온도'].max():.1f} / 냉각: {X['냉각'].min():.1f}~{X['냉각'].max():.1f})")

    except Exception as e:
        st.error(f"

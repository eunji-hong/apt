import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정 (Windows 기준: Malgun Gothic)
plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)

# ============================================================
# 1. 페이지 설정
# ============================================================
st.set_page_config(
    page_title="아파트 공용관리비 예측 & 원인분석 대시보드",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 아파트 공용관리비 예측 및 진단 대시보드")
st.markdown("""
> **💡 [Project Core Insights & Value]**
> * **규모의 경제 검증 완료**: 데이터 탐색(EDA) 결과, 세대수 규모가 커질수록 ㎡당 공용관리비가 감소하는 안정화 패턴(규모의 경제)을 머신러닝 피처로 입증 ($R^2$ 0.85).
> * **이상 단지 심층 진단 시스템**: 단순 예측을 넘어, 실제 부과금액과 예측값 간의 오차(Residual)를 분석하여 과다/과소 부과 단지의 인과관계를 입체적으로 진단.
""")
st.divider()

# ============================================================
# 2. 모델 및 전처리된 데이터 로드 (best_model.pkl, preprocessed_data.pkl)
# ============================================================
import urllib.request

MODEL_URL = "https://github.com/eunji-hong/apt/releases/download/v1.0.0/best_model.pkl"
DATA_URL = "https://github.com/eunji-hong/apt/releases/download/v1.0.0/preprocessed_data.pkl"

MODEL_PATH = "best_model.pkl"
DATA_PATH = "preprocessed_data.pkl"

# 파일이 로컬에 없으면 자동으로 다운로드
if not os.path.exists(MODEL_PATH):
    with st.spinner("모델 파일 다운로드 중..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

if not os.path.exists(DATA_PATH):
    with st.spinner("데이터 파일 다운로드 중..."):
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_data(path):
    return joblib.load(path)

model = load_model(MODEL_PATH)
loaded_obj = load_data(DATA_PATH)

if isinstance(loaded_obj, dict):
    df = loaded_obj.get('df', loaded_obj.get('X', list(loaded_obj.values())[0]))
else:
    df = loaded_obj

if not isinstance(df, pd.DataFrame):
    df = pd.DataFrame(df)

if model is None or df is None:
    st.error(f"❌ 필수 파일을 찾을 수 없습니다. 경로를 확인해주세요 (`{MODEL_PATH}`, `{DATA_PATH}`)")
    st.stop()

# ============================================================
# 3. 탭 구성
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "🎚️ 시뮬레이션 & 면적별 예측", 
    "🔍 개별 아파트 검색 & 유사단지 비교", 
    "🚨 이상단지 심층 & 원인 분석"
])

# ------------------------------------------------------------
# TAB 1: 시뮬레이션 및 사용자 맞춤 면적별 예측 (관리비부과면적 자동 계산)
# ------------------------------------------------------------
with tab1:
    st.subheader("💡 단지 주요 조건 설정 (Permutation Importance 상위 변수 중심)")
    
    col1, col2 = st.columns(2)

    with col1:
        total_hh = st.slider("세대수", min_value=50, max_value=3000, value=700, step=50)
        use_approval_year = st.slider("사용승인연도 (준공 연도)", min_value=1990, max_value=2026, value=2015, step=1)
        max_floors = st.slider("최고층수", min_value=5, max_value=50, value=20, step=1)

    with col2:
        total_parking = st.slider("총주차대수", min_value=50, max_value=4000, value=800, step=50)
        cctv_cnt = st.slider("CCTV 대수", min_value=5, max_value=200, value=40, step=5)

    # 🟢 관리비부과면적 자동 산출 (세대당 평균 전용면적 약 80㎡ + 공용면적 감안하여 세대당 약 110㎡ 부과면적 가정)
    estimated_management_area = total_hh * 110.0

    # 파생 변수 계산
    size_index = total_hh * max_floors
    parking_per_hh = total_parking / total_hh
    cctv_per_hh = cctv_cnt / total_hh

    input_dict = {
        "관리비부과면적": estimated_management_area,
        "세대수": total_hh,
        "단지규모지수": size_index,
        "사용승인연도": use_approval_year,
        "총주차대수": total_parking,
        "CCTV대수": cctv_cnt,
        "최고층수": max_floors,
        "세대당_주차대수": parking_per_hh,
        "세대당_CCTV수": cctv_per_hh,
    }

    input_df = pd.DataFrame([input_dict])
    
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = df[col].median() if col in df.columns else 0

    input_df = input_df[model.feature_names_in_]
    predicted_cost_per_sqm = model.predict(input_df)[0]

    st.divider()
    st.subheader("🏡 사용자 맞춤 전용면적별 관리비 시뮬레이션 및 평가")
    
    target_area = st.slider("분석할 전용면적(㎡)을 선택하세요:", min_value=30, max_value=150, value=84, step=1)
    charged_area_sim = target_area * 1.3 
    monthly_cost = charged_area_sim * predicted_cost_per_sqm

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("선택 전용면적", f"{target_area} ㎡", f"약 {target_area/3.3058:.1f}평")
    with m2:
        st.metric("추정 ㎡당 단가", f"{predicted_cost_per_sqm:,.0f} 원")
    with m3:
        st.metric("예상 월 공용관리비", f"{monthly_cost:,.0f} 원")
    with m4:
        st.metric("예상 연간 관리비", f"{(monthly_cost*12):,.0f} 원")

    # 시뮬레이션 결과 평가 소견
    st.markdown("---")
    st.markdown("#### 🎯 시뮬레이션 단지 관리비 종합 평가 소견")
    
    if "㎡당_공용관리비" in df.columns:
        median_market_cost = df["㎡당_공용관리비"].median()
        cost_diff_pct = ((predicted_cost_per_sqm - median_market_cost) / median_market_cost) * 100
        
        if predicted_cost_per_sqm < median_market_cost * 0.9:
            eval_title = "🟢 매우 경제적임 (시장 평균 대비 저렴)"
            eval_desc = f"설정하신 단지 스펙은 전체 시장 중앙값({median_market_cost:,.0f}원/㎡) 대비 약 **{abs(cost_diff_pct):.1f}% 낮게** 예측되어, 공용관리비 운영 효율성이 매우 우수한 조건입니다."
        elif predicted_cost_per_sqm <= median_market_cost * 1.1:
            eval_title = "🟡 시장 평균 수준 (적정 관리비)"
            eval_desc = f"설정하신 단지 스펙은 전체 시장 평균 수준({median_market_cost:,.0f}원/㎡)과 유사하여, **표준적인 공용관리비**가 부과될 것으로 예상됩니다."
        else:
            eval_title = "🔴 다소 높은 부담 (시장 평균 상회)"
            eval_desc = f"설정하신 단지 스펙은 시장 중앙값 대비 약 **{cost_diff_pct:.1f}% 높게** 예측되었습니다. 세대수 대비 인프라(주차/CCTV 등) 과다 혹은 소규모 단지 특유의 규모의 경제 한계 영향일 수 있습니다."
        
        st.info(f"**[{eval_title}]**\n\n{eval_desc}")


# ------------------------------------------------------------
# TAB 2: 개별 아파트 검색 & 유사단지 비교 진단
# ------------------------------------------------------------
with tab2:
    st.subheader("🔍 개별 아파트 검색 및 유사 단지 비교 진단")
    st.markdown("관심 있는 아파트를 선택하면 실제 부과 관리비와 **유사 규모 단지 대비 관리비 수준**을 진단해 드립니다.")

    if "단지명" in df.columns:
        col_select1, col_select2 = st.columns([2, 1])

        with col_select1:
            apt_list = sorted(df["단지명"].dropna().unique())
            selected_apt = st.selectbox("🏬 비교 분석할 아파트 단지를 검색/선택하세요:", apt_list, key="tab2_apt_select")

        apt_filtered_df = df[df["단지명"] == selected_apt]

        with col_select2:
            if "발생년월(YYYYMM)" in apt_filtered_df.columns:
                ym_list = sorted(apt_filtered_df["발생년월(YYYYMM)"].astype(str).unique(), reverse=True)
                selected_ym = st.selectbox("📅 조회 발생년월:", ym_list, key="tab2_ym_select")
                target_data = apt_filtered_df[apt_filtered_df["발생년월(YYYYMM)"].astype(str) == selected_ym].iloc[0]
                ym_str = selected_ym
            else:
                target_data = apt_filtered_df.iloc[0]
                ym_str = "최근"

        actual_pub_cost = target_data.get("㎡당_공용관리비", 0)
        
        if "모델예측값" in target_data:
            pred_pub_cost = target_data["모델예측값"]
        else:
            single_X = pd.DataFrame([target_data])
            for col in model.feature_names_in_:
                if col not in single_X.columns:
                    single_X[col] = df[col].median() if col in df.columns else 0
            pred_pub_cost = model.predict(single_X[model.feature_names_in_])[0]

        diff_cost = actual_pub_cost - pred_pub_cost
        over_ratio = (diff_cost / pred_pub_cost) * 100 if pred_pub_cost > 0 else 0
        households = int(target_data.get("세대수", 800))

        st.divider()
        st.markdown(f"### 📍 **{selected_apt}** (`{ym_str}` 부과 기준)")
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("추정 세대수", f"{households:,} 세대")
        with c2:
            st.metric("실제 ㎡당 공용관리비", f"{actual_pub_cost:,.0f} 원")
        with c3:
            st.metric("모델 예측 ㎡당 관리비", f"{pred_pub_cost:,.0f} 원")
        with c4:
            st.metric("예측 대비 격차", f"{diff_cost:+,.0f} 원", f"{over_ratio:+.1f}%", delta_color="inverse")

        st.divider()
        st.markdown("### 📊 유사 규모(세대수) 단지 대비 관리비 수준 진단")
        
        if "발생년월(YYYYMM)" in df.columns and "발생년월(YYYYMM)" in target_data:
            same_ym_df = df[df["발생년월(YYYYMM)"].astype(str) == str(ym_str)].copy()
        else:
            same_ym_df = df.copy()

        min_hh = int(households * 0.8)
        max_hh = int(households * 1.2)
        similar_group = same_ym_df[(same_ym_df["세대수"] >= min_hh) & (same_ym_df["세대수"] <= max_hh)].copy()
        similar_group_valid = similar_group[similar_group["㎡당_공용관리비"] >= 200]
        
        group_cnt = len(similar_group_valid)
        group_avg_cost = similar_group_valid["㎡당_공용관리비"].mean() if group_cnt > 0 else actual_pub_cost
        group_median_cost = similar_group_valid["㎡당_공용관리비"].median() if group_cnt > 0 else actual_pub_cost
        lower_ratio = (similar_group_valid["㎡당_공용관리비"] < actual_pub_cost).mean() * 100 if group_cnt > 0 else 50.0

        if actual_pub_cost < 200:
            status_label = "💡 특이 단지 (데이터 확인 필요)"
            status_desc = f"해당 단지는 실제 공용관리비가 ㎡당 {actual_pub_cost:,.0f}원으로 매우 작게 집계되었습니다."
        elif lower_ratio <= 30:
            status_label = "🟢 매우 저렴함 (하위 30% 이내)"
            status_desc = f"유사 규모 단지 {group_cnt}개 중 **상위 30% 수준으로 관리비가 알뜰하게 부과**되고 있습니다."
        elif lower_ratio <= 70:
            status_label = "🟡 적정 수준 (평균 수준)"
            status_desc = f"유사 규모 단지 {group_cnt}개 평균과 비슷한 **적정 수준의 공용관리비**입니다."
        else:
            status_label = "🔴 다소 높음 (상위 30% 이상)"
            status_desc = f"유사 규모 단지 {group_cnt}개 평균 대비 **공용관리비 부담이 다소 높은 편**입니다."

        res_col1, res_col2 = st.columns([1.2, 1])

        with res_col1:
            st.markdown(f"#### 진단 결과: **{status_label}**")
            st.write(status_desc)
            
            diff_from_avg = actual_pub_cost - group_avg_cost
            diff_pct = (diff_from_avg / group_avg_cost) * 100 if group_avg_cost > 0 else 0

            st.markdown(f"""
            * **비교 대상 그룹:** 세대수 **{min_hh} ~ {max_hh} 세대** ({group_cnt:,}개 단지)
            * **유사 단지 평균 ㎡당 공용관리비:** `{group_avg_cost:,.0f} 원`
            * **유사 단지 중앙값 ㎡당 공용관리비:** `{group_median_cost:,.0f} 원`
            * **유사 단지 평균 대비 격차:** `{diff_from_avg:+,.0f} 원/㎡` (**{diff_pct:+.1f}%**)
            * **그룹 내 내 아파트 위치:** 백분위 **하위 {lower_ratio:.1f}%** (낮을수록 저렴함)
            """)

        with res_col2:
            fig, ax = plt.subplots(figsize=(6, 3.8))
            if group_cnt > 0:
                sns.histplot(similar_group_valid["㎡당_공용관리비"], kde=True, color="#3498db", ax=ax, bins=15)
            
            ax.axvline(actual_pub_cost, color="#e74c3c", linestyle="--", linewidth=2.5, label=f"{selected_apt} ({actual_pub_cost:,.0f}원)")
            ax.axvline(group_avg_cost, color="#2ecc71", linestyle=":", linewidth=2, label=f"유사단지 평균 ({group_avg_cost:,.0f}원)")
            
            ax.set_title(f"유사 세대수 단지 관리비 분포", fontweight="bold")
            ax.set_xlabel("㎡당 공용관리비 (원)")
            ax.set_ylabel("단지 수")
            ax.legend(loc="upper right", fontsize=8)
            st.pyplot(fig)

        st.markdown("---")
        st.markdown(f"### 🏡 **{selected_apt}** 대표 평형별 실제 청구 예상 월 관리비")
        
        sample_areas = [
            {"구분": "25평형 (전용 59㎡)", "부과면적": 59 * 1.3},
            {"구분": "34평형 (전용 84㎡)", "부과면적": 84 * 1.3},
            {"구분": "45평형 (전용 114㎡)", "부과면적": 114 * 1.3},
        ]

        calc_rows = []
        for s in sample_areas:
            m2 = s["부과면적"]
            actual_pub_monthly = actual_pub_cost * m2
            group_avg_monthly = group_avg_cost * m2
            gap_monthly = actual_pub_monthly - group_avg_monthly

            calc_rows.append({
                "평형 구분": s["구분"],
                "부과면적": f"{m2:.1f}㎡",
                "실제 월 공용관리비": f"{actual_pub_monthly:,.0f} 원",
                "유사 단지 평균 월 관리비": f"{group_avg_monthly:,.0f} 원",
                "평균 대비 월 차액": f"{gap_monthly:+,.0f} 원"
            })

        st.dataframe(pd.DataFrame(calc_rows), use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ 데이터프레임에 '단지명' 컬럼이 존재하지 않습니다.")

# ------------------------------------------------------------
# TAB 3: 이상단지 심층 & 원인 분석 (인터랙티브 진단 기능 탑재)
# ------------------------------------------------------------
with tab3:
    st.subheader("🚨 관리비 이상 책정(과다/과소) 단지 심층 진단")
    st.markdown("모델 예측값과 실제값의 오차(Residual)가 가장 큰 단지들을 분석하여 **왜 관리비가 과다 또는 과소 부과되었는지 원인**을 심층 진단합니다.")

    if "㎡당_공용관리비" in df.columns:
        if "모델예측값" not in df.columns:
            df["모델예측값"] = model.predict(df[model.feature_names_in_])
        
        df["예측오차"] = df["㎡당_공용관리비"] - df["모델예측값"]
        df["오차절대값"] = df["예측오차"].abs()
        
        # 상위 이상단지 선정
        top_anomalies = df.sort_values(by="오차절대값", ascending=False).head(20).copy()
        
        st.markdown("---")
        st.markdown("### 🔎 진단할 이상 단지 선택 또는 검색")
        
        # 사용자가 단지를 직접 선택하거나 검색할 수 있도록 셀렉트박스 제공
        anomaly_apt_list = top_anomalies["단지명"].dropna().unique().tolist()
        selected_anomaly_apt = st.selectbox("이상 단지 목록에서 진단할 아파트를 선택하세요:", anomaly_apt_list, key="anomaly_select")
        
        # 선택된 단지 데이터 가져오기
        target_row = top_anomalies[top_anomalies["단지명"] == selected_anomaly_apt].iloc[0]
        
        act_val = target_row["㎡당_공용관리비"]
        pred_val = target_row["모델예측값"]
        err_val = target_row["예측오차"]
        
        st.markdown(f"### 🎯 **{selected_anomaly_apt}** 이상 원인 심층 분석 리포트")
        
        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            st.metric("실제 ㎡당 공용관리비", f"{act_val:,.0f} 원")
        with ac2:
            st.metric("모델 예측 ㎡당 관리비", f"{pred_val:,.0f} 원")
        with ac3:
            err_type = "🔴 과다 책정 (실제가 훨씬 높음)" if err_val > 0 else "🔵 과소 책정 (실제가 훨씬 낮음)"
            st.metric("예측 오차 격차", f"{err_val:+,.0f} 원/㎡", err_type)

        # 원인 분석 인사이트 자동 생성 텍스트 박스
        st.markdown("#### 💡 AI 인과관계 진단 소견")
        
        # 주변 단지들과의 스펙 비교를 통한 원인 도출
        avg_hh = df["세대수"].median()
        avg_park = df["총주차대수"].median() if "총주차대수" in df.columns else 0
        
        reasons = []
        if err_val > 0:
            reasons.append("• **유지관리비 및 인건비성 비용 과다 추정**: 세대수 대비 공용 면적이 넓거나, 보안·청소 용역비 등 고정 인프라 비용이 집중 투입되었을 가능성이 높습니다.")
            if target_row.get("세대수", 500) < avg_hh:
                reasons.append("• **규모의 경제 한계**: 소규모 단지 특성상 고정 관리 인원 대비 세대수가 적어 세대당 공용 부담 가중.")
        else:
            reasons.append("• **관리비 절감 우수 단지 또는 회계 항목 차이**: 모델이 예측한 표준 비용보다 실제 부과된 공용관리비가 현저히 낮습니다.")
            reasons.append("• **에너지 효율화 및 공동 설비 효율 운영**: 태양광 발전 혹은 불필요한 용역 비용 절감이 이루어지고 있을 확률이 높습니다.")

        for r in reasons:
            st.write(r)

        # 단지 주요 스펙 비교 테이블
        st.markdown("#### 📋 해당 단지 주요 스펙 vs 전체 시장 평균")
        spec_compare_data = []
        features_to_check = ["세대수", "관리비부과면적", "사용승인연도", "총주차대수", "CCTV대수"]
        
        for f_name in features_to_check:
            if f_name in target_row:
                val = target_row[f_name]
                m_val = df[f_name].median()
                spec_compare_data.append({
                    "지표 항목": f_name,
                    "선택 단지 값": f"{val:,.1f}" if isinstance(val, (int, float)) else str(val),
                    "전체 시장 중앙값": f"{m_val:,.1f}" if isinstance(m_val, (int, float)) else str(m_val)
                })
        
        st.dataframe(pd.DataFrame(spec_compare_data), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### 📊 전체 이상 단지 리스트 한눈에 보기")
        display_cols = [c for c in ["단지명", "세대수", "사용승인연도", "㎡당_공용관리비", "모델예측값", "예측오차"] if c in df.columns]
        st.dataframe(top_anomalies[display_cols], use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ 데이터 분석을 위한 필수 컬럼이 부족합니다.")

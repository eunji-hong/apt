import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정 (Windows 기준: Malgun Gothic, OS별 호환성 처리)
plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)

# ============================================================
# 1. 페이지 설정
# ============================================================
st.set_page_config(
    page_title="아파트 공용관리비 예측 & 이상단지 원인분석 대시보드",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 아파트 공용관리비 예측 & 이상단지 원인분석 대시보드")
st.divider()
import os
import joblib
import urllib.request
import pandas as pd
import streamlit as st
# ============================================================
# 2. 모델 및 데이터 로드 (분할 다운로드로 메모리 최적화)
# ============================================================

# 1) 로컬 저장 경로 설정
MODEL_LOCAL_PATH = "model/apt_management_cost_rf.joblib"

# 2) GitHub Release 웹 URL 설정
MODEL_URL = "https://github.com/eunji-hong/apt/releases/download/v1.0.0/apt_management_cost_rf.joblib"
ANOMALY_URL = "https://github.com/eunji-hong/apt/releases/download/v1.0.0/fee_more.csv"
ALL_DATA_URL = "https://github.com/eunji-hong/apt/releases/download/v1.0.0/fee_result.csv"

# 모델 로드 함수 (1MB씩 청크 단위로 나누어 받아서 RAM 부하 방지)
@st.cache_resource
def load_saved_model(local_path, url):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    if not os.path.exists(local_path):
        with st.spinner("📥 대용량 모델 파일을 안전하게 다운로드 중입니다... (최초 1회만 진행)"):
            try:
                # stream=True 옵션으로 메모리 일시 점유 최소화
                response = requests.get(url, stream=True, timeout=120)
                response.raise_for_status()
                
                # 1MB(1024*1024 bytes) 단위로 나누어 디스크에 작성
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            except Exception as e:
                st.error(f"❌ 모델 다운로드 실패: {e}")
                return None
                
    return joblib.load(local_path)

# CSV 데이터 로드 함수
@st.cache_data
def load_csv_data(url_or_path):
    try:
        return pd.read_csv(url_or_path)
    except Exception as e:
        st.error(f"❌ CSV 파일 로드 실패 (`{url_or_path}`): {e}")
        return None

# 데이터 읽어오기 실행
saved_data = load_saved_model(MODEL_LOCAL_PATH, MODEL_URL)
anomaly_df_raw = load_csv_data(ANOMALY_URL)
all_df_raw = load_csv_data(ALL_DATA_URL)

if saved_data is None:
    st.error(f"❌ 모델 파일을 찾을 수 없거나 불러올 수 없습니다. 경로를 확인해주세요: `{MODEL_URL}`")
    st.stop()

rf_model = saved_data["model"]
encoder = saved_data["encoder"]
numeric_features = saved_data["numeric_features"]
categorical_features = saved_data["categorical_features"]

# ============================================================
# 3. 탭 구성
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "🎚️ 시뮬레이션 & 면적별 예측", 
    "🚨 이상단지 심층 & 원인 분석", 
    "🔍 개별 아파트 검색 & 유사단지 비교"
])

# ------------------------------------------------------------
# TAB 1: 시뮬레이션 및 사용자 맞춤 면적별 예측 (슬라이더 적용)
# ------------------------------------------------------------
with tab1:
    st.subheader("💡 단지 조건 설정")
    col1, col2 = st.columns([1, 1])

    with col1:
        apt_age = st.slider("아파트 연식 (년)", min_value=0, max_value=40, value=10, step=1)
        household_cnt = st.slider("총 세대수", min_value=100, max_value=3000, value=750, step=50)
        elevator_cnt = st.slider("승강기 대수", min_value=1, max_value=100, value=15, step=1)

    with col2:
        try:
            mgmt_options = list(encoder.categories_[0])
            heat_options = list(encoder.categories_[1])
        except AttributeError:
            mgmt_options = ["위탁관리", "자치관리", "기타"]
            heat_options = ["지역난방", "개별난방", "중앙난방", "기타"]

        mgmt_type = st.selectbox("관리방식", mgmt_options)
        heat_type = st.selectbox("난방방식", heat_options)
        
        elevator_per_household = elevator_cnt / household_cnt
        st.info(f"💡 **세대당 승강기 수:** 약 {elevator_per_household:.3f} 대 / 세대")

    # 예측 모델 계산 (㎡당 단가 도출)
    log_household_cnt = np.log1p(household_cnt)
    input_data = pd.DataFrame([{
        "아파트연식": apt_age, "세대수": household_cnt, "log_세대수": log_household_cnt,
        "승강기대수": elevator_cnt, "세대당_승강기수": elevator_per_household,
        "관리방식": mgmt_type, "난방방식": heat_type
    }])

    encoded_cat = encoder.transform(input_data[categorical_features])
    encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(categorical_features), index=input_data.index)
    X_input = pd.concat([input_data[numeric_features], encoded_cat_df], axis=1)
    
    # ㎡당 공용관리비 예측값 (원/㎡)
    predicted_cost_per_sqm = rf_model.predict(X_input)[0]

    st.divider()

    # ------------------------------------------------------------
    # 🔥 전용면적 슬라이더 및 맞춤형 관리비 계산
    # ------------------------------------------------------------
    st.subheader("🏡 사용자 맞춤 전용면적별 관리비 시뮬레이션")

    # 빠른 평형 선택 버튼 (원클릭 설정용)
    st.markdown("**⚡ 대표 평형 빠른 선택**")
    btn_col1, btn_col2, btn_col3, btn_col4, btn_col5 = st.columns(5)
    
    # 세션 상태(Session State)를 활용한 슬라이더 값 동기화
    if "user_area" not in st.session_state:
        st.session_state.user_area = 84

    if btn_col1.button("21평 (전용 49㎡)"):
        st.session_state.user_area = 49
    if btn_col2.button("25평 (전용 59㎡)"):
        st.session_state.user_area = 59
    if btn_col3.button("34평 (전용 84㎡)"):
        st.session_state.user_area = 84
    if btn_col4.button("41평 (전용 102㎡)"):
        st.session_state.user_area = 102
    if btn_col5.button("45평 (전용 114㎡)"):
        st.session_state.user_area = 114

    # 면적 슬라이더 (20㎡ ~ 200㎡)
    target_area = st.slider(
        "분석할 아파트의 전용면적(㎡)을 선택하세요:",
        min_value=20,
        max_value=200,
        value=st.session_state.user_area,
        step=1,
        key="user_area"
    )

    # 부과면적 계산 (전용면적의 약 1.3배 적용, 단지/아파트별 공용면적비율 보정)
    charged_area = target_area * 1.3
    monthly_cost = charged_area * predicted_cost_per_sqm
    yearly_cost = monthly_cost * 12

    # 결과 카드 표출
    st.markdown("#### 📊 시뮬레이션 결과")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.metric("선택 전용면적", f"{target_area} ㎡", help=f"약 {target_area / 3.3058:.1f} 평")
    with m_col2:
        st.metric("추정 부과면적", f"{charged_area:.1f} ㎡", help="전용면적 × 1.3 (공용면적 포함)")
    with m_col3:
        st.metric("예상 월 공용관리비", f"{monthly_cost:,.0f} 원")
    with m_col4:
        st.metric("예상 연간 공용관리비", f"{yearly_cost:,.0f} 원")

    # 면적 변경에 따른 예시 비교 대조표 (선택 면적 ± 20% 범위 자동 제공)
    st.markdown("---")
    st.markdown(f"**💡 선택한 {target_area}㎡ 인근 평형과의 비교**")
    
    comp_areas = [
        max(20, int(target_area * 0.8)),
        target_area,
        min(200, int(target_area * 1.2))
    ]
    comp_areas = sorted(list(set(comp_areas)))

    comp_list = []
    for a in comp_areas:
        c_area = a * 1.3
        m_cost = c_area * predicted_cost_per_sqm
        comp_list.append({
            "구분": "👉 현재 선택 면적" if a == target_area else f"{a}㎡ 평형",
            "전용면적": f"{a}㎡ ({a/3.3058:.1f}평)",
            "부과면적": f"{c_area:.1f}㎡",
            "예상 ㎡당 단가": f"{predicted_cost_per_sqm:,.0f} 원",
            "예상 월 공용관리비": f"{m_cost:,.0f} 원",
            "예상 연간 공용관리비": f"{(m_cost * 12):,.0f} 원"
        })

    st.dataframe(pd.DataFrame(comp_list), use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# TAB 2: 이상단지 심층 & 원인 분석 (예외 분류 로직 적용)
# ------------------------------------------------------------
with tab2:
    st.subheader("🚨 예측 모델 기준 관리비 과다 청구(이상) 단지 및 원인 분석")
    st.markdown("Random Forest 모델 예측 대비 공용관리비가 이상 책정된 단지의 주요 원인을 진단합니다.")

    if anomaly_df_raw is None or len(anomaly_df_raw) == 0:
        st.warning("⚠️ 이상단지 데이터 파일(`data/model_results/관리비_이상단지.csv`)이 없거나 비어있습니다.")
    else:
        # 개별 이상단지 선택
        anomaly_df_raw["선택라벨"] = anomaly_df_raw["단지명"] + " (" + anomaly_df_raw["발생년월(YYYYMM)"].astype(str) + ")"
        selected_apt_label = st.selectbox("분석할 이상단지를 선택하세요:", anomaly_df_raw["선택라벨"].unique())

        target_row = anomaly_df_raw[anomaly_df_raw["선택라벨"] == selected_apt_label].iloc[0]

        actual_cost = target_row["㎡당_공용관리비"]
        pred_cost = target_row["예측_㎡당_공용관리비"]
        diff_cost = target_row["예측오차"]
        over_pct = target_row["예측대비_초과비율"] * 100

        # 기본 지표 카드 표출
        st.info(
            f"📍 **{target_row['단지명']}** | "
            f"실제 ㎡당 관리비: **{actual_cost:,.0f}원** | "
            f"예측 ㎡당 관리비: **{pred_cost:,.0f}원** | "
            f"격차: **+{diff_cost:,.0f}원/㎡** (**{over_pct:+.1f}%**)"
        )

        st.markdown("### 🔎 이상 원인 자동 진단 리포트")

        # ------------------------------------------------------------
        # 🔥 예외 분기: 실제 공용관리비가 비현실적으로 낮거나 수집이 왜곡된 경우
        # ------------------------------------------------------------
        if actual_cost < 200 or pred_cost < 200:
            st.error("⚠️ **[특이 단지 경고] 데이터 누락 또는 관리비 집행 방식 특수 케이스**")
            st.markdown(f"""
            - **원인 분석:** 해당 단지는 ㎡당 공용관리비 수치가 **{actual_cost:,.0f}원**으로 비현실적으로 낮게 집계되었습니다.
            - **상세 사유:** 
              1. K-apt 공개 항목 중 장기수선충당금, 청소/경비 인건비 등 **공용관리비 일부 항목 누락**
              2. 자치관리(직영) 또는 단지 특수 운영에 따른 **관리비 통합/개별 집행 방식 차이**
              3. 단지 전체 관리비 총액과 ㎡당 단가 계산 과정에서의 **데이터 단위 오류**
            - **조치 권고:** 해당 단지는 과다 청구 단지가 아니며, 원본 데이터 수집 항목의 정밀 조회가 필요합니다.
            """)
        else:
            # 정상적인 과다 청구 이상단지 분석 logic
            reasons = []
            avg_households = 800
            avg_elevator_ratio = 0.02

            if "세대수" in target_row and target_row["세대수"] < 300:
                reasons.append(f"📉 **소규모 세대수 ({target_row['세대수']:,}세대):** 규모의 경제 미적용으로 인한 고정 인건비/위탁비 분담 단가 상승")
            if "세대당_승강기수" in target_row and target_row["세대당_승강기수"] > (avg_elevator_ratio * 1.3):
                reasons.append(f"🛗 **세대당 승강기 과다 ({target_row['세대당_승강기수']:.3f}대/세대):** 승강기 유지보수비 및 승강기 공용전기료 부담 가중")
            if "아파트연식" in target_row and target_row["아파트연식"] > 20:
                reasons.append(f"🛠️ **노후 단지 ({target_row['아파트연식']}년차):** 시설물 수선유지비 및 공용부 보수비용 지속 발생")
            
            reasons.append("💸 **기타:** 위탁수수료, 용역비 등 개별 관리비 항목의 비효율적 집행 가능성")

            st.warning("⚠️ **모델 분석 기반 예측 대비 초과 원인 추정**")
            for r in reasons:
                st.markdown(f"- {r}")

        st.markdown("---")

        # 면적별 실제 vs 예측 비교 계산 표
        area_sizes = [
            {"평형": "전용 59㎡ (25평형)", "부과면적": 59 * 1.3},
            {"평형": "전용 84㎡ (34평형)", "부과면적": 84 * 1.3},
            {"평형": "전용 114㎡ (45평형)", "부과면적": 114 * 1.3},
        ]
        
        diff_analysis = []
        for a in area_sizes:
            area_m2 = a["부과면적"]
            act_total = actual_cost * area_m2
            pred_total = pred_cost * area_m2
            gap = act_total - pred_total
            
            diff_analysis.append({
                "평형": a["평형"],
                "부과면적(㎡)": f"{area_m2:.1f}㎡",
                "실제 월 관리비": f"{act_total:,.0f} 원",
                "예측 적정 관리비": f"{pred_total:,.0f} 원",
                "월 차액": f"{gap:+,.0f} 원",
                "연간 차액": f"{(gap * 12):+,.0f} 원"
            })

        st.dataframe(pd.DataFrame(diff_analysis), use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# TAB 3: 개별 아파트 검색 & 유사단지 비교 진단 (최종 통합)
# ------------------------------------------------------------
def render_apt_search_section(df_all):
    st.subheader("🔍 개별 아파트 검색 및 유사 단지 비교 진단")
    st.markdown("관심 있는 아파트를 선택하면 실제 부과 관리비와 **유사 규모 단지 대비 관리비 수준**을 진단해 드립니다.")

    if df_all is None or len(df_all) == 0:
        st.warning("⚠️ 전체 아파트 관리비 예측 결과 데이터파일(`data/model_results/관리비_예측결과.csv`)이 없거나 비어있습니다.")
        return

    # 1. 아파트 단지 및 발생년월 선택 (2단계 드롭다운 필터)
    col_select1, col_select2 = st.columns([2, 1])

    with col_select1:
        apt_list = sorted(df_all["단지명"].dropna().unique())
        selected_apt = st.selectbox("🏬 비교 분석할 아파트 단지를 검색/선택하세요:", apt_list)

    apt_filtered_df = df_all[df_all["단지명"] == selected_apt]

    with col_select2:
        ym_list = sorted(apt_filtered_df["발생년월(YYYYMM)"].astype(str).unique(), reverse=True)
        selected_ym = st.selectbox("📅 조회 발생년월:", ym_list)

    target_data = apt_filtered_df[apt_filtered_df["발생년월(YYYYMM)"].astype(str) == selected_ym].iloc[0]

    apt_name = target_data["단지명"]
    ym = target_data["발생년월(YYYYMM)"]
    actual_pub_cost = target_data["㎡당_공용관리비"]
    pred_pub_cost = target_data["예측_㎡당_공용관리비"]
    diff_cost = target_data["예측오차"]
    over_ratio = target_data["예측대비_초과비율"] * 100

    households = int(target_data["세대수"]) if "세대수" in target_data and pd.notnull(target_data["세대수"]) else 800

    st.divider()

    # 2. 선택 단지 기본 현황 요약 카드
    st.markdown(f"### 📍 **{apt_name}** (`{ym}` 부과 기준)")
    
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

    # 3. 유사 세대수(±20%) 단지 비교 및 수준 진단
    st.markdown("### 📊 유사 규모(세대수) 단지 대비 관리비 수준 진단")
    
    same_ym_df = df_all[df_all["발생년월(YYYYMM)"].astype(str) == str(ym)].copy()
    
    if "세대수" in same_ym_df.columns:
        min_hh = int(households * 0.8)
        max_hh = int(households * 1.2)
        similar_group = same_ym_df[(same_ym_df["세대수"] >= min_hh) & (same_ym_df["세대수"] <= max_hh)].copy()
    else:
        min_hh, max_hh = "전체", "전체"
        similar_group = same_ym_df.copy()

    similar_group_valid = similar_group[similar_group["㎡당_공용관리비"] >= 200]
    
    group_cnt = len(similar_group_valid)
    group_avg_cost = similar_group_valid["㎡당_공용관리비"].mean() if group_cnt > 0 else actual_pub_cost
    group_median_cost = similar_group_valid["㎡당_공용관리비"].median() if group_cnt > 0 else actual_pub_cost

    lower_ratio = (similar_group_valid["㎡당_공용관리비"] < actual_pub_cost).mean() * 100 if group_cnt > 0 else 50.0

    if actual_pub_cost < 200:
        status_label = "💡 특이 단지 (데이터 확인 필요)"
        status_desc = f"해당 단지는 실제 공용관리비가 ㎡당 {actual_pub_cost:,.0f}원으로 매우 작게 집계되었습니다. 데이터 작성 누락 또는 자치 직영 관리 여부를 확인하세요."
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
        * **비교 대상 그룹:** `{ym}` 기준 세대수 **{min_hh} ~ {max_hh} 세대** ({group_cnt:,}개 단지)
        * **유사 단지 평균 ㎡당 공용관리비:** `{group_avg_cost:,.0f} 원`
        * **유사 단지 중앙값 ㎡당 공용관리비:** `{group_median_cost:,.0f} 원`
        * **유사 단지 평균 대비 격차:** `{diff_from_avg:+,.0f} 원/㎡` (**{diff_pct:+.1f}%**)
        * **그룹 내 내 아파트 위치:** 백분위 **하위 {lower_ratio:.1f}%** (낮을수록 저렴함)
        """)

    with res_col2:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        if group_cnt > 0:
            sns.histplot(similar_group_valid["㎡당_공용관리비"], kde=True, color="#3498db", ax=ax, bins=15)
        
        ax.axvline(actual_pub_cost, color="#e74c3c", linestyle="--", linewidth=2.5, label=f"{apt_name} ({actual_pub_cost:,.0f}원)")
        ax.axvline(group_avg_cost, color="#2ecc71", linestyle=":", linewidth=2, label=f"유사단지 평균 ({group_avg_cost:,.0f}원)")
        
        ax.set_title(f"유사 세대수 단지 관리비 분포 ({ym})", fontweight="bold")
        ax.set_xlabel("㎡당 공용관리비 (원)")
        ax.set_ylabel("단지 수")
        ax.legend()
        st.pyplot(fig)

    # 4. 대표 평형별 실제 청구 월 관리비 계산표
    st.markdown("---")
    st.markdown(f"### 🏡 **{apt_name}** 대표 평형별 실제 청구 예상 월 관리비")
    
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

with tab3:
    render_apt_search_section(all_df_raw)

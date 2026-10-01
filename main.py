import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 페이지 기본 설정
st.set_page_config(page_title="서울 기온 예측기", layout="centered")

st.title("🌡️ 서울 연평균 기온 예측기")
st.write("1908년부터의 서울 기온 데이터를 바탕으로 선형 회귀 분석을 진행하여 연도별 예상 기온을 예측합니다.")

# 데이터 불러오기 함수 (캐싱 처리)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding="utf-8")
    
    # 날짜 컬럼을 datetime 형식으로 변환 및 연도 추출
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    
    # 연도별 관측일수 및 평균기온 계산
    yearly_summary = df.groupby("연도").agg(
        관측일수=("평균기온", "count"),
        연평균기온=("평균기온", "mean")
    ).reset_index()
    
    # 조건 필터링: 2025년 이하 & 관측일수 300일 이상
    filtered_df = yearly_summary[
        (yearly_summary["연도"] <= 2025) & 
        (yearly_summary["관측일수"] >= 300)
    ].copy()
    
    # 1908년부터 경과한 연수 계산 (독립변수 X)
    filtered_df["지난연수"] = filtered_df["연도"] - 1908
    
    return filtered_df

try:
    df_filtered = load_data()

    # 데이터 통계 정보 추출
    num_years = len(df_filtered)
    start_year = int(df_filtered["연도"].min())
    end_year = int(df_filtered["연도"].max())

    # 선형 회귀 계산 (독립변수 X: 지난연수, 종속변수 y: 연평균기온)
    X = df_filtered["지난연수"].values
    y = df_filtered["연평균기온"].values

    # 기울기(slope)와 절편(intercept) 계산
    slope, intercept = np.polyfit(X, y, 1)

    # 상관계수 계산
    correlation = np.corrcoef(X, y)[0, 1]

    # 기본 정보 출력
    st.info(f"📊 **분석 데이터 정보:** 총 **{num_years}개** 연도 데이터 사용 ({start_year}년 ~ {end_year}년)")

    # 1. 기온 예측 슬라이더 세션
    st.subheader("🔮 연도별 기온 예측")
    target_year = st.slider("예측할 연도를 선택하세요:", min_value=1900, max_value=2100, value=2026, step=1)

    # 선택한 연도의 예상 기온 계산
    predicted_temp = slope * (target_year - 1908) + intercept
    
    # 예상 기온 강조 표시
    st.metric(label=f"🗓️ {target_year}년 예상 연평균 기온", value=f"{predicted_temp:.2f} °C")

    # 2. 산점도 및 회귀선 그래프
    st.subheader("📈 연평균 기온 변화 및 회귀 직선")
    
    # 회귀선 추세선 데이터 생성 (1900년 ~ 2100년 확장)
    trend_years = np.arange(1900, 2101)
    trend_X = trend_years - 1908
    trend_y = slope * trend_X + intercept

    fig = go.Figure()

    # 실제 데이터 산점도 추가
    fig.add_trace(go.Scatter(
        x=df_filtered["연도"],
        y=df_filtered["연평균기온"],
        mode="markers",
        name="관측 연평균기온",
        marker=dict(color="royalblue", size=7)
    ))

    # 회귀선 추가
    fig.add_trace(go.Scatter(
        x=trend_years,
        y=trend_y,
        mode="lines",
        name="회귀 직선",
        line=dict(color="firebrick", width=2, dash="dash")
    ))

    # 선택한 연도의 예측 지점 표시
    fig.add_trace(go.Scatter(
        x=[target_year],
        y=[predicted_temp],
        mode="markers",
        name=f"선택 연도 ({target_year}년)",
        marker=dict(color="orange", size=12, symbol="star")
    ))

    # 그래프 레이아웃 설정
    fig.update_layout(
        title=f"서울 연도별 평균기온 (상관계수: {correlation:.4f})",
        xaxis_title="연도",
        yaxis_title="평균기온 (°C)",
        hovermode="x unified",
        xaxis=dict(range=[1895, 2105])
    )

    st.plotly_chart(fig, use_container_width=True)

    # 상관계수 및 회귀식 부연 설명
    st.write(f"- **상관계수:** `{correlation:.4f}`")
    st.write(f"- **회귀 방정식:** `예상 기온 = {slope:.4f} × (연도 - 1908) + {intercept:.4f}`")

except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")

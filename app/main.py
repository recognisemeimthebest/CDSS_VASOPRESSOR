"""Streamlit dashboard — Vasopressor CDSS Demo.

Layout
------
Sidebar  : patient selector + cohort info
Main col : 1) honesty banner  2) patient vitals timeline
           3) recommendation cards (GBM | dBCQ) side-by-side
           4) agreement badge + clinician actual action
"""
from __future__ import annotations

import torch  # must import before pandas on Windows (DLL order)

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (
    ACTION_COLORS,
    ACTION_LABELS,
    LAB_DISPLAY,
    RUNS_DIR,
    VITALS_DISPLAY,
    get_test_stay_ids,
    load_bc_predictions,
    load_cohort,
    load_dbcq_predictions,
    load_features,
    load_gbm_predictions,
    patient_timeline,
)

st.set_page_config(
    page_title="CDSS — Vasopressor Recommendation",
    page_icon="💊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Honesty banner
# ---------------------------------------------------------------------------
st.warning(
    "⚠️ **연구용 프로토타입 — 임상 사용 절대 불가.** "
    "모든 추천은 오프라인 학습 모델의 예측이며 prospective 검증 미수행. "
    "실제 처방 결정은 반드시 담당 의사가 내려야 합니다.",
    icon="🚫",
)

st.title("💊 Vasopressor CDSS — MIMIC-IV Demo")
st.caption("Supervised (GBM) vs Offline RL (dBCQ) 추천 비교 | Sepsis-3 코호트 13,071 ICU stays")

# ---------------------------------------------------------------------------
# Load data (cached)
# ---------------------------------------------------------------------------
with st.spinner("데이터 로딩 중…"):
    features = load_features()
    cohort = load_cohort()
    gbm_pred = load_gbm_predictions()
    dbcq_pred = load_dbcq_predictions()
    bc_pred = load_bc_predictions()

test_stay_ids = get_test_stay_ids(gbm_pred)

# ---------------------------------------------------------------------------
# Sidebar — patient selector
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("환자 선택")
    st.caption(f"Test fold: {len(test_stay_ids):,} stays")

    stay_id = st.selectbox(
        "Stay ID",
        options=test_stay_ids,
        format_func=lambda x: f"Stay {x}",
    )

    st.divider()
    st.subheader("코호트 요약")
    st.metric("총 ICU stays", "13,071")
    st.metric("입원 사망률", "23.1 %")
    st.metric("Train / Val / Test", "9,149 / 1,961 / 1,961")

    st.divider()
    st.subheader("모델 성능 (test fold)")
    st.markdown("""
| 모델 | macro-F1 | top-2 acc |
|---|---|---|
| GBM | **0.529** | **87.5 %** |
| BC  | WIS 6.94 | ESS 100 % |
| dBCQ | FQE 0.121 | ESS 3.9 % ⚠️ |
""")
    st.divider()
    st.page_link("pages/01_ope_report.py", label="📊 OPE 리포트 보기", icon="📊")

# ---------------------------------------------------------------------------
# Cohort info for selected patient
# ---------------------------------------------------------------------------
cohort_row = cohort[cohort.stay_id == stay_id]
timeline = patient_timeline(features, stay_id)

if cohort_row.empty or timeline.empty:
    st.error(f"Stay {stay_id} 데이터 없음.")
    st.stop()

info = cohort_row.iloc[0]

col_a, col_b, col_c, col_d, col_e = st.columns(5)
col_a.metric("Age", f"{info.get('age', 'N/A'):.0f} yr" if pd.notna(info.get("age")) else "N/A")
col_b.metric("Gender", "Male" if info.get("gender") == "M" else "Female")
col_c.metric("ICU unit", str(info.get("first_careunit", "N/A"))[:20])
col_d.metric("LOS (days)", f"{info.get('los', 0):.1f}")
col_e.metric(
    "Outcome",
    "💀 Died" if info.get("hospital_expire_flag") == 1 else "✅ Survived",
)

# ---------------------------------------------------------------------------
# Vitals + SOFA timeline
# ---------------------------------------------------------------------------
st.subheader("환자 시계열 (4h bin)")

tab_vitals, tab_labs = st.tabs(["Vitals", "Labs / SOFA"])

with tab_vitals:
    vital_cols = [c for c in VITALS_DISPLAY if c in timeline.columns]
    if vital_cols:
        melted = timeline[["time_h"] + vital_cols].melt(
            id_vars="time_h", var_name="variable", value_name="value"
        )
        melted["variable"] = melted["variable"].map(VITALS_DISPLAY)
        fig = px.line(
            melted,
            x="time_h",
            y="value",
            color="variable",
            markers=True,
            labels={"time_h": "Hours in ICU", "value": ""},
            title="Vital Signs",
        )
        fig.update_layout(height=320, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

with tab_labs:
    lab_cols = [c for c in LAB_DISPLAY if c in timeline.columns]
    if lab_cols:
        melted2 = timeline[["time_h"] + lab_cols].melt(
            id_vars="time_h", var_name="variable", value_name="value"
        )
        melted2["variable"] = melted2["variable"].map(LAB_DISPLAY)
        fig2 = px.line(
            melted2,
            x="time_h",
            y="value",
            color="variable",
            markers=True,
            labels={"time_h": "Hours in ICU", "value": ""},
            title="Labs / SOFA / NE dose",
        )
        fig2.update_layout(height=320, margin=dict(t=40, b=20))
        st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# Timestep selector
# ---------------------------------------------------------------------------
st.subheader("추천 조회 시점 선택")
max_bin = int(timeline["t_bin"].max())
t_bin = st.slider(
    "시간 bin (4h 단위)",
    min_value=0,
    max_value=max_bin,
    value=min(4, max_bin),
    format="%d (t=%d h)" % (0, 0),
    help="선택한 시점 이후 4h의 NE 용량을 추천합니다.",
)
st.caption(f"선택: bin {t_bin} → {t_bin * 4}h – {t_bin * 4 + 4}h 구간 처방 추천")

# Current state row
state_row = timeline[timeline.t_bin == t_bin]
if state_row.empty:
    st.warning("해당 bin에 데이터가 없습니다.")
    st.stop()
state = state_row.iloc[0]

# GBM prediction row
gbm_row = gbm_pred[
    (gbm_pred.stay_id == stay_id)
    & (gbm_pred.t_bin == t_bin)
    & (gbm_pred.fold == "test")
]
# dBCQ prediction row
dbcq_row = dbcq_pred[
    (dbcq_pred.stay_id == stay_id)
    & (dbcq_pred.t_bin == t_bin)
    & (dbcq_pred.fold == "test")
]

# ---------------------------------------------------------------------------
# Recommendation cards
# ---------------------------------------------------------------------------
st.subheader("모델 추천 비교")

def _proba_bar_fig(proba: dict[int, float], rec_action: int, title: str) -> go.Figure:
    """Horizontal bar chart of action probabilities."""
    labels = [ACTION_LABELS[a] for a in range(5)]
    values = [proba.get(a, 0.0) for a in range(5)]
    colors = [
        "#1f77b4" if a == rec_action else "#cccccc" for a in range(5)
    ]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1%}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        xaxis=dict(range=[0, 1.1], tickformat=".0%"),
        height=240,
        margin=dict(t=40, b=10, l=10, r=10),
        showlegend=False,
    )
    return fig


def _q_bar_fig(q_vals: dict[int, float], rec_action: int, title: str) -> go.Figure:
    labels = [ACTION_LABELS[a] for a in range(5)]
    values = [q_vals.get(a, 0.0) for a in range(5)]
    colors = ["#ff7f0e" if a == rec_action else "#cccccc" for a in range(5)]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        height=240,
        margin=dict(t=40, b=10, l=10, r=10),
        showlegend=False,
    )
    return fig


col_gbm, col_dbcq = st.columns(2)

# --- GBM card ---
with col_gbm:
    st.markdown("### 🌳 GBM (지도학습)")
    if gbm_row.empty:
        st.info("이 시점 GBM 예측 없음")
        gbm_action = None
    else:
        gr = gbm_row.iloc[0]
        gbm_action = int(gr["y_pred"])
        clinician_action = int(gr["y_true"])
        proba = {a: float(gr[f"proba_{a}"]) for a in range(5)}

        color = ACTION_COLORS[gbm_action]
        st.markdown(
            f"<div style='background:{color};padding:12px;border-radius:8px;"
            f"color:white;font-size:1.2em;font-weight:bold;text-align:center'>"
            f"추천: {ACTION_LABELS[gbm_action]}</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(_proba_bar_fig(proba, gbm_action, "예측 확률"), use_container_width=True)
        st.caption(f"신뢰도: {proba[gbm_action]:.1%} | ECE 0.030 (잘 보정됨)")

# --- dBCQ card ---
with col_dbcq:
    st.markdown("### 🤖 dBCQ (Offline RL)")
    if dbcq_row.empty:
        st.info("이 시점 dBCQ 예측 없음")
        dbcq_action = None
    else:
        dr = dbcq_row.iloc[0]
        dbcq_action = int(dr["policy_action"])
        q_vals = {a: float(dr[f"q_{a}"]) for a in range(5)}

        color = ACTION_COLORS[dbcq_action]
        st.markdown(
            f"<div style='background:{color};padding:12px;border-radius:8px;"
            f"color:white;font-size:1.2em;font-weight:bold;text-align:center'>"
            f"추천: {ACTION_LABELS[dbcq_action]}</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(_q_bar_fig(q_vals, dbcq_action, "Q-value (높을수록 선호)"), use_container_width=True)
        st.caption("⚠️ OPE ESS 4.5% — 통계적 신뢰도 제한적. 참고용.")

# ---------------------------------------------------------------------------
# Agreement + clinician actual
# ---------------------------------------------------------------------------
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    if gbm_action is not None and dbcq_action is not None:
        if gbm_action == dbcq_action:
            st.success(f"✅ **두 모델 일치** → {ACTION_LABELS[gbm_action]}")
        else:
            st.warning(
                f"⚡ **두 모델 불일치**\n"
                f"- GBM: {ACTION_LABELS[gbm_action]}\n"
                f"- dBCQ: {ACTION_LABELS[dbcq_action]}"
            )

with col2:
    if not gbm_row.empty:
        clin = int(gbm_row.iloc[0]["y_true"])
        st.info(f"👨‍⚕️ **실제 의사 결정**: {ACTION_LABELS[clin]}")

with col3:
    ne_now = float(state.get("ne_dose_now", 0) or 0)
    sofa_now = float(state.get("sofa", 0) or 0)
    st.metric("현재 NE 용량", f"{ne_now:.1f} mcg/min")
    st.metric("현재 SOFA", f"{sofa_now:.0f}")

# ---------------------------------------------------------------------------
# Current state snapshot
# ---------------------------------------------------------------------------
with st.expander("현재 상태 스냅샷 (bin 주요 변수)"):
    snap_cols = list(VITALS_DISPLAY.keys()) + list(LAB_DISPLAY.keys())
    snap_cols = [c for c in snap_cols if c in state.index]
    snap = pd.DataFrame({
        "변수": [VITALS_DISPLAY.get(c, LAB_DISPLAY.get(c, c)) for c in snap_cols],
        "값": [f"{state[c]:.2f}" if pd.notna(state[c]) else "N/A" for c in snap_cols],
    })
    st.dataframe(snap, use_container_width=True, hide_index=True)

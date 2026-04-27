"""OPE Report page — off-policy evaluation results for all RL policies."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="OPE Report", layout="wide")

st.title("📊 Off-Policy Evaluation (OPE) 리포트")
st.warning(
    "**정직성 가드**: 아래 수치는 오프라인 OPE 추정값입니다. "
    "실제 임상 효과(사망률 감소 등)의 증거가 아닙니다. "
    "Prospective RCT 미수행.",
    icon="⚠️",
)

RUNS_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "runs"

# ---------------------------------------------------------------------------
# Load metrics
# ---------------------------------------------------------------------------

@st.cache_data
def _load_metrics(run_name: str) -> dict:
    path = RUNS_DIR / run_name / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


RUNS = {
    "BC (mlp)": "2026-04-27_rl_bc_mlp_42",
    "dBCQ (mlp)": "2026-04-27_rl_dbcq_mlp_42",
    "CQL (mlp)": "2026-04-27_rl_cql_mlp_42",
}

metrics = {name: _load_metrics(run) for name, run in RUNS.items()}

# ---------------------------------------------------------------------------
# Supervised model comparison table
# ---------------------------------------------------------------------------
st.subheader("Phase A — 지도학습 모델 비교 (test fold)")

sup_data = {
    "모델": ["LR", "GBM ⭐", "MLP", "TCN†"],
    "macro-F1": [0.455, 0.529, 0.504, 0.433],
    "top-2 acc": [0.814, 0.875, 0.858, 0.893],
    "ECE": [0.049, 0.030, 0.088, "—"],
    "비고": [
        "베이스라인",
        "종합 우승 (Optuna 50 trials)",
        "focal loss → calibration 부족",
        "†stay당 1 sample (다른 태스크)",
    ],
}
st.dataframe(pd.DataFrame(sup_data), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# OPE table
# ---------------------------------------------------------------------------
st.subheader("Phase B — Offline RL OPE 비교 (val fold)")

def _extract_row(name: str, m: dict, fold: str = "val") -> dict | None:
    learned = m.get(fold, {}).get("learned", {})
    if not learned:
        return None
    return {
        "정책": name,
        "WIS": f"{learned['wis']:.3f}",
        "WIS 95% CI": f"[{learned['wis_ci_lo']:.3f}, {learned['wis_ci_hi']:.3f}]",
        "ESS %": f"{learned['ess_frac'] * 100:.1f}",
        "FQE": f"{learned['fqe']:.3f}",
        "FQE 95% CI": f"[{learned['fqe_ci_lo']:.3f}, {learned['fqe_ci_hi']:.3f}]",
        "Match %": f"{learned['match_rate'] * 100:.1f}",
        "OOD %": f"{learned['ood_rate'] * 100:.1f}",
        "경고": ", ".join(learned.get("warnings", [])) or "없음",
    }

rows = []
for name, m in metrics.items():
    r = _extract_row(name, m, "val")
    if r:
        rows.append(r)

# Add clinician row from BC metrics
bc_m = metrics.get("BC (mlp)", {})
clin = bc_m.get("val", {}).get("clinician", {})
if clin:
    rows.append({
        "정책": "Clinician (기준)",
        "WIS": f"{clin['wis']:.3f}",
        "WIS 95% CI": f"[{clin['wis_ci_lo']:.3f}, {clin['wis_ci_hi']:.3f}]",
        "ESS %": f"{clin['ess_frac'] * 100:.1f}",
        "FQE": f"{clin['fqe']:.3f}",
        "FQE 95% CI": f"[{clin['fqe_ci_lo']:.3f}, {clin['fqe_ci_hi']:.3f}]",
        "Match %": "100.0",
        "OOD %": f"{clin['ood_rate'] * 100:.1f}",
        "경고": "없음",
    })

if rows:
    ope_df = pd.DataFrame(rows)
    st.dataframe(ope_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# WIS bar chart with CI
# ---------------------------------------------------------------------------
st.subheader("WIS 비교 (오차 막대 포함)")

wis_policies, wis_vals, wis_lo, wis_hi, ess_flags = [], [], [], [], []
for name, m in metrics.items():
    learned = m.get("val", {}).get("learned", {})
    if not learned:
        continue
    wis_policies.append(name)
    wis_vals.append(learned["wis"])
    wis_lo.append(learned["wis"] - learned["wis_ci_lo"])
    wis_hi.append(learned["wis_ci_hi"] - learned["wis"])
    ess_flags.append(learned["ess_frac"] < 0.05)

if clin:
    wis_policies.append("Clinician")
    wis_vals.append(clin["wis"])
    wis_lo.append(clin["wis"] - clin["wis_ci_lo"])
    wis_hi.append(clin["wis_ci_hi"] - clin["wis"])
    ess_flags.append(False)

colors = ["#ff7f0e" if flag else "#1f77b4" for flag in ess_flags]
colors[-1] = "#2ca02c"  # clinician = green

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=wis_policies,
        y=wis_vals,
        error_y=dict(
            type="data",
            symmetric=False,
            array=wis_hi,
            arrayminus=wis_lo,
        ),
        marker_color=colors,
        text=[f"{v:.3f}" for v in wis_vals],
        textposition="outside",
    )
)
fig.update_layout(
    title="WIS (높을수록 좋음) — 주황=ESS<5% 경고",
    yaxis_title="WIS 값",
    height=380,
    margin=dict(t=50, b=20),
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "WIS는 BC(ESS=100%)만 신뢰도 높음. dBCQ/CQL은 ESS<5%로 CI 넓음 → "
    "FQE 기준 해석 권장."
)

# ---------------------------------------------------------------------------
# Null policy comparison
# ---------------------------------------------------------------------------
st.subheader("Null baseline 비교 (val fold, BC run)")

null_data = bc_m.get("val", {}).get("nulls", {})
if null_data:
    null_rows = []
    for pname, nd in null_data.items():
        null_rows.append({
            "Null 정책": pname,
            "WIS": f"{nd['wis']:.3f}",
            "WIS CI": f"[{nd['wis_ci_lo']:.3f}, {nd['wis_ci_hi']:.3f}]",
            "ESS %": f"{nd['ess_frac'] * 100:.1f}",
            "FQE": f"{nd['fqe']:.3f}",
            "OOD %": f"{nd['ood_rate'] * 100:.1f}",
        })
    st.dataframe(pd.DataFrame(null_rows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------
st.subheader("해석 가이드")
st.markdown("""
| 지표 | 의미 | 주의 |
|---|---|---|
| **WIS** | trajectory IS 가중 수익. 값이 클수록 정책이 더 나은 궤적 선택 | ESS<5%이면 신뢰 불가 |
| **ESS %** | 유효 표본 비율. 낮을수록 IS 가중치 분산 폭발 | <5% = 경고 |
| **FQE** | 별도 Q-critic으로 추정한 정책 가치. IS 문제 없음 | CI 넓으면 불확실 |
| **Match %** | 의사 결정과의 일치율 | 높다고 좋은 것이 아님 |
| **OOD %** | 의사가 한 번도 안 한 액션 추천 비율 | >10%이면 경고 |

**결론**:
- WIS 신뢰도: **BC > Clinician >> dBCQ ≈ CQL** (ESS 문제)
- FQE 정책 가치: **dBCQ (0.145) > CQL (0.127) > Clinician (0.055) > BC (0.017)**
- 단, 모든 FQE CI가 0을 포함 → 통계적 유의성 없음
- 안전 권고: **BC를 안전 베이스라인, dBCQ를 연구 후보**로 사용
""")

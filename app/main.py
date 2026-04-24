"""Streamlit entry point for the vasopressor recommendation demo.

Run with:
    streamlit run app/main.py
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="CDSS — Vasopressor Recommendation", layout="wide")
st.title("CDSS — Vasopressor Recommendation (MIMIC-IV)")
st.caption("Practice prototype. Not for clinical use.")

st.info("환경 셋업 완료. 모델/데이터 파이프라인 연결 후 본격 데모가 채워집니다.")

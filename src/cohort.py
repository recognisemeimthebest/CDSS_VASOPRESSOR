"""Cohort definition for vasopressor recommendation task.

Target cohort (initial draft, refine in EDA):
- Adult ICU stays from MIMIC-IV
- Patients meeting sepsis-3 criteria (will use mimiciv_derived.sepsis3 if available)
- Stays where vasopressor was administered at any point

Refine in notebooks/01_cohort_eda.ipynb.
"""
from __future__ import annotations

# Placeholder — to be implemented after DB connection is verified.

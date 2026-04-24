"""Time-series feature engineering for vasopressor recommendation.

Planned features (per ICU stay, hourly bins):
- Vitals: HR, SBP, DBP, MAP, SpO2, RR, temperature
- Labs: lactate, creatinine, BUN, WBC, platelets, INR, bilirubin, pH, PaO2, PaCO2
- Fluids: cumulative IV in/out, urine output
- Ventilation: PEEP, FiO2, vent mode flag
- Demographics/static: age, sex, weight, admission type, comorbidities
- Severity: SOFA, SAPS-II (from mimiciv_derived if available)
"""
from __future__ import annotations

# Placeholder — to be implemented after cohort is defined.

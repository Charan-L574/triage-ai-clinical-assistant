"""
frontend/test_cases.py — Extracted Pre-loaded Test Cases viewer.
"""
from __future__ import annotations
import sys, os
import json
from pathlib import Path
import pandas as pd
import streamlit as st

def render_test_cases_ui():
    st.markdown("### 📋 Pre-loaded Test Cases")
    bench_file = Path(__file__).parent.parent / "tests" / "benchmark_cases.json"
    cases = []
    if bench_file.exists():
        try:
            cases = json.loads(bench_file.read_text(encoding="utf-8"))
        except Exception as e:
            st.error(f"Could not load cases: {e}")

    if cases:
        preview_rows = [{
            "Case ID": c["case_id"],
            "Chief Complaint": c["description"],
            "Correct ESI": c["correct_esi_level"],
            "Clinical Notes": c.get("clinical_notes", "")[:80] + "...",
        } for c in cases]
        st.dataframe(pd.DataFrame(preview_rows), use_container_width=True)
    else:
        st.warning("No benchmark cases found.")

"""Streamlit review dashboard. Upload both tables, run the engine, review results.

Run:  streamlit run app.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from report_cleanup.config import load_config
from report_cleanup.export_excel import report_row
from report_cleanup.pipeline import run_pipeline
from report_cleanup.security import mask_df, sensitive_mode_enabled

st.set_page_config(page_title="Report Decommissioning", layout="wide")
st.title("Report Decommissioning & Consolidation")

with st.sidebar:
    st.header("Inputs")
    f1 = st.file_uploader("Table 1 — Metadata", type=["csv", "xlsx", "xls"])
    f2 = st.file_uploader("Table 2 — Execution", type=["csv", "xlsx", "xls"])
    f3 = st.file_uploader("Table 3 — Field Export", type=["csv", "xlsx", "xls"])
    run_btn = st.button("Run analysis", type="primary", disabled=f1 is None)


def _save_tmp(uploaded) -> str:
    suffix = Path(uploaded.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.close()
    return tmp.name


if run_btn and f1 is not None:
    p1 = _save_tmp(f1)
    p2 = _save_tmp(f2) if f2 is not None else None
    p3 = _save_tmp(f3) if f3 is not None else None
    try:
        with st.spinner("Analyzing report catalog..."):
            res = run_pipeline(p1, p2, p3, out_dir=tempfile.mkdtemp())
        st.session_state["res"] = res
    finally:
        # [SECURITY] Always delete temp upload copies, even on failure, so raw
        # HR exports are not left behind in the OS temp dir.
        for p in filter(None, [p1, p2, p3]):
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass

res = st.session_state.get("res")
if not res:
    st.info("Upload Table 1 (and optionally Table 2), then click **Run analysis**.")
    st.stop()

records = res["records"]
groups = res["groups"]
df = pd.DataFrame([report_row(r) for r in records])

# KPI row.
hard = sum(1 for r in records if r.get("is_hard_rule"))
band = lambda b: sum(1 for r in records if not r.get("is_hard_rule") and r.get("recommendation") == b)
k = st.columns(6)
k[0].metric("Total", len(records))
k[1].metric("Hard rule (100)", hard)
k[2].metric("High-Priority Review", band("High-Priority Decommissioning Review"))
k[3].metric("Decommissioning Review", band("Decommissioning Review"))
k[4].metric("Owner Review / Monitor", band("Owner Review / Monitor"))
k[5].metric("Dup groups", len(groups))

st.bar_chart(df["recommendation"].value_counts())

# Filters.
with st.sidebar:
    st.header("Filters")
    recs = sorted(df["recommendation"].dropna().unique())
    fr = st.multiselect("Recommendation", recs)
    owners = sorted([o for o in df["Owner"].dropna().unique() if o])
    fo = st.multiselect("Owner", owners)
    sources = sorted([s for s in df["Data Source"].dropna().unique() if s])
    fs = st.multiselect("Data source", sources)
    reason_q = st.text_input("Reason contains")
    with open(res["xlsx"], "rb") as fh:
        st.download_button("Download Excel workbook", fh.read(),
                           file_name=Path(res["xlsx"]).name)

fdf = df.copy()
if fr:
    fdf = fdf[fdf["recommendation"].isin(fr)]
if fo:
    fdf = fdf[fdf["Owner"].isin(fo)]
if fs:
    fdf = fdf[fdf["Data Source"].isin(fs)]
if reason_q:
    fdf = fdf[fdf["Reason Trail"].astype(str).str.contains(reason_q, case=False, na=False)]

t1, t2, t3, t4 = st.tabs(["Reports", "Duplicate groups", "Reason drill-down", "Data quality"])

with t1:
    st.caption(f"{len(fdf)} of {len(df)} reports")
    # [SECURITY] Mask person-identifying columns in the preview when sensitive
    # mode is on. Masking is display-only — filters above use the raw frame.
    show = mask_df(fdf) if sensitive_mode_enabled() else fdf

    NAME_COL = "Report Name"
    SCORE_COL = "Overall Score"
    SUGG_COL = "recommendation"   # the Overall-Score recommendation label
    TRAIL_COL = "Reason Trail"
    detail_cols = [c for c in show.columns if c not in (NAME_COL, TRAIL_COL)]

    # Table shows report name, overall score, the recommendation, and the reason
    # trail; all other fields live behind the dropdown so the page isn't a column dump.
    summary_cols = [c for c in (NAME_COL, SCORE_COL, SUGG_COL, TRAIL_COL) if c in show.columns]
    summary = show[summary_cols].rename(columns={SUGG_COL: "Recommendation"})
    st.dataframe(summary, use_container_width=True, hide_index=True)

    if show.empty:
        st.info("No reports match the current filters.")
    else:
        # Dropdown into the different fields for a chosen report. Indexed by
        # position so duplicate report names stay distinct.
        positions = list(range(len(show)))
        pick = st.selectbox(
            "View fields for report",
            positions,
            format_func=lambda i: str(show.iloc[i].get(NAME_COL) or "(unnamed report)"),
        )
        row = show.iloc[pick]
        detail = pd.DataFrame(
            [(c, "" if pd.isna(row[c]) else row[c]) for c in detail_cols],
            columns=["Field", "Value"],
        )
        st.dataframe(detail, use_container_width=True, hide_index=True)

with t2:
    by_uid = {r["report_uid"]: r for r in records}
    if not groups:
        st.write("No duplicate groups detected.")
    for g in sorted(groups, key=lambda x: x.group_id):
        keeper = by_uid[g.keeper_uid]
        with st.expander(f"{g.group_id} — {len(g.members)} reports — keeper: "
                         f"{keeper.get('report_name')}"):
            rows = []
            for u in g.members:
                r = by_uid[u]
                rows.append({
                    "Keeper?": "KEEPER" if u == g.keeper_uid else "",
                    "Report Name": r.get("report_name"),
                    "Field Count": r.get("field_count"),
                    "Field Similarity %": r.get("field_similarity_percent"),
                    "Field Containment %": r.get("field_containment_percent"),
                    "Classification": r.get("duplicate_classification"),
                    "Suggested Action": r.get("suggested_action"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with t3:
    names = {f"{r.get('report_name')}  (#{r['report_uid']})": r for r in records}
    pick = st.selectbox("Report", list(names.keys()))
    r = names[pick]
    rows = [{"Category": rs.category, "Reason": rs.label,
             "Points": "" if rs.points is None else rs.points}
            for rs in r.get("all_reasons", [])]
    rec = r.get("recurrence") or {}
    st.write(f"**Overall Score:** {r.get('overall_score')}"
             f"  |  **Recommendation:** {r.get('recommendation')}"
             f"  |  **Suggested action:** {r.get('suggested_action')}")
    st.write(f"**Hard rule:** {r.get('hard_rule_name') or 'No'}"
             f"  |  **Effective last run:** {r.get('effective_last_run_date')}"
             f"  |  **Cleanup:** {r.get('cleanup_risk_points')}/{r.get('cleanup_risk_max')}"
             f"  |  **Protection credit:** {r.get('business_protection_credit')}")
    st.write(f"**Recurrence:** {rec.get('classification')} ({rec.get('cadence')})"
             f"  |  **Potential duplicate of:** {r.get('potential_duplicate_of') or '—'}"
             f"  |  **Duplicate similarity:** {r.get('duplicate_similarity') or '—'}"
             f"  ({r.get('duplicate_relationship')})")
    if r.get("data_quality_flags"):
        st.write("**Data-quality flags:** " + "; ".join(r["data_quality_flags"]))
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with t4:
    st.write("**Join diagnostics**", res["diag"])
    for w in res["warnings"]:
        st.warning(w)

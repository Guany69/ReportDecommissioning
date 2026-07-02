"""Streamlit review dashboard. Upload both tables, run the engine, review results.

Run:  streamlit run app.py
"""
from __future__ import annotations

import hmac
import logging
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from report_cleanup.config import load_config
from report_cleanup.export_excel import duplicate_identification, report_row
from report_cleanup.pipeline import run_pipeline
from report_cleanup.security import (UnsafeUpload, auth_required, mask_df,
                                     sanitize_df_for_excel, sensitive_mode_enabled)

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Report Decommissioning", layout="wide")


def _enforce_auth() -> None:
    """[SECURITY] Optional access gate (finding: no app-level authentication).

    Disabled by default (REPORT_CLEANUP_REQUIRE_AUTH unset) so trusted local use
    is unchanged. When enabled, no upload/report/download UI renders until the
    visitor authenticates:

      * If Streamlit native OIDC is configured (`[auth]` in secrets.toml), require
        `st.login()` — the production-grade path (e.g. Microsoft Entra ID).
      * Otherwise fall back to a shared access code from REPORT_CLEANUP_ACCESS_CODE,
        compared in constant time. This is a minimal gate, not a substitute for
        per-user SSO; configure OIDC for real deployments.
    """
    if not auth_required():
        return

    # Prefer native OIDC when the deployment configured it.
    user = getattr(st, "user", None)
    if user is not None and "is_logged_in" in user:
        if not user.is_logged_in:
            st.title("Report Decommissioning & Consolidation")
            st.info("Authentication is required to use this application.")
            st.button("Log in", on_click=st.login, type="primary")
            st.stop()
        return

    # Fallback: shared access code.
    code = os.environ.get("REPORT_CLEANUP_ACCESS_CODE", "")
    if not code:
        st.error(
            "Access is restricted but no authentication method is configured. "
            "Set up Streamlit OIDC (`[auth]` in secrets.toml) or set "
            "REPORT_CLEANUP_ACCESS_CODE."
        )
        st.stop()
    if st.session_state.get("_authed"):
        return
    st.title("Report Decommissioning & Consolidation")
    entered = st.text_input("Access code", type="password")
    if not entered:
        st.stop()
    if not hmac.compare_digest(entered, code):
        st.error("Incorrect access code.")
        st.stop()
    st.session_state["_authed"] = True


_enforce_auth()

st.title("Report Decommissioning & Consolidation")

with st.sidebar:
    st.header("Inputs")
    f1 = st.file_uploader("Table 1 — Metadata", type=["csv", "xlsx", "xls"])
    f2 = st.file_uploader("Table 2 — Execution", type=["csv", "xlsx", "xls"])
    f3 = st.file_uploader("Table 3 — Field Export", type=["csv", "xlsx", "xls"])
    run_btn = st.button("Run analysis", type="primary", disabled=f1 is None)


def _save_tmp(uploaded) -> str:
    """Stream the upload to a temp file in bounded chunks.

    [SECURITY] Avoid materializing the whole upload in memory at once
    (uploaded.getbuffer()); copy in fixed-size chunks so an oversized file does
    not spike RAM before the per-file size/signature checks in read_any run.
    """
    suffix = Path(uploaded.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        uploaded.seek(0)
        shutil.copyfileobj(uploaded, tmp, length=1024 * 1024)
    finally:
        tmp.close()
    return tmp.name


def _cleanup_dir(path: str | None) -> None:
    """[SECURITY] Remove a generated output dir (Excel + SQLite) and surface,
    not swallow, any failure so an operator knows sensitive artifacts may remain."""
    if not path:
        return
    try:
        shutil.rmtree(path, ignore_errors=False)
    except OSError as exc:
        logger.warning("Could not remove output directory %s: %s", path, exc)
        st.warning(f"Could not remove a temporary output directory ({path}). "
                   "Sensitive files may remain on disk; remove them manually.")


if run_btn and f1 is not None:
    p1 = _save_tmp(f1)
    p2 = _save_tmp(f2) if f2 is not None else None
    p3 = _save_tmp(f3) if f3 is not None else None
    # [SECURITY] Delete the *previous* run's output dir (Excel + SQLite) before
    # starting a new one, so generated artifacts don't accumulate on disk.
    _cleanup_dir(st.session_state.pop("out_dir", None))
    out_dir = tempfile.mkdtemp()
    try:
        bar = st.progress(0.0, text="Starting analysis…")

        def _on_progress(fraction: float, label: str) -> None:
            bar.progress(min(fraction, 1.0), text=f"{label}…")

        res = run_pipeline(p1, p2, p3, out_dir=out_dir, progress=_on_progress)
        bar.progress(1.0, text="Analysis complete")
        bar.empty()
        st.session_state["res"] = res
        st.session_state["out_dir"] = out_dir
    except UnsafeUpload as exc:
        # Input-validation messages are safe to show (no internals/paths).
        _cleanup_dir(out_dir)
        st.error(f"Upload rejected: {exc}")
        st.stop()
    except Exception:
        # [SECURITY] Never leak stack traces, internal module names, or local
        # paths into the UI. Log the detail server-side; show a generic message.
        logger.exception("Pipeline failed")
        _cleanup_dir(out_dir)
        st.error("Analysis failed while processing the uploaded files. "
                 "Check that the inputs are valid Workday exports and try again.")
        st.stop()
    finally:
        # [SECURITY] Always delete temp upload copies, even on failure, so raw
        # HR exports are not left behind in the OS temp dir.
        for p in filter(None, [p1, p2, p3]):
            try:
                Path(p).unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Could not delete temp upload %s: %s", p, exc)
                st.warning(f"Could not delete a temporary upload file ({p}). "
                           "Remove it manually if it contains sensitive data.")

res = st.session_state.get("res")
if not res:
    st.info("Upload Table 1 (and optionally Table 2), then click **Run analysis**.")
    st.stop()

records = res["records"]
groups = res["groups"]
df = pd.DataFrame([report_row(r) for r in records])

# Duplicate identification (status + public DG-id) and grouped ordering — same source
# of truth as the decommission summary export, so the on-screen table, its CSV
# download, and the downloadable summary workbook all show identical values/order.
DUP_STATUS_COL = "Duplicate Status"
DUP_GID_COL = "Duplicate Group ID"
_ident = duplicate_identification(records)
df[DUP_STATUS_COL] = [_ident[r["report_uid"]][0] for r in records]
df[DUP_GID_COL] = [_ident[r["report_uid"]][1] for r in records]
# Drop the redundant internal group id (DUP-0001) display in favor of the public
# DG-001 column, so only one group-id column is shown.
df = df.drop(columns=["duplicate_group_id"], errors="ignore")
# Grouped order: duplicate groups first (as contiguous blocks), then non-duplicates.
_rank = {r["report_uid"]: i
         for i, r in enumerate(sorted(records, key=lambda r: _ident[r["report_uid"]][2]))}
df["_rank"] = [_rank[r["report_uid"]] for r in records]
df = df.sort_values("_rank").drop(columns="_rank").reset_index(drop=True)

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
    # [SECURITY] In sensitive mode, the filter still needs distinct, selectable
    # owner values, but must not display real names. Show stable pseudonyms
    # ("Owner 1", "Owner 2", …) via format_func while filtering on the real value.
    if sensitive_mode_enabled():
        _owner_alias = {o: f"Owner {i}" for i, o in enumerate(owners, start=1)}
        fo = st.multiselect("Owner", owners,
                            format_func=lambda o: _owner_alias.get(o, "***"))
    else:
        fo = st.multiselect("Owner", owners)
    sources = sorted([s for s in df["Data Source"].dropna().unique() if s])
    fs = st.multiselect("Data source", sources)
    reason_q = st.text_input("Reason contains")
    with open(res["xlsx"], "rb") as fh:
        st.download_button("Download Excel workbook", fh.read(),
                           file_name=Path(res["xlsx"]).name)
    if res.get("summary_xlsx"):
        with open(res["summary_xlsx"], "rb") as fh:
            st.download_button("Download decommission summary", fh.read(),
                               file_name=Path(res["summary_xlsx"]).name)

fdf = df.copy()
if fr:
    fdf = fdf[fdf["recommendation"].isin(fr)]
if fo:
    fdf = fdf[fdf["Owner"].isin(fo)]
if fs:
    fdf = fdf[fdf["Data Source"].isin(fs)]
if reason_q:
    # [SECURITY] regex=False treats the query as a literal substring. Without it,
    # pandas compiles user input as a regex — a malformed pattern crashes the page
    # and a catastrophic-backtracking pattern burns CPU across every row (ReDoS).
    fdf = fdf[fdf["Reason Trail"].astype(str).str.contains(
        reason_q, case=False, na=False, regex=False)]

t1, t2, t3, t4 = st.tabs(["Reports", "Duplicate groups", "Reason drill-down", "Data quality"])

with t1:
    st.caption(f"{len(fdf)} of {len(df)} reports")
    # [SECURITY] Mask person-identifying columns in the preview when sensitive
    # mode is on. Masking is display-only — filters above use the raw frame.
    show = mask_df(fdf) if sensitive_mode_enabled() else fdf
    # [SECURITY] Neutralize Excel/Sheets formula injection in every text cell
    # before rendering. st.dataframe offers a CSV download and copy-to-clipboard,
    # so an attacker-controlled value like "=HYPERLINK(...)" must be defanged the
    # same way the generated workbook is (export_excel._prep).
    show = sanitize_df_for_excel(show)

    NAME_COL = "Report Name"
    SCORE_COL = "Overall Score"
    SUGG_COL = "recommendation"   # the Overall-Score recommendation label
    TRAIL_COL = "Reason Trail"
    detail_cols = [c for c in show.columns
                   if c not in (NAME_COL, TRAIL_COL, DUP_STATUS_COL, DUP_GID_COL)]

    # Table shows report name, overall score, the recommendation, the reason trail,
    # and the duplicate status + group id; all other fields live behind the dropdown
    # so the page isn't a column dump. Rows are already in grouped order, so the
    # table (and its CSV download) presents duplicate groups as contiguous blocks.
    summary_cols = [c for c in (NAME_COL, SCORE_COL, SUGG_COL, TRAIL_COL,
                                DUP_STATUS_COL, DUP_GID_COL) if c in show.columns]
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
            st.dataframe(sanitize_df_for_excel(pd.DataFrame(rows)),
                         use_container_width=True, hide_index=True)

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
    st.dataframe(sanitize_df_for_excel(pd.DataFrame(rows)),
                 use_container_width=True, hide_index=True)

with t4:
    st.write("**Join diagnostics**", res["diag"])
    for w in res["warnings"]:
        st.warning(w)

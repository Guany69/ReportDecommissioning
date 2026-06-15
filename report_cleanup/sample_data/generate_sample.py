"""Generate synthetic Workday-shaped sample exports using the real source headers.

Writes three CSVs next to this script:
  metadata.csv    — Comprehensive report (one row per report)
  execution.csv   — Runs report (one row per execution; ~6 month window)
  fields.csv      — Fields report (one row per unique field; Where_Used links reports)

Rows are built as dicts keyed by header so columns can never drift out of order.

Run:  python sample_data/generate_sample.py
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
TODAY = datetime(2026, 6, 15)


def _d(days_ago: int) -> str:
    return (TODAY - timedelta(days=days_ago)).strftime("%Y-%m-%d")


def _dt(days_ago: int, hour: int = 9, minute: int = 0) -> str:
    return (TODAY - timedelta(days=days_ago)).replace(hour=hour, minute=minute).strftime("%Y-%m-%d %H:%M:%S")


COMPREHENSIVE_HEADERS = [
    "Custom Report", "Report Tag(s)", "Category", "Worklet Landing Pages", "Owner",
    "Created by", "Description", "Created Date", "Available Usage",
    "Number of Times Executed", "Last Updated Date", "Last Run By", "Last Run Date",
    "Data Source", "Report Type", "Worklet", "Chart Type", "Shared", "Landing Page",
    "Report Prompts", "Area Where Used",
]


def _comp(name, *, tag="HR", category="Workers", worklet_landing="", owner="Alice Smith",
          created_by=None, desc="", created=1200, usage="All", times="0",
          updated=1200, last_run_by="Alice Smith", last_run=1200, source="All Workers",
          rtype="Advanced", worklet="No", chart="Matrix", shared="No", landing="",
          prompts="", areas=""):
    return {
        "Custom Report": name, "Report Tag(s)": tag, "Category": category,
        "Worklet Landing Pages": worklet_landing, "Owner": owner,
        "Created by": created_by or owner, "Description": desc, "Created Date": _d(created),
        "Available Usage": usage, "Number of Times Executed": times,
        "Last Updated Date": _d(updated), "Last Run By": last_run_by,
        "Last Run Date": _d(last_run), "Data Source": source, "Report Type": rtype,
        "Worklet": worklet, "Chart Type": chart, "Shared": shared, "Landing Page": landing,
        "Report Prompts": prompts, "Area Where Used": areas,
    }


COMPREHENSIVE_ROWS = [
    # Keeper of a duplicate group: recent, shared, broadly used, placed.
    _comp("Active Employee Report", worklet_landing="Worker Dashboard", desc="All active employees",
          created=900, times="240", updated=40, last_run_by="Bob Lee", last_run=20,
          shared="Yes", landing="Worker Dashboard", prompts="Location",
          areas="HR; Payroll; Compliance"),
    # Stale near-duplicate of the keeper (same fields via Where_Used).
    _comp("Active Employee Report Copy", owner="Carol White", desc="Copy of active employee report",
          created=1300, times="2", updated=1300, last_run_by="Carol White", last_run=1300,
          shared="No"),
    # Hard rule: DNU in name.
    _comp("Headcount Summary DNU", owner="Dan Brown", desc="Deprecated", created=1500,
          times="0", updated=1400, last_run_by="Dan Brown", last_run=1400),
    # Hard rule: orphaned worklet (worklet, no landing pages, no area where used).
    _comp("Onboarding Tile", owner="Eve Black", desc="Tile with no placement", created=800,
          times="5", updated=800, last_run_by="Eve Black", last_run=700,
          rtype="Tile", worklet="Yes", chart="Tile"),
    # High cleanup risk: very old, not shared, no placement.
    _comp("Old Compensation Report", tag="Comp", category="Compensation", owner="Frank Green",
          desc="Legacy comp report", created=1800, times="0", updated=1500,
          last_run_by="Frank Green", last_run=1500, source="Compensation", prompts="Salary"),
    # Moderate: quarterly finance report, some usage, shared.
    _comp("Quarterly Finance Report", tag="Finance", category="Financials", owner="Grace Hall",
          desc="Finance summary", created=700, times="30", updated=120,
          last_run_by="Grace Hall", last_run=95, source="Financials", shared="Yes",
          prompts="Period", areas="Finance"),
]


RUNS_HEADERS = [
    "Report Name", "Report Type", "Report Data Source Name", "Execution Mode",
    "Execution Type", "Start Date and Time", "Requested By",
    "CF_#_of_Direct_Reports_TF", "Direct Reports (Employees)", "Available Usage",
    "Area Where Used", "Brief Description", "Last Run By", "Last Run Date",
]


def _run(name, start, requested_by, *, mode="Scheduled", etype="Background",
         area="HR", source="All Workers", rtype="All Workers", last_run=None):
    return {
        "Report Name": name, "Report Type": rtype, "Report Data Source Name": source,
        "Execution Mode": mode, "Execution Type": etype, "Start Date and Time": start,
        "Requested By": requested_by, "CF_#_of_Direct_Reports_TF": "",
        "Direct Reports (Employees)": "", "Available Usage": "All", "Area Where Used": area,
        "Brief Description": "", "Last Run By": requested_by,
        "Last Run Date": last_run if last_run is not None else "",
    }


def _runs_rows():
    rows = []
    # Active Employee Report — monthly cadence over 6 months, 5 distinct requesters.
    requesters = ["Bob Lee", "Jane Doe", "Sam Roe", "Bob Lee", "Mia Fox", "Lee Kim"]
    for i, days in enumerate([20, 50, 80, 110, 140, 170]):
        rows.append(_run("Active Employee Report", _dt(days), requesters[i]))
    # Burst: a second run within 30 minutes of the most recent -> one occurrence.
    rows.append(_run("Active Employee Report", _dt(20, 9, 10), "Bob Lee",
                     mode="Manual", etype="Foreground"))
    # Quarterly Finance Report — two valid runs, two requesters.
    rows.append(_run("Quarterly Finance Report", _dt(95), "Grace Hall",
                     mode="Manual", etype="Foreground", area="Finance", source="Financials",
                     rtype="Financials"))
    rows.append(_run("Quarterly Finance Report", _dt(10), "Tom Ray",
                     mode="Manual", etype="Foreground", area="Finance", source="Financials",
                     rtype="Financials"))
    # Invalid timestamp row: excluded from recurrence, still counts as a run.
    rows.append(_run("Quarterly Finance Report", "not-a-date", "Tom Ray",
                     mode="Manual", etype="Foreground", area="Finance", source="Financials",
                     rtype="Financials"))
    return rows


FIELDS_HEADERS = [
    "Business Object", "Field Name", "Description", "Report Field Type",
    "Related Business Object Name", "Built-in Prompts", "Domain", "Categories",
    "Authorized Usage", "Where_Used",
]


def _field(bo, name, *, desc="", ftype="Text", related="", prompts="No", domain="Workers",
           categories="HR", auth="All", where_used=""):
    return {
        "Business Object": bo, "Field Name": name, "Description": desc,
        "Report Field Type": ftype, "Related Business Object Name": related,
        "Built-in Prompts": prompts, "Domain": domain, "Categories": categories,
        "Authorized Usage": auth, "Where_Used": where_used,
    }


# The keeper and its copy share field rows -> high Jaccard / containment.
_SHARED = "Active Employee Report\nActive Employee Report Copy"
FIELDS_ROWS = [
    _field("Worker", "Employee ID", desc="Worker id", where_used=_SHARED),
    _field("Worker", "Employee Name", desc="Worker name", where_used=_SHARED),
    _field("Worker", "Manager", ftype="Reference", related="Worker", where_used=_SHARED),
    _field("Worker", "Location", ftype="Reference", related="Location", prompts="Yes",
           where_used="Active Employee Report"),
    _field("Worker", "Hire Date", ftype="Date", where_used="Active Employee Report"),
    _field("Financial", "Period", ftype="Text", prompts="Yes", domain="Finance",
           categories="Finance", where_used="Quarterly Finance Report"),
    _field("Financial", "Amount", ftype="Numeric", domain="Finance", categories="Finance",
           where_used="Quarterly Finance Report"),
]


def _write(path: Path, headers, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)
    return path


def main():
    _write(HERE / "metadata.csv", COMPREHENSIVE_HEADERS, COMPREHENSIVE_ROWS)
    _write(HERE / "execution.csv", RUNS_HEADERS, _runs_rows())
    _write(HERE / "fields.csv", FIELDS_HEADERS, FIELDS_ROWS)
    print(f"Wrote sample exports to {HERE}")


if __name__ == "__main__":
    main()

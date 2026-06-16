from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

MODULE_ROOT = Path(__file__).resolve().parents[1]


def _stringify_list_columns(dataframe: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = dataframe.copy()
    for column in columns:
        result[column] = result[column].apply(
            lambda value: " | ".join(value) if isinstance(value, list) else value
        )
    return result


def build_summary(
    clustered_students: pd.DataFrame,
    seating_plan: pd.DataFrame,
    room_summary: pd.DataFrame,
    faculty_allocation: pd.DataFrame,
) -> str:
    total_students = len(clustered_students)
    total_rooms_used = room_summary["room_id"].nunique()
    total_exams = clustered_students["exam"].nunique()
    total_faculty_assigned = faculty_allocation[faculty_allocation["faculty_id"] != "UNASSIGNED"]["faculty_id"].nunique()

    return (
        "Automated Exam Management Summary\n"
        f"Total students: {total_students}\n"
        f"Total exams: {total_exams}\n"
        f"Rooms used: {total_rooms_used}\n"
        f"Seats allocated: {len(seating_plan)}\n"
        f"Faculty members assigned: {total_faculty_assigned}\n"
    )


def export_reports(
    clustered_students: pd.DataFrame,
    seating_plan: pd.DataFrame,
    room_summary: pd.DataFrame,
    faculty_allocation: pd.DataFrame,
    output_dir: str,
) -> dict[str, str]:
    resolved_output_dir = Path(output_dir).expanduser()
    if not resolved_output_dir.is_absolute():
        resolved_output_dir = MODULE_ROOT / resolved_output_dir

    os.makedirs(resolved_output_dir, exist_ok=True)

    room_summary_export = _stringify_list_columns(room_summary, ["exams", "domains", "clusters"])

    files = {
        "clustered_students": os.path.join(resolved_output_dir, "clustered_students.csv"),
        "seating_plan": os.path.join(resolved_output_dir, "seating_plan.csv"),
        "faculty_allocation": os.path.join(resolved_output_dir, "faculty_allocation.csv"),
        "room_summary": os.path.join(resolved_output_dir, "room_summary.csv"),
        "summary_report": os.path.join(resolved_output_dir, "summary_report.txt"),
    }

    clustered_students.to_csv(files["clustered_students"], index=False)
    seating_plan.to_csv(files["seating_plan"], index=False)
    faculty_allocation.to_csv(files["faculty_allocation"], index=False)
    room_summary_export.to_csv(files["room_summary"], index=False)

    summary = build_summary(clustered_students, seating_plan, room_summary, faculty_allocation)
    with open(files["summary_report"], "w", encoding="utf-8") as report_file:
        report_file.write(summary)

    return files

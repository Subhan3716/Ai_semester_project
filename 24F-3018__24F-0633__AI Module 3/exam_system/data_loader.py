from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_STUDENT_COLUMNS = {"student_id", "name", "batch", "domain", "exam"}
REQUIRED_ROOM_COLUMNS = {"room_id", "capacity"}
REQUIRED_FACULTY_COLUMNS = {"faculty_id", "name", "domain", "available"}
MODULE_ROOT = Path(__file__).resolve().parents[1]


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _resolve_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return MODULE_ROOT / candidate


def _validate_columns(dataframe: pd.DataFrame, required_columns: set[str], label: str) -> None:
    missing = required_columns - set(dataframe.columns)
    if missing:
        raise ValueError(f"{label} file is missing required columns: {sorted(missing)}")


def load_students(path: str) -> pd.DataFrame:
    students = pd.read_csv(_resolve_path(path))
    _validate_columns(students, REQUIRED_STUDENT_COLUMNS, "Students")

    students = students.copy()
    for column in ["student_id", "name", "batch", "domain", "exam"]:
        students[column] = students[column].fillna("").astype(str).str.strip()

    if "special_requirements" not in students.columns:
        students["special_requirements"] = ""
    else:
        students["special_requirements"] = students["special_requirements"].fillna("").astype(str).str.strip()

    students = students[
        (students["student_id"] != "")
        & (students["name"] != "")
        & (students["batch"] != "")
        & (students["domain"] != "")
        & (students["exam"] != "")
    ]
    students = students.sort_values(["exam", "domain", "batch", "student_id"]).reset_index(drop=True)
    return students


def load_rooms(path: str) -> pd.DataFrame:
    rooms = pd.read_csv(_resolve_path(path))
    _validate_columns(rooms, REQUIRED_ROOM_COLUMNS, "Rooms")

    rooms = rooms.copy()
    rooms["room_id"] = _normalize_text(rooms["room_id"])
    rooms["capacity"] = pd.to_numeric(rooms["capacity"], errors="coerce")
    rooms = rooms.dropna(subset=["room_id", "capacity"])
    rooms["capacity"] = rooms["capacity"].astype(int)
    rooms = rooms[rooms["capacity"] > 0].reset_index(drop=True)
    rooms = rooms.sort_values(["capacity", "room_id"], ascending=[False, True]).reset_index(drop=True)
    return rooms


def load_faculty(path: str) -> pd.DataFrame:
    faculty = pd.read_csv(_resolve_path(path))
    _validate_columns(faculty, REQUIRED_FACULTY_COLUMNS, "Faculty")

    faculty = faculty.copy()
    for column in ["faculty_id", "name", "domain"]:
        faculty[column] = _normalize_text(faculty[column])

    faculty["available"] = (
        faculty["available"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y", "available"])
    )
    if "max_rooms" not in faculty.columns:
        faculty["max_rooms"] = 2
    else:
        faculty["max_rooms"] = pd.to_numeric(faculty["max_rooms"], errors="coerce").fillna(2).astype(int)

    faculty = faculty.reset_index(drop=True)
    return faculty


def load_all_data(students_path: str, rooms_path: str, faculty_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_students(students_path), load_rooms(rooms_path), load_faculty(faculty_path)

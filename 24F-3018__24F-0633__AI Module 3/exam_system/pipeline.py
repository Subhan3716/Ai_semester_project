from __future__ import annotations

from exam_system.clustering import cluster_students
from exam_system.data_loader import load_all_data
from exam_system.faculty import allocate_faculty
from exam_system.reporting import build_summary, export_reports
from exam_system.seating import create_seating_plan


def run_exam_management(students_path: str, rooms_path: str, faculty_path: str, output_dir: str = "output") -> dict[str, object]:
    students, rooms, faculty = load_all_data(students_path, rooms_path, faculty_path)
    clustered_students = cluster_students(students, rooms)
    seating_plan, room_summary = create_seating_plan(clustered_students, rooms)
    faculty_allocation = allocate_faculty(room_summary, faculty)
    files = export_reports(clustered_students, seating_plan, room_summary, faculty_allocation, output_dir)
    summary = build_summary(clustered_students, seating_plan, room_summary, faculty_allocation)

    return {
        "students": students,
        "rooms": rooms,
        "faculty": faculty,
        "clustered_students": clustered_students,
        "seating_plan": seating_plan,
        "room_summary": room_summary,
        "faculty_allocation": faculty_allocation,
        "files": files,
        "summary": summary,
    }


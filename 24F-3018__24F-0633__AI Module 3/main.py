from __future__ import annotations

import argparse

from exam_system.pipeline import run_exam_management


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Automated Exam Management System")
    parser.add_argument("--students", default="sample_data/students.csv", help="Path to students CSV")
    parser.add_argument("--rooms", default="sample_data/rooms.csv", help="Path to rooms CSV")
    parser.add_argument("--faculty", default="sample_data/faculty.csv", help="Path to faculty CSV")
    parser.add_argument("--output", default="output", help="Directory where reports will be saved")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_exam_management(
        students_path=args.students,
        rooms_path=args.rooms,
        faculty_path=args.faculty,
        output_dir=args.output,
    )
    print(result["summary"])
    print("\nGenerated files:")
    for name, path in result["files"].items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()


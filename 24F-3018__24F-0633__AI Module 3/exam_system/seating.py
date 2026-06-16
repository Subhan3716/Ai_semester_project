from __future__ import annotations

from collections import deque

import pandas as pd


def create_seating_plan(clustered_students: pd.DataFrame, rooms: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if clustered_students.empty:
        raise ValueError("Clustered student data is empty.")

    pending_groups = deque()
    for cluster_id, cluster_students in clustered_students.groupby("cluster_id", sort=True):
        group_data = cluster_students.copy()
        group_data["priority_flag"] = group_data["special_requirements"].astype(str).str.strip().ne("").astype(int)
        group_data = group_data.sort_values(
            ["priority_flag", "batch", "domain", "student_id"],
            ascending=[False, True, True, True],
        ).reset_index(drop=True)
        pending_groups.append({"cluster_id": cluster_id, "students": group_data})

    seating_rows: list[pd.DataFrame] = []

    for _, room in rooms.iterrows():
        room_id = room["room_id"]
        capacity = int(room["capacity"])
        remaining = capacity
        seat_number = 1

        while remaining > 0 and pending_groups:
            current_group = pending_groups[0]
            current_students = current_group["students"]
            take_count = min(remaining, len(current_students))

            assigned = current_students.iloc[:take_count].copy()
            assigned["room_id"] = room_id
            assigned["seat_number"] = list(range(seat_number, seat_number + take_count))
            seating_rows.append(assigned)

            remaining -= take_count
            seat_number += take_count
            leftover = current_students.iloc[take_count:].reset_index(drop=True)

            if leftover.empty:
                pending_groups.popleft()
            else:
                pending_groups[0]["students"] = leftover

    if pending_groups:
        remaining_students = sum(len(group["students"]) for group in pending_groups)
        raise ValueError(
            f"Not enough room capacity for all students. {remaining_students} students could not be seated."
        )

    seating_plan = pd.concat(seating_rows, ignore_index=True)
    seating_plan = seating_plan[
        [
            "room_id",
            "seat_number",
            "student_id",
            "name",
            "batch",
            "domain",
            "exam",
            "cluster_id",
            "special_requirements",
        ]
    ].sort_values(["room_id", "seat_number"]).reset_index(drop=True)

    room_summary = (
        seating_plan.groupby("room_id")
        .agg(
            occupied_seats=("student_id", "count"),
            exams=("exam", lambda values: sorted(set(values))),
            domains=("domain", lambda values: sorted(set(values))),
            clusters=("cluster_id", lambda values: sorted(set(values))),
        )
        .reset_index()
    )
    room_summary = room_summary.merge(rooms, on="room_id", how="left")
    room_summary = room_summary[["room_id", "capacity", "occupied_seats", "exams", "domains", "clusters"]]
    return seating_plan, room_summary

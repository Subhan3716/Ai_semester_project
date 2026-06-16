from __future__ import annotations

import pandas as pd


def allocate_faculty(room_summary: pd.DataFrame, faculty: pd.DataFrame) -> pd.DataFrame:
    available_faculty = faculty[faculty["available"]].copy().reset_index(drop=True)
    if available_faculty.empty:
        raise ValueError("No faculty members are available for allocation.")

    available_faculty["assigned_rooms"] = 0
    assignments: list[dict[str, object]] = []

    for _, room in room_summary.iterrows():
        domains = room["domains"]
        if isinstance(domains, str):
            domains = [value.strip() for value in domains.split("|") if value.strip()]

        for domain in domains:
            matching = available_faculty[available_faculty["domain"] == domain].copy()
            if matching.empty:
                assignments.append(
                    {
                        "room_id": room["room_id"],
                        "domain": domain,
                        "faculty_id": "UNASSIGNED",
                        "faculty_name": "No faculty available",
                        "status": "missing-domain-faculty",
                    }
                )
                continue

            chosen = matching.sort_values(["assigned_rooms", "faculty_id"]).iloc[0]
            over_limit = int(chosen["assigned_rooms"]) >= int(chosen["max_rooms"])
            chosen_index = chosen.name
            available_faculty.loc[chosen_index, "assigned_rooms"] += 1

            assignments.append(
                {
                    "room_id": room["room_id"],
                    "domain": domain,
                    "faculty_id": chosen["faculty_id"],
                    "faculty_name": chosen["name"],
                    "status": "overloaded" if over_limit else "assigned",
                }
            )

    faculty_allocation = pd.DataFrame(assignments)
    faculty_allocation = faculty_allocation.sort_values(["room_id", "domain"]).reset_index(drop=True)
    return faculty_allocation


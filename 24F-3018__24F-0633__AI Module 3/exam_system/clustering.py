from __future__ import annotations

import math

import pandas as pd
from sklearn.cluster import KMeans


def _estimate_clusters(student_count: int, average_room_capacity: float) -> int:
    if student_count <= 0:
        return 1
    safe_capacity = max(1, int(round(average_room_capacity)))
    return max(1, math.ceil(student_count / safe_capacity))


def _build_features(students: pd.DataFrame) -> pd.DataFrame:
    feature_columns = pd.get_dummies(students[["domain", "batch"]].astype(str), dtype=int)
    return feature_columns


def cluster_students(students: pd.DataFrame, rooms: pd.DataFrame) -> pd.DataFrame:
    if students.empty:
        raise ValueError("Student data is empty.")
    if rooms.empty:
        raise ValueError("Room data is empty.")

    average_room_capacity = rooms["capacity"].mean()
    clustered_frames: list[pd.DataFrame] = []

    for exam_name, exam_students in students.groupby("exam", sort=True):
        exam_students = exam_students.copy().reset_index(drop=True)
        cluster_count = min(len(exam_students), _estimate_clusters(len(exam_students), average_room_capacity))

        if cluster_count == 1:
            exam_students["cluster_number"] = 1
        else:
            features = _build_features(exam_students)
            model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
            exam_students["cluster_number"] = model.fit_predict(features) + 1

        exam_students["cluster_id"] = exam_students["cluster_number"].apply(lambda value: f"{exam_name}-C{value}")
        clustered_frames.append(exam_students)

    clustered_students = pd.concat(clustered_frames, ignore_index=True)
    clustered_students = clustered_students.sort_values(
        ["exam", "cluster_id", "domain", "batch", "student_id"]
    ).reset_index(drop=True)
    return clustered_students


# Automated Exam Management System

This is a very basic Python project for generating:

- student clusters using k-means
- an exam seating plan based on room capacity
- faculty allocation based on domain expertise
- CSV and text reports for administrators

## Features

- Reads student, room, and faculty data from CSV files
- Uses k-means clustering to group students of the same exam by batch and domain
- Fills rooms without exceeding capacity
- Allows multiple clustered groups to share the same room when seats remain
- Assigns faculty members according to the domains present in each room
- Exports final reports to the `output` folder

## Project Structure

- [main.py](./main.py)
- [gui.py](./gui.py)
- [exam_system/pipeline.py](./exam_system/pipeline.py)
- [exam_system/data_loader.py](./exam_system/data_loader.py)
- [exam_system/clustering.py](./exam_system/clustering.py)
- [exam_system/seating.py](./exam_system/seating.py)
- [exam_system/faculty.py](./exam_system/faculty.py)
- [exam_system/reporting.py](./exam_system/reporting.py)
- [sample_data/students.csv](./sample_data/students.csv)
- [sample_data/rooms.csv](./sample_data/rooms.csv)
- [sample_data/faculty.csv](./sample_data/faculty.csv)

## Input Files

### Students CSV

Required columns:

- `student_id`
- `name`
- `batch`
- `domain`
- `exam`

Optional column:

- `special_requirements`

### Rooms CSV

Required columns:

- `room_id`
- `capacity`

### Faculty CSV

Required columns:

- `faculty_id`
- `name`
- `domain`
- `available`

Optional column:

- `max_rooms`

## How the System Works

1. Student data is cleaned and standardized.
2. Students are separated by exam.
3. For each exam, k-means clustering groups students using `batch` and `domain`.
4. Clusters are placed into rooms without exceeding room capacity.
5. Faculty members are assigned to each room for the domains present in that room.
6. Reports are saved in CSV and text format.

## Assumptions

- One run represents one exam session timetable.
- A room can contain more than one cluster if seats are still available.
- Faculty are assigned for the domains that appear in a room.
- If there are not enough faculty members in one domain, the system still assigns the least-loaded available faculty member from that domain and marks the assignment status.
- The code supports any batch labels, although the sample data uses batches 19, 20, 21, and 22.

## Run From Command Line

```powershell
python main.py
```

If your files are stored somewhere else:

```powershell
python main.py --students path\to\students.csv --rooms path\to\rooms.csv --faculty path\to\faculty.csv --output output
```

## Run Basic GUI

```powershell
python gui.py
```

## Output Files

The program creates:

- `clustered_students.csv`
- `seating_plan.csv`
- `faculty_allocation.csv`
- `room_summary.csv`
- `summary_report.txt`

inside the selected output folder.

## Libraries Used

- `pandas`
- `numpy`
- `scikit-learn`

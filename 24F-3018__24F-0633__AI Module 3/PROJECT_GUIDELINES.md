# Completed Project Guidelines

## Problem Coverage

This project implements a basic automated exam management system for:

- Computer Science
- Artificial Intelligence
- Business Analytics
- Software Engineering
- Electrical Engineering

and supports multiple batches such as 19, 20, 21, and 22.

## Completed Design Guidelines

### 1. Data Collection

The system accepts CSV data for:

- students
- rooms
- faculty

This makes the project easy to expand for real university data.

### 2. Data Preprocessing

The preprocessing stage:

- removes blank spaces
- standardizes text values
- checks required columns
- converts room capacity to numeric values
- converts faculty availability to boolean values

### 3. K-Means Clustering

K-means is applied separately for each exam so that students are grouped in a practical way.

Clustering features:

- `domain`
- `batch`

The number of clusters is estimated from the exam size and average room capacity.

### 4. Seating Plan Generation

The seating module:

- keeps room capacity within limits
- fills rooms efficiently
- allows multiple clustered groups to share one room if seats remain
- gives each student a seat number

### 5. Faculty Allocation

The faculty module:

- checks which domains are present in each room
- allocates matching faculty members
- balances load using each faculty member's assigned room count
- supports a `max_rooms` limit

### 6. Reporting

The reporting module exports:

- clustered student list
- final seating plan
- faculty assignment list
- room summary
- text summary report

## Why K-Means Is Used Here

K-means helps group students with similar properties, especially:

- same domain
- same batch
- same exam session

This makes seating more systematic and easier to manage.

## Limitations Of This Basic Version

- It uses a simple room-filling strategy instead of advanced optimization.
- It assumes input data is already mostly correct.
- It does not yet support time-slot conflict detection.
- It does not yet generate charts or dashboards.

## Future Improvements

- add database support
- add login for administrators
- add exam timetable conflict checking
- add PDF report generation
- add web-based dashboard


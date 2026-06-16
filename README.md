# AI Semester Project

This repository contains the full semester project split into four modules. Each module solves a different AI task, but everything is kept in one repo so the submission is easy to review and run.

## Project Overview

- Module 1: intelligent urban delivery robot simulation
- Module 2: Sudoku solving with CSP techniques
- Module 3: automated exam management system
- Module 4: learning in AI using the Iris dataset

Each file includes the module number in its name so the purpose is clear at a glance.

## Modules

### Module 1: Intelligent Urban Delivery Robot

File:

- `24F0633_MODULE1_Intelligent_Urban_Delivery_Robot.py`

What it does:

- builds a 15x15 urban grid
- places roads, buildings, traffic, and delivery locations
- compares search algorithms such as BFS, DFS, UCS, Greedy, and A*
- measures performance and can animate the robot path

### Module 2: Sudoku Solver

File:

- `24F-0633_Module2.py`

What it does:

- solves 9x9 Sudoku puzzles
- uses CSP-based reasoning
- compares AC3 with backtracking
- includes a Tkinter GUI and self-test mode

### Module 3: Automated Exam Management System

Folder:

- `24F-3018__24F-0633__AI Module 3/`

What it does:

- loads students, rooms, and faculty data from CSV files
- clusters students by exam, batch, and domain
- creates a seating plan without exceeding room capacity
- allocates faculty according to domain availability
- exports CSV and text reports

Important files inside the module 3 folder:

- `main.py` for command-line execution
- `gui.py` for the Tkinter interface
- `exam_system/` for the core pipeline
- `sample_data/` for example input files
- `output/` for generated reports

### Module 4: Learning in AI

Files:

- `24F-0633_MODULE4_Learning_in_AI.py`
- `24F-0633_MODULE4_Learning_in_AI_Report.txt`

What it does:

- trains and compares a multiclass perceptron
- trains and compares a gradient descent delta rule model
- uses the Iris dataset
- prints experiment results and can save a report

## Folder Structure

```text
Ai_semester-project/
|-- 24F0633_MODULE1_Intelligent_Urban_Delivery_Robot.py
|-- 24F-0633_Module2.py
|-- 24F-0633_MODULE4_Learning_in_AI.py
|-- 24F-0633_MODULE4_Learning_in_AI_Report.txt
|-- 24F-3018__24F-0633__AI Module 3/
|   |-- main.py
|   |-- gui.py
|   |-- exam_system/
|   |-- sample_data/
|   |-- output/
|   |-- requirements.txt
|   |-- README.md
|   `-- PROJECT_GUIDELINES.md
`-- README.md
```

## Requirements

Recommended Python packages:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`

Notes:

- Module 1 uses `matplotlib` and `numpy`
- Module 2 uses `tkinter` and standard Python libraries
- Module 3 uses `pandas`, `numpy`, and `scikit-learn`
- Module 4 uses `numpy` and `scikit-learn`

## Setup

If you need to install dependencies for the full project:

```powershell
python -m pip install -r ".\24F-3018__24F-0633__AI Module 3\requirements.txt"
python -m pip install numpy scikit-learn matplotlib
```

## How To Run

Run commands from the repository root unless stated otherwise.

### Module 1

```powershell
python .\24F0633_MODULE1_Intelligent_Urban_Delivery_Robot.py
```

Optional:

```powershell
python .\24F0633_MODULE1_Intelligent_Urban_Delivery_Robot.py --no-visualization
```

### Module 2

```powershell
python .\24F-0633_Module2.py
```

Self-test:

```powershell
python .\24F-0633_Module2.py --self-test
```

GUI smoke test:

```powershell
python .\24F-0633_Module2.py --gui-smoke-test
```

### Module 3

Command line:

```powershell
python ".\24F-3018__24F-0633__AI Module 3\main.py"
```

GUI:

```powershell
python ".\24F-3018__24F-0633__AI Module 3\gui.py"
```

The module ships with sample CSV files in `sample_data/`, so it can run immediately.

### Module 4

```powershell
python .\24F-0633_MODULE4_Learning_in_AI.py
```

Self-test:

```powershell
python .\24F-0633_MODULE4_Learning_in_AI.py --self-test
```

Optional report file:

```powershell
python .\24F-0633_MODULE4_Learning_in_AI.py --report-file module4_report.txt
```

## Included Output

The repository also contains sample/generated outputs for review:

- module 3 CSV reports in `24F-3018__24F-0633__AI Module 3/output/`
- module 4 report text in `24F-0633_MODULE4_Learning_in_AI_Report.txt`

## Design Notes

- File names include the module number so the submission stays organized.
- Module 3 paths are written to work from the repository root.
- The project is intended as one final submission, not four separate repos.

## Quick Summary

- Module 1: pathfinding and simulation
- Module 2: constraint solving
- Module 3: data processing and exam planning
- Module 4: supervised learning comparison

If you want, I can also make a shorter submission-style README or add screenshots and a module-wise feature table.

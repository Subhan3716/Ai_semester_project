from __future__ import annotations

import argparse
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import Dict, List, Optional, Set, Tuple

DIGITS = "123456789"
ROWS = range(9)
COLS = range(9)
Cell = Tuple[int, int]
Board = List[List[int]]
DomainMap = Dict[Cell, Set[int]]


DATASET = {
    "easy": [
        "003020600900305001001806400008102900700000008006708200002609500800203009005010300",
        "200080300060070084030500209000105408000000000402706000301007040720040060004010003",
        "100007090030020008009600500005300900010080002600004000300000010040000007007000300",
        "030050040008010500460000012070502080000603000040109030250000098001020600080060020",
    ],
    "medium": [
        "000260701680070090190004500820100040004602900050003028009300074040050036703018000",
        "300000000005009000200504000020000700160000058704310600000890100000067080000005437",
        "530070000600195000098000060800060003400803001700020006060000280000419005000080079",
        "005300000800000020070010500400005300010070006003200080060500009004000030000009700",
    ],
    "hard": [
        "800000000003600000070090200050007000000045700000100030001000068008500010090000400",
        "100007090030020008009600500005300900010080002600004000300000010040000007007000300",
        "000900002050123400030000160908000000070000090000000205091000050007439020400007000",
        "600120384008459072000006005000264030070080006940003000310000050089700000502000190",
    ],
}


@dataclass
class SolveResult:
    solved: bool
    board: Optional[Board]
    elapsed_seconds: float
    assignments: int


class SudokuCSP:
    def __init__(self, board: Board):
        self.board = [row[:] for row in board]
        self.cells: List[Cell] = [(r, c) for r in ROWS for c in COLS]
        self.units = self._build_units()
        self.neighbors = self._build_neighbors()
        self.domains: DomainMap = self._build_domains()

    def _build_units(self) -> List[List[Cell]]:
        units: List[List[Cell]] = []
        for r in ROWS:
            units.append([(r, c) for c in COLS])
        for c in COLS:
            units.append([(r, c) for r in ROWS])
        for br in range(0, 9, 3):
            for bc in range(0, 9, 3):
                units.append([(r, c) for r in range(br, br + 3) for c in range(bc, bc + 3)])
        return units

    def _build_neighbors(self) -> Dict[Cell, Set[Cell]]:
        neighbors: Dict[Cell, Set[Cell]] = {cell: set() for cell in self.cells}
        for unit in self.units:
            for cell in unit:
                neighbors[cell].update(peer for peer in unit if peer != cell)
        return neighbors

    def _build_domains(self) -> DomainMap:
        domains: DomainMap = {}
        for r in ROWS:
            for c in COLS:
                value = self.board[r][c]
                if value == 0:
                    domains[(r, c)] = set(range(1, 10))
                else:
                    domains[(r, c)] = {value}
        return domains


def parse_grid(grid: str) -> Board:
    cleaned = [ch for ch in grid if ch in DIGITS or ch in ".0"]
    if len(cleaned) != 81:
        raise ValueError("Grid must contain 81 cells.")
    values = [0 if ch in ".0" else int(ch) for ch in cleaned]
    return [values[i : i + 9] for i in range(0, 81, 9)]


def board_to_string(board: Board) -> str:
    return "".join(str(n) for row in board for n in row)


def clone_board(board: Board) -> Board:
    return [row[:] for row in board]


def domains_to_board(domains: DomainMap) -> Optional[Board]:
    board = [[0 for _ in COLS] for _ in ROWS]
    for (r, c), domain in domains.items():
        if len(domain) != 1:
            return None
        board[r][c] = next(iter(domain))
    return board


def is_assignment_complete(domains: DomainMap) -> bool:
    return all(len(domain) == 1 for domain in domains.values())


def is_consistent_value(csp: SudokuCSP, domains: DomainMap, cell: Cell, value: int) -> bool:
    for peer in csp.neighbors[cell]:
        peer_domain = domains[peer]
        if len(peer_domain) == 1 and value in peer_domain:
            return False
    return True


def select_unassigned_variable(domains: DomainMap) -> Optional[Cell]:
    unresolved = [cell for cell, domain in domains.items() if len(domain) > 1]
    if not unresolved:
        return None
    return min(unresolved, key=lambda cell: len(domains[cell]))


def revise(domains: DomainMap, xi: Cell, xj: Cell) -> bool:
    revised = False
    to_remove: Set[int] = set()
    for x in domains[xi]:
        if all(x == y for y in domains[xj]):
            to_remove.add(x)
    if to_remove:
        domains[xi] -= to_remove
        revised = True
    return revised


def ac3(csp: SudokuCSP, domains: DomainMap) -> bool:
    queue = deque((xi, xj) for xi in csp.cells for xj in csp.neighbors[xi])
    while queue:
        xi, xj = queue.popleft()
        if revise(domains, xi, xj):
            if len(domains[xi]) == 0:
                return False
            for xk in csp.neighbors[xi] - {xj}:
                queue.append((xk, xi))
    return True


def backtracking_search(csp: SudokuCSP) -> SolveResult:
    start = time.perf_counter()
    assignments = 0

    def backtrack(domains: DomainMap) -> Optional[DomainMap]:
        nonlocal assignments
        if is_assignment_complete(domains):
            return domains

        var = select_unassigned_variable(domains)
        if var is None:
            return domains

        for value in sorted(domains[var]):
            if not is_consistent_value(csp, domains, var, value):
                continue

            assignments += 1
            next_domains = {cell: domain.copy() for cell, domain in domains.items()}
            next_domains[var] = {value}
            result = backtrack(next_domains)
            if result is not None:
                return result
        return None

    initial_domains = {cell: domain.copy() for cell, domain in csp.domains.items()}
    result_domains = backtrack(initial_domains)
    elapsed = time.perf_counter() - start
    solved_board = domains_to_board(result_domains) if result_domains else None
    return SolveResult(solved=solved_board is not None, board=solved_board, elapsed_seconds=elapsed, assignments=assignments)


def mac_search_with_ac3(csp: SudokuCSP) -> SolveResult:
    start = time.perf_counter()
    assignments = 0

    def backtrack(domains: DomainMap) -> Optional[DomainMap]:
        nonlocal assignments
        if is_assignment_complete(domains):
            return domains

        var = select_unassigned_variable(domains)
        if var is None:
            return domains

        for value in sorted(domains[var]):
            if not is_consistent_value(csp, domains, var, value):
                continue

            assignments += 1
            next_domains = {cell: domain.copy() for cell, domain in domains.items()}
            next_domains[var] = {value}
            if not ac3(csp, next_domains):
                continue
            result = backtrack(next_domains)
            if result is not None:
                return result
        return None

    initial_domains = {cell: domain.copy() for cell, domain in csp.domains.items()}
    if not ac3(csp, initial_domains):
        elapsed = time.perf_counter() - start
        return SolveResult(solved=False, board=None, elapsed_seconds=elapsed, assignments=assignments)

    result_domains = backtrack(initial_domains)
    elapsed = time.perf_counter() - start
    solved_board = domains_to_board(result_domains) if result_domains else None
    return SolveResult(solved=solved_board is not None, board=solved_board, elapsed_seconds=elapsed, assignments=assignments)


def solve_grid(grid: str, algorithm: str) -> SolveResult:
    board = parse_grid(grid)
    csp = SudokuCSP(board)
    if algorithm == "ac3":
        return mac_search_with_ac3(csp)
    if algorithm == "backtracking":
        return backtracking_search(csp)
    raise ValueError("Unsupported algorithm. Choose 'ac3' or 'backtracking'.")


def run_self_test() -> int:
    print("Running solver verification on full dataset...")
    for level, puzzles in DATASET.items():
        for idx, puzzle in enumerate(puzzles, start=1):
            ac3_result = solve_grid(puzzle, "ac3")
            bt_result = solve_grid(puzzle, "backtracking")
            if not ac3_result.solved or not bt_result.solved:
                print(f"[FAIL] {level} puzzle {idx}: one algorithm could not solve.")
                return 1
            if board_to_string(ac3_result.board) != board_to_string(bt_result.board):
                print(f"[FAIL] {level} puzzle {idx}: algorithm solutions differ.")
                return 1
            print(
                f"[OK] {level} puzzle {idx} | AC3: {ac3_result.elapsed_seconds:.5f}s, "
                f"BT: {bt_result.elapsed_seconds:.5f}s"
            )
    print("All puzzles solved by both algorithms.")
    return 0


class SudokuGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Sudoku - Module 2")
        self.root.resizable(False, False)

        self.level_var = tk.StringVar(value="easy")
        self.algorithm_var = tk.StringVar(value="ac3")
        self.puzzle_var = tk.IntVar(value=1)
        self.time_var = tk.StringVar(value="Time: -")

        self.entries: List[List[tk.Entry]] = []
        self.given_cells: Set[Cell] = set()
        self.initial_board: Board = [[0 for _ in COLS] for _ in ROWS]
        self.current_solution: Optional[Board] = None
        self.solution_cache: Dict[Tuple[str, int, str], Board] = {}

        self._build_layout()
        self.load_selected_puzzle()

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.grid(row=0, column=0, sticky="nsew")

        board_frame = ttk.Frame(container)
        board_frame.grid(row=0, column=0, rowspan=3, padx=(0, 20), sticky="nw")

        vcmd = (self.root.register(self._validate_entry), "%P")
        for r in ROWS:
            row_entries: List[tk.Entry] = []
            for c in COLS:
                pad_left = (3, 1) if c % 3 == 0 else (1, 1)
                pad_top = (3, 1) if r % 3 == 0 else (1, 1)
                pad_right = (3, 1) if c == 8 else (1, 1)
                pad_bottom = (3, 1) if r == 8 else (1, 1)

                entry = tk.Entry(
                    board_frame,
                    width=2,
                    font=("Times New Roman", 18, "bold"),
                    justify="center",
                    validate="key",
                    validatecommand=vcmd,
                    relief="solid",
                    bd=1,
                )
                entry.grid(row=r, column=c, padx=(pad_left[0], pad_right[0]), pady=(pad_top[0], pad_bottom[0]), ipadx=8, ipady=6)
                row_entries.append(entry)
            self.entries.append(row_entries)

        side_frame = ttk.Frame(container)
        side_frame.grid(row=0, column=1, sticky="n")

        ttk.Label(side_frame, text="Level").grid(row=0, column=0, sticky="w")
        level_combo = ttk.Combobox(
            side_frame,
            state="readonly",
            textvariable=self.level_var,
            values=["easy", "medium", "hard"],
            width=12,
        )
        level_combo.grid(row=1, column=0, sticky="w")
        level_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_level_changed())

        ttk.Separator(side_frame, orient="horizontal").grid(row=2, column=0, pady=10, sticky="ew")

        ttk.Button(side_frame, text="Reset", command=self.reset_board, width=20).grid(row=3, column=0, pady=(2, 8), sticky="w")
        ttk.Button(side_frame, text="Solve", command=self.solve_board, width=20).grid(row=4, column=0, pady=8, sticky="w")
        ttk.Button(side_frame, text="Hint", command=self.give_hint, width=20).grid(row=5, column=0, pady=8, sticky="w")

        ttk.Separator(side_frame, orient="horizontal").grid(row=6, column=0, pady=10, sticky="ew")

        ttk.Label(side_frame, text="Algorithms").grid(row=7, column=0, sticky="w")
        ttk.Radiobutton(side_frame, text="Arc Consistency-3", value="ac3", variable=self.algorithm_var).grid(row=8, column=0, sticky="w")
        ttk.Radiobutton(side_frame, text="Backtracking", value="backtracking", variable=self.algorithm_var).grid(row=9, column=0, sticky="w")

        ttk.Separator(side_frame, orient="horizontal").grid(row=10, column=0, pady=10, sticky="ew")

        ttk.Label(side_frame, text="Choose Puzzle").grid(row=11, column=0, sticky="w")
        for i in range(1, 5):
            ttk.Radiobutton(
                side_frame,
                text=f"Puzzle {i}",
                value=i,
                variable=self.puzzle_var,
                command=self.load_selected_puzzle,
            ).grid(row=11 + i, column=0, sticky="w")

        ttk.Label(container, textvariable=self.time_var, font=("Segoe UI", 10, "bold")).grid(row=3, column=0, columnspan=2, sticky="w", pady=(10, 0))

    def _validate_entry(self, value: str) -> bool:
        return value == "" or (len(value) == 1 and value in DIGITS)

    def _on_level_changed(self) -> None:
        self.puzzle_var.set(1)
        self.load_selected_puzzle()

    def _dataset_key(self) -> Tuple[str, int, str]:
        return (self.level_var.get(), self.puzzle_var.get(), self.algorithm_var.get())

    def _selected_grid(self) -> str:
        level = self.level_var.get()
        puzzle_index = self.puzzle_var.get() - 1
        return DATASET[level][puzzle_index]

    def _set_entry(self, row: int, col: int, value: int, editable: bool, color: str) -> None:
        entry = self.entries[row][col]
        entry.configure(state="normal")
        entry.delete(0, tk.END)
        if value != 0:
            entry.insert(0, str(value))
        entry.configure(fg=color)
        if editable:
            entry.configure(state="normal", disabledforeground=color)
        else:
            entry.configure(state="disabled", disabledforeground=color)

    def load_selected_puzzle(self) -> None:
        self.time_var.set("Time: -")
        self.current_solution = None
        grid = self._selected_grid()
        self.initial_board = parse_grid(grid)
        self.given_cells.clear()

        for r in ROWS:
            for c in COLS:
                value = self.initial_board[r][c]
                if value != 0:
                    self.given_cells.add((r, c))
                    self._set_entry(r, c, value, editable=False, color="#101010")
                else:
                    self._set_entry(r, c, 0, editable=True, color="#1D4ED8")

    def read_board_from_ui(self) -> Board:
        board: Board = [[0 for _ in COLS] for _ in ROWS]
        for r in ROWS:
            for c in COLS:
                raw = self.entries[r][c].get().strip()
                board[r][c] = int(raw) if len(raw) == 1 and raw in DIGITS else 0
        return board

    def render_solution(self, solved_board: Board) -> None:
        for r in ROWS:
            for c in COLS:
                value = solved_board[r][c]
                if (r, c) in self.given_cells:
                    self._set_entry(r, c, value, editable=False, color="#101010")
                else:
                    self._set_entry(r, c, value, editable=True, color="#C81E5B")

    def _solve_current_board(self) -> SolveResult:
        board = self.read_board_from_ui()
        grid = board_to_string(board)
        return solve_grid(grid, self.algorithm_var.get())

    def solve_board(self) -> None:
        result = self._solve_current_board()
        if not result.solved or result.board is None:
            self.time_var.set(f"Time: {result.elapsed_seconds:.5f} seconds")
            messagebox.showerror("No Solution", "This board configuration has no valid Sudoku solution.")
            return

        self.render_solution(result.board)
        self.time_var.set(f"Time: {result.elapsed_seconds:.5f} seconds")

        key = self._dataset_key()
        self.solution_cache[key] = clone_board(result.board)
        self.current_solution = clone_board(result.board)

    def _get_or_compute_solution_for_loaded_puzzle(self) -> Optional[Board]:
        key = self._dataset_key()
        if key in self.solution_cache:
            return clone_board(self.solution_cache[key])

        result = solve_grid(self._selected_grid(), self.algorithm_var.get())
        if not result.solved or result.board is None:
            return None

        self.solution_cache[key] = clone_board(result.board)
        return clone_board(result.board)

    def give_hint(self) -> None:
        solution = self.current_solution if self.current_solution is not None else self._get_or_compute_solution_for_loaded_puzzle()
        if solution is None:
            messagebox.showerror("Hint", "Could not generate hint because puzzle solution was not found.")
            return

        current_board = self.read_board_from_ui()
        for r in ROWS:
            for c in COLS:
                if current_board[r][c] == 0:
                    self._set_entry(r, c, solution[r][c], editable=True, color="#C81E5B")
                    self.time_var.set("Time: Hint applied")
                    return

        messagebox.showinfo("Hint", "No empty cells found. Fill or correct entries and try again.")

    def reset_board(self) -> None:
        self.load_selected_puzzle()


def run_gui_smoke_test() -> int:
    root = tk.Tk()
    app = SudokuGUI(root)
    app.solve_board()
    app.reset_board()
    app.give_hint()
    root.update_idletasks()
    root.destroy()
    print("GUI smoke test passed.")
    return 0


def launch_gui() -> int:
    root = tk.Tk()
    SudokuGUI(root)
    root.mainloop()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Module 2 Sudoku Puzzle (CSP + Tkinter)")
    parser.add_argument("--self-test", action="store_true", help="Run verification on all dataset puzzles.")
    parser.add_argument("--gui-smoke-test", action="store_true", help="Run non-interactive GUI smoke test.")
    parser.add_argument("--algorithm", choices=["ac3", "backtracking"], help="Solve a single custom grid.")
    parser.add_argument("--grid", help="81-char grid using 0 or . for blanks.")
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()

    if args.gui_smoke_test:
        return run_gui_smoke_test()

    if args.algorithm and args.grid:
        result = solve_grid(args.grid, args.algorithm)
        if not result.solved:
            print("No solution found.")
            return 1
        print(board_to_string(result.board))
        print(f"time={result.elapsed_seconds:.6f}s assignments={result.assignments}")
        return 0

    return launch_gui()


if __name__ == "__main__":
    raise SystemExit(main())

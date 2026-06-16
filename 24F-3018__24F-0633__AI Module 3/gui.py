from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from exam_system.pipeline import run_exam_management


class ExamManagementGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Automated Exam Management System")
        self.root.geometry("720x420")

        self.students_var = tk.StringVar(value="sample_data/students.csv")
        self.rooms_var = tk.StringVar(value="sample_data/rooms.csv")
        self.faculty_var = tk.StringVar(value="sample_data/faculty.csv")
        self.output_var = tk.StringVar(value="output")

        self._build_form()

    def _build_form(self) -> None:
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill="both", expand=True)

        ttk.Label(container, text="Students CSV").grid(row=0, column=0, sticky="w", pady=6)
        ttk.Entry(container, textvariable=self.students_var, width=60).grid(row=0, column=1, padx=8)
        ttk.Button(container, text="Browse", command=lambda: self._pick_file(self.students_var)).grid(row=0, column=2)

        ttk.Label(container, text="Rooms CSV").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Entry(container, textvariable=self.rooms_var, width=60).grid(row=1, column=1, padx=8)
        ttk.Button(container, text="Browse", command=lambda: self._pick_file(self.rooms_var)).grid(row=1, column=2)

        ttk.Label(container, text="Faculty CSV").grid(row=2, column=0, sticky="w", pady=6)
        ttk.Entry(container, textvariable=self.faculty_var, width=60).grid(row=2, column=1, padx=8)
        ttk.Button(container, text="Browse", command=lambda: self._pick_file(self.faculty_var)).grid(row=2, column=2)

        ttk.Label(container, text="Output Folder").grid(row=3, column=0, sticky="w", pady=6)
        ttk.Entry(container, textvariable=self.output_var, width=60).grid(row=3, column=1, padx=8)
        ttk.Button(container, text="Browse", command=self._pick_folder).grid(row=3, column=2)

        ttk.Button(container, text="Generate Plan", command=self.generate_plan).grid(row=4, column=1, sticky="w", pady=16)

        self.output_box = tk.Text(container, height=14, width=84)
        self.output_box.grid(row=5, column=0, columnspan=3, pady=8)

    def _pick_file(self, target_var: tk.StringVar) -> None:
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if path:
            target_var.set(path)

    def _pick_folder(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.output_var.set(path)

    def generate_plan(self) -> None:
        try:
            result = run_exam_management(
                students_path=self.students_var.get(),
                rooms_path=self.rooms_var.get(),
                faculty_path=self.faculty_var.get(),
                output_dir=self.output_var.get(),
            )
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return

        self.output_box.delete("1.0", tk.END)
        self.output_box.insert(tk.END, result["summary"] + "\n\nGenerated files:\n")
        for name, path in result["files"].items():
            self.output_box.insert(tk.END, f"- {name}: {path}\n")


def main() -> None:
    root = tk.Tk()
    ExamManagementGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()


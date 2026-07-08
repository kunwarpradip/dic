from __future__ import annotations

import sqlite3
from pathlib import Path
from uuid import UUID

from .models import DicLine, Point


def app_data_dir_for_image(image_path: str | Path) -> Path:
    image_path = Path(image_path)
    app_data = image_path.parent / "app_data"
    app_data.mkdir(parents=True, exist_ok=True)
    return app_data


def db_path_for_image(image_path: str | Path) -> Path:
    image_path = Path(image_path)
    return app_data_dir_for_image(image_path) / f"{image_path.stem}.db"


class DicLineRepository:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    def create_tables_if_needed(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS dic_lines (
                    id TEXT PRIMARY KEY,
                    is_manual INTEGER,
                    intensity_difference_tolerance INTEGER,
                    best_fit_line_tolerance REAL,
                    minimum_intensity INTEGER,
                    seed_point_x INTEGER,
                    seed_point_y INTEGER
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS line_points (
                    line_id TEXT REFERENCES dic_lines(id),
                    x INTEGER,
                    y INTEGER,
                    id INTEGER PRIMARY KEY AUTOINCREMENT
                )
                """
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS line_points_line_id_index ON line_points(line_id)"
            )
            con.execute("CREATE INDEX IF NOT EXISTS dic_lines_id_index ON dic_lines(id)")

    def load_lines(self) -> list[DicLine]:
        self.create_tables_if_needed()
        with sqlite3.connect(self.db_path) as con:
            line_rows = con.execute(
                """
                SELECT id, is_manual, intensity_difference_tolerance,
                       best_fit_line_tolerance, minimum_intensity,
                       seed_point_x, seed_point_y
                FROM dic_lines
                ORDER BY rowid
                """
            ).fetchall()
            point_rows = con.execute(
                "SELECT line_id, x, y FROM line_points ORDER BY id"
            ).fetchall()

        points_by_line: dict[str, set[Point]] = {}
        for line_id, x, y in point_rows:
            points_by_line.setdefault(str(line_id), set()).add(Point(int(x), int(y)))

        lines: list[DicLine] = []
        for (
            line_id,
            is_manual,
            intensity_tolerance,
            bfl_tolerance,
            min_intensity,
            seed_x,
            seed_y,
        ) in line_rows:
            seed_point = (
                Point(int(seed_x), int(seed_y))
                if seed_x is not None and seed_y is not None
                else None
            )
            lines.append(
                DicLine(
                    id=UUID(str(line_id)),
                    is_manual=bool(is_manual),
                    points=points_by_line.get(str(line_id), set()),
                    seed_point=seed_point,
                    intensity_difference_tolerance=int(intensity_tolerance or 0),
                    bfl_tolerance=float(bfl_tolerance or 0.0),
                    min_intensity=int(min_intensity or 0),
                )
            )
        return lines

    def save_new_lines(self, lines: list[DicLine]) -> None:
        if not lines:
            return
        self.create_tables_if_needed()
        with sqlite3.connect(self.db_path) as con:
            for line in lines:
                con.execute(
                    """
                    INSERT OR REPLACE INTO dic_lines (
                        id, is_manual, intensity_difference_tolerance,
                        best_fit_line_tolerance, minimum_intensity,
                        seed_point_x, seed_point_y
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(line.id),
                        int(line.is_manual),
                        int(line.intensity_difference_tolerance),
                        float(line.bfl_tolerance),
                        int(line.min_intensity),
                        line.seed_point.x if line.seed_point else None,
                        line.seed_point.y if line.seed_point else None,
                    ),
                )
                con.execute("DELETE FROM line_points WHERE line_id = ?", (str(line.id),))
                con.executemany(
                    "INSERT INTO line_points (line_id, x, y) VALUES (?, ?, ?)",
                    [(str(line.id), p.x, p.y) for p in sorted(line.points)],
                )

    def replace_all(self, lines: list[DicLine]) -> None:
        self.create_tables_if_needed()
        with sqlite3.connect(self.db_path) as con:
            con.execute("DELETE FROM line_points")
            con.execute("DELETE FROM dic_lines")
        self.save_new_lines(lines)

    def delete_lines(self, line_ids: set[UUID]) -> None:
        if not line_ids:
            return
        self.create_tables_if_needed()
        with sqlite3.connect(self.db_path) as con:
            for line_id in line_ids:
                con.execute("DELETE FROM line_points WHERE line_id = ?", (str(line_id),))
                con.execute("DELETE FROM dic_lines WHERE id = ?", (str(line_id),))

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4


@dataclass(frozen=True, order=True)
class Point:
    x: int
    y: int


@dataclass
class DicLine:
    id: UUID = field(default_factory=uuid4)
    is_manual: bool = False
    points: set[Point] = field(default_factory=set)
    seed_point: Point | None = None
    intensity_difference_tolerance: int = 75
    bfl_tolerance: float = 7.0
    min_intensity: int = 120

    @property
    def size(self) -> int:
        return len(self.points)


@dataclass
class DicSession:
    image_path: str | None = None
    db_path: str | None = None
    image_width: int = 0
    image_height: int = 0
    lines: list[DicLine] = field(default_factory=list)
    visible_line_ids: set[UUID] = field(default_factory=set)

    def set_lines(self, lines: list[DicLine]) -> None:
        self.lines = lines
        existing_ids = {line.id for line in lines}
        self.visible_line_ids.intersection_update(existing_ids)

    def add_line(self, line: DicLine) -> None:
        self.lines.append(line)
        self.visible_line_ids.add(line.id)

    def delete_lines(self, line_ids: set[UUID]) -> None:
        self.lines = [line for line in self.lines if line.id not in line_ids]
        self.visible_line_ids.difference_update(line_ids)

    def line_by_id(self, line_id: UUID) -> DicLine | None:
        return next((line for line in self.lines if line.id == line_id), None)

"""Occupancy-grid construction and A* search for oracle planning."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import numpy as np

from .representations import ObjectState


@dataclass(frozen=True)
class GridConfig:
    size: float = 6.4
    resolution: float = 0.05
    safety_margin: float = 0.04

    @property
    def cells(self) -> int:
        return round(self.size / self.resolution)

    @property
    def origin(self) -> np.ndarray:
        return np.array([-self.size / 2.0, -self.size / 2.0])


class OccupancyGrid:
    """Robot-center occupancy grid generated from oriented object boxes."""

    def __init__(self, config: GridConfig | None = None) -> None:
        self.config = config or GridConfig()
        self.data = np.zeros(
            (self.config.cells, self.config.cells), dtype=np.bool_
        )

    def world_to_cell(self, point: np.ndarray) -> tuple[int, int]:
        xy = np.asarray(point, dtype=np.float64)[:2]
        cell_xy = np.floor(
            (xy - self.config.origin) / self.config.resolution
        ).astype(int)
        return int(cell_xy[1]), int(cell_xy[0])

    def cell_to_world(self, cell: tuple[int, int]) -> np.ndarray:
        row, col = cell
        return self.config.origin + self.config.resolution * np.array(
            [col + 0.5, row + 0.5]
        )

    def in_bounds(self, cell: tuple[int, int]) -> bool:
        row, col = cell
        return 0 <= row < self.data.shape[0] and 0 <= col < self.data.shape[1]

    def rasterize(
        self,
        objects: tuple[ObjectState, ...],
        robot_radius: float,
        excluded_ids: frozenset[int] = frozenset(),
    ) -> np.ndarray:
        self.data.fill(False)
        padding = robot_radius + self.config.safety_margin
        for obj in objects:
            if obj.object_id in excluded_ids:
                continue
            self._add_oriented_box(obj, padding)
        return self.data.copy()

    def _add_oriented_box(self, obj: ObjectState, padding: float) -> None:
        half_size = np.asarray(obj.size[:2], dtype=np.float64) / 2.0 + padding
        radius = float(np.linalg.norm(half_size))
        lower = self.world_to_cell(np.asarray(obj.center[:2]) - radius)
        upper = self.world_to_cell(np.asarray(obj.center[:2]) + radius)
        row_min = max(0, min(lower[0], upper[0]))
        row_max = min(self.data.shape[0] - 1, max(lower[0], upper[0]))
        col_min = max(0, min(lower[1], upper[1]))
        col_max = min(self.data.shape[1] - 1, max(lower[1], upper[1]))
        if row_min > row_max or col_min > col_max:
            return

        rows, cols = np.mgrid[row_min : row_max + 1, col_min : col_max + 1]
        world_x = self.config.origin[0] + (cols + 0.5) * self.config.resolution
        world_y = self.config.origin[1] + (rows + 0.5) * self.config.resolution
        delta_x = world_x - obj.center[0]
        delta_y = world_y - obj.center[1]
        cos_yaw = math.cos(obj.yaw)
        sin_yaw = math.sin(obj.yaw)
        local_x = cos_yaw * delta_x + sin_yaw * delta_y
        local_y = -sin_yaw * delta_x + cos_yaw * delta_y
        occupied = (np.abs(local_x) <= half_size[0]) & (
            np.abs(local_y) <= half_size[1]
        )
        self.data[row_min : row_max + 1, col_min : col_max + 1] |= occupied

    def astar(
        self, start_xy: np.ndarray, goal_xy: np.ndarray
    ) -> list[np.ndarray] | None:
        start = self.world_to_cell(start_xy)
        goal = self.world_to_cell(goal_xy)
        if not self.in_bounds(start) or not self.in_bounds(goal):
            return None
        if self.data[start] or self.data[goal]:
            return None

        neighbors = (
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        )
        frontier: list[tuple[float, float, tuple[int, int]]] = []
        heapq.heappush(frontier, (0.0, 0.0, start))
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        cost_so_far = {start: 0.0}

        while frontier:
            _, current_cost, current = heapq.heappop(frontier)
            if current == goal:
                return self._reconstruct(came_from, goal)
            if current_cost > cost_so_far[current]:
                continue
            for d_row, d_col, step_cost in neighbors:
                neighbor = (current[0] + d_row, current[1] + d_col)
                if not self.in_bounds(neighbor) or self.data[neighbor]:
                    continue
                if d_row and d_col:
                    if self.data[current[0] + d_row, current[1]]:
                        continue
                    if self.data[current[0], current[1] + d_col]:
                        continue
                new_cost = current_cost + step_cost
                if new_cost >= cost_so_far.get(neighbor, math.inf):
                    continue
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                heuristic = math.hypot(goal[0] - neighbor[0], goal[1] - neighbor[1])
                heapq.heappush(
                    frontier, (new_cost + heuristic, new_cost, neighbor)
                )
        return None

    def _reconstruct(
        self,
        came_from: dict[tuple[int, int], tuple[int, int] | None],
        goal: tuple[int, int],
    ) -> list[np.ndarray]:
        cells = []
        current: tuple[int, int] | None = goal
        while current is not None:
            cells.append(current)
            current = came_from[current]
        cells.reverse()
        return [self.cell_to_world(cell) for cell in cells]

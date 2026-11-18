"""Non-dominated sets for deployment trade-off reports."""

from typing import Any


def nondominated_names(points: dict[str, tuple[float, ...]]) -> list[str]:
    """Return names that no other point weakly improves in every objective.

    Every coordinate is minimized. A point is dominated when another point is
    less than or equal on all objectives and strictly less on at least one.
    """
    if not points:
        return []
    width = len(next(iter(points.values())))
    if width == 0 or any(len(point) != width for point in points.values()):
        raise ValueError("Pareto points must be non-empty and the same length")
    frontier = []
    for name, point in points.items():
        dominated = any(
            other != name
            and all(points[other][index] <= point[index] for index in range(width))
            and any(points[other][index] < point[index] for index in range(width))
            for other in points
        )
        if not dominated:
            frontier.append(name)
    return frontier


def frontier_from_rows(rows: list[dict[str, Any]], objectives: Any) -> list[str]:
    """Build a frontier from row dictionaries. `objectives` maps a row to minimized values."""
    return nondominated_names({row["name"]: tuple(objectives(row)) for row in rows})

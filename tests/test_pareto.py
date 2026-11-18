from edge_ai_bench.pareto import frontier_from_rows, nondominated_names


def test_nondominated_returns_only_non_dominated_points():
    frontier = nondominated_names({
        "fast_large": (100.0, 10.0),
        "slow_small": (50.0, 20.0),
        "dominated": (100.0, 20.0),
    })
    assert set(frontier) == {"fast_large", "slow_small"}


def test_frontier_from_rows_uses_objective_factory():
    rows = [
        {"name": "a", "size": 10, "latency": 1.0},
        {"name": "b", "size": 5, "latency": 2.0},
        {"name": "c", "size": 10, "latency": 2.0},
    ]
    frontier = frontier_from_rows(rows, lambda row: (row["size"], row["latency"]))
    assert set(frontier) == {"a", "b"}
"""Hungarian algorithm for the linear sum assignment problem (`zero_optax.assignment`)."""

from __future__ import annotations

from zero_optax._tree import asarray
from zero_optax._types import ArrayLike, Tensor


def _solve_hungarian_rect(
    matrix: list[list[float]],
    n_rows: int,
    n_cols: int,
) -> tuple[list[int], list[int]]:
    """Solve the rectangular linear assignment problem for `n_rows <= n_cols` via Kuhn-Munkres.

    Args:
        matrix: 2D cost matrix with `n_rows` rows and `n_cols` columns (`n_rows <= n_cols`).
        n_rows: Number of rows (`>= 0`).
        n_cols: Number of columns (`>= n_rows`).

    Returns:
        Tuple `(row_indices, col_indices)` of length `n_rows`.
    """
    if n_rows == 0 or n_cols == 0:
        return [], []
    u = [0.0] * (n_rows + 1)
    v = [0.0] * (n_cols + 1)
    p = [0] * (n_cols + 1)
    way = [0] * (n_cols + 1)

    for i in range(1, n_rows + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (n_cols + 1)
        used = [False] * (n_cols + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, n_cols + 1):
                if not used[j]:
                    cur = matrix[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    row_to_col: dict[int, int] = {}
    for j in range(1, n_cols + 1):
        if p[j] > 0:
            row_to_col[p[j] - 1] = j - 1

    rows = list(range(n_rows))
    cols = [row_to_col[r] for r in rows]
    return rows, cols


def base_hungarian_algorithm(cost_matrix: ArrayLike) -> tuple[Tensor, Tensor]:
    """Compute an optimal linear sum assignment `(row_indices, col_indices)` for `cost_matrix`.

    Args:
        cost_matrix: 2D array-like of shape `(n, m)` representing assignment costs.

    Returns:
        Pair `(i, j)` of 1D integer `Tensor`s of length `min(n, m)` minimizing `cost_matrix[i, j].sum()`.

    Raises:
        ValueError: If `cost_matrix` is not 2-dimensional.
    """
    c_t = asarray(cost_matrix)
    if c_t.ndim != 2:
        raise ValueError(f"cost_matrix must be 2D, got shape {c_t.shape}.")
    n, m = c_t.shape
    if n == 0 or m == 0:
        return asarray([], dtype="int32"), asarray([], dtype="int32")

    grid: list[list[float]] = [
        [float(c_t[r, c].item()) for c in range(m)] for r in range(n)
    ]
    if n <= m:
        rows, cols = _solve_hungarian_rect(grid, n, m)
        return asarray(rows, dtype="int32"), asarray(cols, dtype="int32")

    grid_t: list[list[float]] = [[grid[r][c] for r in range(n)] for c in range(m)]
    cols_t, rows_t = _solve_hungarian_rect(grid_t, m, n)
    paired = sorted(zip(rows_t, cols_t), key=lambda rc: rc[0])
    out_rows = [r for r, _ in paired]
    out_cols = [c for _, c in paired]
    return asarray(out_rows, dtype="int32"), asarray(out_cols, dtype="int32")


def hungarian_algorithm(cost_matrix: ArrayLike) -> tuple[Tensor, Tensor]:
    """The Hungarian algorithm for the linear assignment problem.

    Args:
        cost_matrix: 2D matrix of assignment costs of shape `(n, m)`.

    Returns:
        Pair `(i, j)` of 1D integer `Tensor`s of row and column indices.
    """
    return base_hungarian_algorithm(cost_matrix)


__all__ = [
    "base_hungarian_algorithm",
    "hungarian_algorithm",
]

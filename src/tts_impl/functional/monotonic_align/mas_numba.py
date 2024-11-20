import numpy as np
from numba import njit, prange


@njit
def maximum_path_each_numba(path, value, t_y, t_x, max_neg_val=-np.inf):
    """Calculate a single maximum path with numba."""
    index = t_x - 1
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            if x == y:
                v_cur = max_neg_val
            else:
                v_cur = value[y - 1, x]
            if x == 0:
                if y == 0:
                    v_prev = 0.0
                else:
                    v_prev = max_neg_val
            else:
                v_prev = value[y - 1, x - 1]
            value[y, x] += max(v_prev, v_cur)

    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1
        if index != 0 and (index == y or value[y - 1, index] < value[y - 1, index - 1]):
            index = index - 1


@njit(parallel=True)
def maximum_path_numba(paths, values, t_ys, t_xs):
    """Calculate batch maximum path with numba."""
    for i in prange(paths.shape[0]):
        maximum_path_each_numba(paths[i], values[i], t_ys[i], t_xs[i])

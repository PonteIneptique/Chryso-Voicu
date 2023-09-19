import numpy as np
from scipy.cluster import _hierarchy
from scipy.cluster.hierarchy import _convert_to_double, _warning, _LINKAGE_METHODS, optimal_leaf_ordering
from scipy.spatial import distance as distance


__all__ = ["manhattan_ward"]


def manhattan_ward(y, optimal_ordering=False):
    """ Hierarchical (agglomerative) clustering on Euclidean data.

    This code is a hack of scipy.cluster.linkage to allow manhattan with ward
    """
    method = "ward"
    metric = "cityblock"

    y = _convert_to_double(np.asarray(y, order='c'))

    #if y.ndim == 2:
    if y.shape[0] == y.shape[1] and np.allclose(np.diag(y), 0):
        if np.all(y >= 0) and np.allclose(y, y.T):
            _warning('The symmetric non-negative hollow observation '
                     'matrix looks suspiciously like an uncondensed '
                     'distance matrix')
    y = distance.pdist(y, metric)

    if not np.all(np.isfinite(y)):
        raise ValueError("The condensed distance matrix must contain only "
                         "finite values.")

    n = int(distance.num_obs_y(y))
    method_code = _LINKAGE_METHODS[method]

    #if method in ['complete', 'average', 'weighted', 'ward']:
    result = _hierarchy.nn_chain(y, n, method_code)

    if optimal_ordering:
        return optimal_leaf_ordering(result, y)
    else:
        return result

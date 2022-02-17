import numpy as np
from sortedcontainers import SortedDict


class DCI:

    """Total index class containing all data and indices."""

    def __init__(self, dim, num_simple, num_composite, data=None):
        """Set up projection vectors and data lists.

        Parameters
        ----------
        dim: int
            Dimension of a datapoint (d)
        num_simple: int
            Number of simple indices (m)
        num_composite: int
            Number of composite indices (L)
        data: numpy.ndarray
            d dimensional data points stored as rows
        """
        self._num_simple = num_simple
        self._num_composite = num_composite
        rng = np.random.default_rng()
        # create num_simple random directions for each composite index
        self._directions = np.array(
            [rng.random((num_simple, dim)) for _ in range(num_composite)])
        self._projections = list()
        for ell in range(num_composite):
            self._directions[ell] /= np.linalg.norm(
                self._directions[ell], axis=1)[:, np.newaxis]
        self._projections = [[SortedDict()] * num_simple] * num_composite
        if data is not None:
            self.add(data)

    def add(self, data):
        """Add data to the index.

        Parameters
        ----------
        data : numpy.ndarray
            d dimensional data points stored as rows
        """
        for ell in range(self._num_composite):
            projections = self._directions @ data.T
            for (j, point_projections) in enumerate(projections):
                for (i, point_projection) in enumerate(point_projections):
                    self._projections[ell][j][point_projection] = data[i]

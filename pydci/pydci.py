import numpy as np
from sortedcontainers import SortedDict
from collections import defaultdict


class DCI:

    """Search data structure containing all data and indices."""

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
            d-dimensional data points stored as rows
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
            d-dimensional data points stored as rows
        """
        for ell in range(self._num_composite):
            projections = self._directions[ell] @ data.T
            for (j, point_projections) in enumerate(projections):
                for (i, point_projection) in enumerate(point_projections):
                    self._projections[ell][j][point_projection] = data[i]

    def query(self, q, k, max_retrieve, max_composite_visit):
        """Find the nearest neighbors to a query.
        Note that this implements the prioritized DCI algorithm.

        Parameters
        ----------
        q : numpy.ndarray
            d-dimensional query points
        k : int
            Number of neighbors to return
        max_retrieve : int
            The maximum number of candidates to pull from each composite index
        max_composite_visit : int
            The maximum number of times to visit each composite index

        Returns
        -------
        k nearest neighbors

        """
        query_projections = [self._directions[ell] @
                             query.T for ell in range(self._num_composite)]
        priorities = [SortedDict()] * self._num_composite
        for ell in range(self._num_composite):
            for j = range(self._num_simple):
                data_projections = self._projections[ell][j]
                query_projection = query_projections[ell][j]
                # find closest projection
                nearest = data_projections.bisect_left(query_projection)
                dist = abs(data_projections.peek_item(nearest)[0]
                           - query_projection)
                if (
                    nearest > 0 and dist >
                    abs(data_projections.peek_item(nearest - 1)[0]
                        - query_projection)
                ):
                    nearest -= 1
                    dist = abs(data_projections[nearest][0] - query_projection)
                best_projection, best_point = data_projections.peek_item(
                    nearest)

                # TODO: Needs refactoring, especially to handle boundaries of
                # projection lists more cleanly

                # add it to the priority queue and track where it came from
                priorities[ell][-dist] = {
                    'projection': best_projection,
                    'point': best_point
                    'simple_index': j,
                    'lower': nearest - 1 if nearest > 0 else None,
                    'upper': nearest + 1
                    if nearest < len(data_projections) - 1 else None
                }
        counter = [defaultdict(int)] * self._num_composite
        candidates = [[]] * self._num_composite
        for _ in range(max_composite_visit):
            for ell in range(self._num_composite):
                if len(candidates[ell]) < max_retrieve:
                    best = priorities[ell].popitem()
                    query_projection = \
                        query_projections[ell][best['simple_index']]
                    # TODO: extract out into its own helper method
                    # find new nearest in jth projections
                    if best['lower'] is not None:
                        lower = self._projections[ell][best['simple_index']].\
                            peek_item(best['lower'])
                        lower_dist = abs(lower[0] - query_projection)
                    else:
                        lower = None
                        lower_dist = np.inf
                    if best["upper"] is not None:
                        upper = self._projections[ell][best['simple_index']].\
                            peek_item(best['upper'])
                        upper_dist = abs(upper[0] - query_projection)
                    else:
                        upper = None
                        upper_dist = np.inf
                    best_point = best['point']  # save before overwriting
                    if lower_dist < upper_dist:  # lower is closer
                        best['projection'], best['point'] = lower
                        best['lower'] = lower - 1 if lower > 0 else None
                        dist = lower_dist
                    else:  # upper is closer
                        best['projection'], best['point'] = upper
                        best['upper'] = upper + 1\
                            if upper <\
                            len(self._projections[ell][best['j']]) - 1\
                            else None
                        dist = upper_dist
                    priorities[ell][-dist] = best

                    # check whether best point is candidate
                    counter[ell][best_point] += 1
                    if counter[ell][best_point] == self._num_simple:
                        candidates[ell].append[best_point]
        candidates_array = np.unique(np.reshape(
            np.array(candidates), -1, self._dim), axis=0)
        best = np.argsort(np.linalg.norm(candidates_array - query, axis=1))
        return candidates_array[best[:k], :]

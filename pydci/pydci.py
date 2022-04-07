import numpy as np
from sortedcontainers import SortedDict
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors


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
        self._dim = dim
        self._num_simple = num_simple
        self._num_composite = num_composite
        rng = np.random.default_rng()
        # create num_simple random directions for each composite index
        self._directions = np.array(
            [rng.standard_normal((num_simple, dim)) for _ in range(num_composite)])
        self._projections = list()
        for ell in range(num_composite):
            self._directions[ell] /= np.linalg.norm(
                self._directions[ell], axis=1)[:, np.newaxis]
        self._projections = [[SortedDict() for _ in range(num_simple)]
                             for _ in range(num_composite)]
        self._num_points = int(0)
        self._data = np.empty((0, dim))
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
                    self._projections[ell][j][point_projection] = i + \
                        self._num_points  # only track indices
        self._data = np.vstack((self._data, data))
        self._num_points += len(data)

    def query(self, q, k, max_retrieve_const, max_composite_visit_const):
        """Find the nearest neighbors to a query.
        Note that this implements the prioritized DCI algorithm.

        Parameters
        ----------
        q : numpy.ndarray
            d-dimensional query points
        k : int
            Number of neighbors to return
        max_retrieve_const : float
            Scaling factor for maximum number of candidates to pull from each
            composite index. Values in the range of 1-10 work well.
        max_composite_visit_const : float
            Scaling factor for the maximum number of times to visit each
            composite index. Values in the range of 1-10 work well.

        Returns
        -------
        k nearest neighbors as indices and points, as well as the number of
        candidates searched through and number of insertions and deletions in
        priority queues to create candidate list

        """
        # set parameters via scaling factors
        # scaling of parameters is based on theory and qualitative uning
        max_retrieve = int(max_retrieve_const * self._num_simple *
                           k * np.log(self._num_points / k) ** 2)
        max_composite_visit = int(max_composite_visit_const * self._num_simple
                                  * k * np.log(self._num_points / k) ** 2)
        # make sure that parameters are large enough to run
        max_retrieve = max(max_retrieve, k)
        max_composite_visit = max(max_composite_visit, k)

        # define counters for operation counting
        int_candidates_counter = 0
        insert_del_counter = 0

        query_projections = [self._directions[ell] @
                             q.T for ell in range(self._num_composite)]
        priorities = [SortedDict() for _ in range(self._num_composite)]
        for ell in range(self._num_composite):
            for j in range(self._num_simple):
                best = self._closest_projection(query_projections[ell][j], ell,
                                                j)
                # add to priority queue using -dist as key
                priorities[ell][-best['dist']] = best

        counter = [defaultdict(int) for _ in range(self._num_composite)]
        candidate_indices = [[] for _ in range(self._num_composite)]

        for current_visit in range(max_composite_visit):
            for ell in range(self._num_composite):
                if (
                    len(candidate_indices[ell]) < max_retrieve
                    and len(priorities[ell]) > 0
                ):
                    best = priorities[ell].popitem()[1]
                    best_point = best['point']
                    # check whether best point is candidate
                    counter[ell][best_point] += 1
                    if counter[ell][best_point] == self._num_simple:
                        candidate_indices[ell].append(best_point)
                    # find new nearest in jth projections (if it exists)
                    query_projection = \
                        query_projections[ell][best['simple_index']]
                    try:
                        self._update_closest(query_projection, best, ell)
                    except IndexError as e:
                        continue
                    priorities[ell][-best['dist']] = best
                    insert_del_counter += 1

        # clear out the empty lists so concatenation doesn't fail
        while [] in candidate_indices:
            candidate_indices.remove([])

        if len(candidate_indices) > 0:
            candidate_indices_array = np.unique(np.ravel(
                np.concatenate(candidate_indices)))
            int_candidates_counter = candidate_indices_array.size
            best = self._brute_force(q, k, candidate_indices_array)
            return (
                best,
                int_candidates_counter,
                insert_del_counter)
        else:
            return ([], int_candidates_counter, insert_del_counter)

    def _closest_projection(self, query_projection, ell, j):
        """Find the closest projection to a query in a simple index.

        Parameters
        ----------
        query_projection : float
            projection of the query using the (`ell`, `j`)th projection
        ell : int
            composite index to look in
        j : int
            simple index to look in

        Returns
        -------
        closest : dict
            A dictionary containing necessary information about the nearest
            point in the simple index. The dictionary has the following keys:
            `projection`: the closest projection
            `point`: the point corresponding to the closest projection
            `dist`: the distance from the query projection to the closest
                projection
            `simple_index`: `j`
            `lower`: the index of the projection one lower than the closest.
                NOTE: if there is no such projection, `lower` is None.
            `upper`: the index of the projection one higher than the closest.
                NOTE: if there is no such projection, `upper` is None.
        """
        data_projections = self._projections[ell][j]
        # find insertion point
        nearest = data_projections.bisect_left(query_projection)
        # check whether the closest point is on left or right of insertion
        if (  # closest point on left
            nearest == len(data_projections)
            or
            (
                nearest > 0
                and
                abs(data_projections.peekitem(nearest)[0] - query_projection) >
                abs(data_projections.peekitem(nearest-1)[0] - query_projection)
            )
        ):
            nearest -= 1
        dist = abs(data_projections.peekitem(nearest)[0] - query_projection)
        best_projection, best_point = data_projections.peekitem(nearest)
        return {
            'projection': best_projection,
            'point': best_point,
            'dist': dist[0],
            'simple_index': j,
            'lower': nearest - 1 if nearest > 0 else None,
            'upper': nearest + 1
            if nearest < len(data_projections) - 1 else None
        }

    def _update_closest(self, query_projection, best, ell):
        """Updates the `best` dictionary with the new closest projection.
        See `_closest_projection` for more information.

        Parameters
        ----------
        query_projection : float
            some projection of a query point
        best : dict
            information on previous closest projections to `query_projection`
        ell : int
            the composite index to look in

        Raises
        -----
        IndexError
            if there are no more projections in the same simple index as best
        """
        # get the lower and upper projection (if they exist)
        if best['lower'] is not None:
            lower = self._projections[ell][best['simple_index']].\
                peekitem(best['lower'])
            lower_dist = abs(lower[0] - query_projection)
        else:
            lower = None
            lower_dist = np.inf
        if best["upper"] is not None:
            upper = self._projections[ell][best['simple_index']].\
                peekitem(best['upper'])
            upper_dist = abs(upper[0] - query_projection)
        else:
            upper = None
            upper_dist = np.inf

        # figure out whether upper or lower projection is closer
        if lower_dist is np.inf and upper_dist is np.inf:
            raise IndexError(
                f'''All projections in ({ell}, {best['simple_index']}) simple
                index have been analyzed'''
            )
        elif lower_dist < upper_dist:  # lower is closer
            best['projection'], best['point'] = lower
            best['lower'] = best['lower'] - 1\
                if best['lower'] > 0 else None
            best['dist'] = lower_dist[0]
        else:  # upper is closer
            best['projection'], best['point'] = upper
            best['upper'] = best['upper'] + 1\
                if best['upper'] <\
                len(self._projections[ell][best['simple_index']])\
                - 1 else None
            best['dist'] = upper_dist[0]

    def _brute_force(self, q, k, candidate_indices):
        """Do a brute force search on specified data.

        Parameters
        ----------
        q : numpy.ndarray
            d-dimensional query points
        k : int
            Number of neighbors to return
        candidate_indices : numpy.ndarray
            List of indices of points in `self._data` to restrict the
            search over

        Returns
        -------
        Indices of k nearest neighbors to query in `self._data`

        """
        if k >= len(candidate_indices):
            return candidate_indices
        else:
            brute = NearestNeighbors(n_neighbors=k, algorithm='brute')
            brute.fit(self._data[candidate_indices])
            _, nearest_indices = brute.kneighbors(q)
            return candidate_indices[nearest_indices.ravel()]

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
                    self._projections[ell][j][point_projection] = (
                        i + self._num_points, data[i])
        self._num_points += len(data)

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
        candidates = [[] for _ in range(self._num_composite)]
        candidate_indices = [[] for _ in range(self._num_composite)]

        for current_visit in range(max_composite_visit):
            for ell in range(self._num_composite):
                if (
                    len(candidates[ell]) < max_retrieve
                    and len(priorities[ell]) > 0
                ):
                    best = priorities[ell].popitem()[1]
                    best_point = best['point']
                    
                    # check whether best point is candidate
                    counter[ell][best_point[0]] += 1
                    if counter[ell][best_point[0]]  == self._num_simple:

                        candidate_indices[ell].append(best_point[0])
                        candidates[ell].append(best_point[1])
                        
                    query_projection = \
                        query_projections[ell][best['simple_index']]
                    # find new nearest in jth projections (if it exists)
                    try:
                        self._update_closest(query_projection, best, ell)
                    except IndexError as e:
                        continue
                        
                    priorities[ell][-best['dist']] = best
                    insert_del_counter += 1
                    
        # clear out the empty lists so concatenation doesn't fail

        while [] in candidates:
            candidates.remove([])
        while [] in candidate_indices:
            candidate_indices.remove([])
            
        if len(candidates) > 0:
            candidates_array, indices = np.unique(np.reshape(np.concatenate(
                candidates), (-1, self._dim)), axis=0, return_index=True)
            candidate_indices_array = np.ravel(
                np.concatenate(candidate_indices))[indices]
            int_candidates_counter = candidate_indices_array.size
            best = np.argsort(np.linalg.norm(candidates_array - q, axis=1))
            return (
                candidate_indices_array[best[:min(k, len(best))]],
                candidates_array[best[:min(k, len(best))], :],
                int_candidates_counter,
                insert_del_counter)
        else:
            return ([], [], int_candidates_counter, insert_del_counter)

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

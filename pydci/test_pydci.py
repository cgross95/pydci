import numpy as np
from pydci import DCI
from sklearn.neighbors import NearestNeighbors
from pytest import approx


def test_knn():
    """Test against scikit-learn's brute force method"""
    d = 3  # Dimension of data
    N = 100  # Number of data points
    k = 3  # Number of nearest neighbors to find

    # Setup random data and query point
    rng = np.random.default_rng(42)
    X = rng.random((N, d))
    q = rng.random((1, d))

    # sklearn
    brute = NearestNeighbors(n_neighbors=k, algorithm='brute')
    brute.fit(X)
    true_distances, true_indices = brute.kneighbors(q)
    true_neighbors = X[true_indices.ravel(), :]

    # DCI
    dci = DCI(dim=d, num_simple=10, num_composite=3, data=X)
    approx_indices, _, _ = dci.query(q, k, max_retrieve_const=1,
                                     max_composite_visit_const=1)
    approx_neighbors = X[approx_indices, :]

    # Actual first 10 neighbors
    # neighbors = np.array([[0.78273523, 0.08273,    0.48665833],
    #                              [0.7783835,  0.19463871, 0.466721],
    #                              [0.67765867, 0.1218325,  0.50632993],
    #                              [0.92480843, 0.02485949, 0.55519804],
    #                              [0.55403614, 0.10857574, 0.67224009],
    #                              [0.92097047, 0.16553172, 0.28472008],
    #                              [0.58106114, 0.3468698,  0.59091549],
    #                              [0.71946296, 0.43209304, 0.62730884],
    #                              [0.46187723, 0.16127178, 0.50104478],
    #                              [0.6824955,  0.13975248, 0.1999082]])
    assert approx_neighbors == approx(true_neighbors)
    assert approx_indices == approx(true_indices.ravel())


if __name__ == '__main__':
    test_knn()

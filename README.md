# Dynamic Continuous Indexing (in Python)

This is a Python only implementation of the [Dynamic Continuous Indexing (DCI)](https://github.com/ke-li/dci-knn) algorithm proposed by [Jitendra Malik](https://people.eecs.berkeley.edu/~malik/) and [Ke Li](https://www.sfu.ca/~keli/) for quickly solving the k-nearest neighbors problem. See also Ke Li's [original implementation](https://github.com/ke-li/dci-knn) in C with Python and TensorFlow bindings.

# Installation

- To just use `pydci`:
	`pip install .`
- If you'd like to modify it:
	`pip install --editable .`

# Tests 

- Dependencies:
	`pip install -r requirements_test.txt`
- Running:
	`pytest pydci/test_pydci.py`

# Usage

- Instantiate a DCI index with `dci = DCI(dim, num_simple, num_composite)`
	- `dim`: Dimension of data points
	- `num_simple`: Number of simple indices (i.e., projection directions) to contain in each composite index; can be about half the dimension of data points
	- `num_composite`: Number of groups of simple indices; can be small, e.g., 3
- Add data set with `dci.add(X)`; can be called multiple times to add more data
	- `X`: Dataset with each point as a row
- Search for k-nearest neighbors of query point with `dci.query(q, k, max_retrieve_const, max_composite_visit_const)`
	- `q`: Query point
	- `k`: Number of nearest neighbors
	- `max_retrieve_const`: Determines computational extent of query step; higher takes longer but is more accurate, values between 1-10 work well
	- `max_composite_visit_const`: Determines computational extent of query step; higher takes longer but is more accurate, values between 1-10 work well

See `pydci/test_pydci.py` for a starter example.

# About

Written by Craig Gross (<grosscra@msu.edu>) and Cullen Haselby (<haselbyc@msu.edu>) for a project in Michigan State University's [MSIM](https://math.msu.edu/msim/) course.

# Why reimplement?

An important property of DCI is that it supports fast (sublinear) insertions to the underlying data structure. This functionality is not available in the original implementation. Though this pure Python implementation will be slower than the interface to the C code, we are only interested in understanding the overall scaling of DCI when applied to solve nearest neighbor problems on streaming datasets.


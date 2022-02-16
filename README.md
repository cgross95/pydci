# Dynamic Continuous Indexing (in Python)

This is a Python only implementation of the [Dynamic Continuous Indexing (DCI)](https://github.com/ke-li/dci-knn) algorithm proposed by [Jitendra Malik](https://people.eecs.berkeley.edu/~malik/) and [Ke Li](https://www.sfu.ca/~keli/) for quickly solving the k-nearest neighbors problem. See also Ke Li's [original implementation](https://github.com/ke-li/dci-knn) in C with Python and TensorFlow bindings.

# About

Written by Craig Gross (<grosscra@msu.edu>) for a project in Michigan State University's [MSIM](https://math.msu.edu/msim/) course.

# Why reimplement?

An important property of DCI is that it supports fast (sublinear) insertions to the underlying data structure. This functionality is not available in the original implementation. Though this pure Python implementation will be slower than the interface to the C code, we are only interested in understanding the overall scaling of DCI when applied to solve nearest neighbor problems on streaming datasets.


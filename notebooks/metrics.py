import numba
from numba import prange
import numpy as np


@numba.njit
def jaccard_similarity_matrix_weighted(references: np.ndarray, queries: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Returns matrix of weighted jaccard indices between all-vs-all vectors of references
    and queries.

    Parameters
    ----------
    references
        Reference vectors as 2D numpy array. Expects that vector_i corresponds to
        references[i, :].
    queries
        Query vectors as 2D numpy array. Expects that vector_i corresponds to
        queries[i, :].

    Returns
    -------
    scores
        Matrix of all-vs-all similarity scores. scores[i, j] will contain the score
        between the vectors references[i, :] and queries[j, :].
    """
    size1 = references.shape[0]
    size2 = queries.shape[0]
    scores = np.zeros((size1, size2))
    for i in range(size1):
        for j in range(size2):
            scores[i, j] = jaccard_index_weighted(references[i, :], queries[j, :], weights)
    return scores


@numba.njit
def jaccard_index_weighted(u: np.ndarray, v: np.ndarray, weights: np.ndarray) -> np.float64:
    r"""Computes a weighted Jaccard-index (or Jaccard similarity coefficient) of two boolean
    1-D arrays.
    The Jaccard index between 1-D boolean arrays `u` and `v`,
    is defined as

    .. math::

       J(u,v) = \\frac{u \cap v}
                {u \cup v}

    Parameters
    ----------
    u :
        Input array. Expects boolean vector.
    v :
        Input array. Expects boolean vector.

    Returns
    -------
    jaccard_similarity
        The Jaccard similarity coefficient between vectors `u` and `v`.
    """
    u_or_v = np.bitwise_or(u != 0, v != 0)
    u_and_v = np.bitwise_and(u != 0, v != 0)
    jaccard_score = 0
    if u_or_v.sum() != 0:
        u_or_v = u_or_v * weights
        u_and_v = u_and_v * weights
        jaccard_score = np.float64(u_and_v.sum()) / np.float64(u_or_v.sum())
    return jaccard_score


def jaccard_similarity_matrix(references: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Returns matrix of jaccard indices between all-vs-all vectors of references
    and queries.

    Parameters
    ----------
    references
        Reference vectors as 2D numpy array. Expects that vector_i corresponds to
        references[i, :].
    queries
        Query vectors as 2D numpy array. Expects that vector_i corresponds to
        queries[i, :].

    Returns
    -------
    scores
        Matrix of all-vs-all similarity scores. scores[i, j] will contain the score
        between the vectors references[i, :] and queries[j, :].
    """
    # The trick to fast inference is to use float32 since it allows using BLAS
    references = np.array(references, dtype=np.float32)  # R,N
    queries = np.array(queries, dtype=np.float32)  # Q,N
    intersection = references @ queries.T  # R,N @ N,Q -> R,Q
    union = np.sum(references, axis=1, keepdims=True) + np.sum(queries,axis=1, keepdims=True).T  # R,1+1,Q -> R,Q
    union -= intersection
    jaccard = np.nan_to_num(intersection / union)  # R,Q
    return jaccard


@numba.njit
def ruzicka_similarity(A, B):
    """
    Calculate the Ruzicka similarity between two count vectors.
    
    Parameters:
    A (array-like): First count vector.
    B (array-like): Second count vector.
    
    Returns:
    float: Ruzicka similarity.
    """
    
    min_sum = np.sum(np.minimum(A, B))
    max_sum = np.sum(np.maximum(A, B))
    
    return min_sum / max_sum


@numba.njit
def ruzicka_similarity_sparse(keys1, values1, keys2, values2) -> float:
    """
    Calculate the Ruzicka similarity between two sparse count vectors.

    Parameters:
    keys1, values1 (array-like): Keys and values for the first sparse vector.
    keys2, values2 (array-like): Keys and values for the second sparse vector.
    """
    i, j = 0, 0
    min_sum, max_sum = 0.0, 0.0

    while i < len(keys1) and j < len(keys2):
        if keys1[i] == keys2[j]:
            min_sum += min(values1[i], values2[j])
            max_sum += max(values1[i], values2[j])
            i += 1
            j += 1
        elif keys1[i] < keys2[j]:
            max_sum += values1[i]
            i += 1
        else:
            max_sum += values2[j]
            j += 1

    # Add remaining values from both vectors
    while i < len(keys1):
        max_sum += values1[i]
        i += 1

    while j < len(keys2):
        max_sum += values2[j]
        j += 1

    return min_sum / max_sum


@numba.jit(nopython=True, fastmath=True, parallel=True)
def ruzicka_similarity_matrix(references: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Returns matrix of Ruzicka similarity between all-vs-all vectors of references and queries.

    Parameters
    ----------
    references
        Reference vectors as 2D numpy array. Expects that vector_i corresponds to
        references[i, :].
    queries
        Query vectors as 2D numpy array. Expects that vector_i corresponds to
        queries[i, :].

    Returns
    -------
    scores
        Matrix of all-vs-all similarity scores. scores[i, j] will contain the score
        between the vectors references[i, :] and queries[j, :].
    """
    size1 = references.shape[0]
    size2 = queries.shape[0]
    scores = np.zeros((size1, size2)) #, dtype=np.float32)
    for i in prange(size1)::
        for j in range(size2):
            scores[i, j] = ruzicka_similarity(references[i, :], queries[j, :])
    return scores


@numba.jit(nopython=True, fastmath=True, parallel=True)
def ruzicka_similarity_matrix_sparse(
    references: list, queries: list) -> np.ndarray:
    """Returns matrix of Ruzicka similarity between all-vs-all vectors of references and queries.

    Parameters
    ----------
    references:
        List of sparse fingerprints (tuple of two arrays: keys and counts).
    queries
        List of sparse fingerprints (tuple of two arrays: keys and counts).

    Returns
    -------
    scores:
        Matrix of all-vs-all similarity scores. scores[i, j] will contain the score
        between the vectors references[i, :] and queries[j, :].
    """
    size1 = len(references)
    size2 = len(queries)
    scores = np.zeros((size1, size2))
    for i in prange(size1):
        for j in range(size2):
            scores[i, j] = ruzicka_similarity_sparse(
                references[i][0], references[i][1],
                queries[j][0], queries[j][1])
    return scores


@numba.njit
def ruzicka_similarity_weighted(A, B, weights):
    """
    Calculate the weighted Ruzicka similarity between two count vectors.
    
    Parameters:
    ----------
        A (array-like): First count vector.
        B (array-like): Second count vector.
        weights: weights for every vector bit
    
    Returns:
    float: Ruzicka similarity.
    """
    
    min_sum = np.sum(np.minimum(A, B) * weights)
    max_sum = np.sum(np.maximum(A, B) * weights)
    
    return min_sum / max_sum


@numba.jit(nopython=True, fastmath=True, parallel=True)
def ruzicka_similarity_matrix_weighted(references: np.ndarray, queries: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Returns matrix of Ruzicka similarity between all-vs-all vectors of references and queries.

    Parameters
    ----------
    references
        Reference vectors as 2D numpy array. Expects that vector_i corresponds to
        references[i, :].
    queries
        Query vectors as 2D numpy array. Expects that vector_i corresponds to
        queries[i, :].

    Returns
    -------
    scores
        Matrix of all-vs-all similarity scores. scores[i, j] will contain the score
        between the vectors references[i, :] and queries[j, :].
    """
    size1 = references.shape[0]
    size2 = queries.shape[0]
    scores = np.zeros((size1, size2)) #, dtype=np.float32)
    for i in prange(size1):
        for j in range(size2):
            scores[i, j] = ruzicka_similarity_weighted(references[i, :], queries[j, :], weights)
    return scores

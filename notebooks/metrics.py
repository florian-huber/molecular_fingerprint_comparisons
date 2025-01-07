import numba
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


from numba import prange

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
    #A = np.array(A)
    #B = np.array(B)
    
    min_sum = np.sum(np.minimum(A, B))
    max_sum = np.sum(np.maximum(A, B))
    
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
    for i in prange(size1):
    #for i in range(size1):
        for j in range(size2):
            scores[i, j] = ruzicka_similarity(references[i, :], queries[j, :])
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
    #for i in range(size1):
        for j in range(size2):
            scores[i, j] = ruzicka_similarity_weighted(references[i, :], queries[j, :], weights)
    return scores

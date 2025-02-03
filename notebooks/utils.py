import os

import numpy as np
import numba
import urllib.request
from pathlib import Path
from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem
from tqdm import tqdm


class FingerprintGenerator:
    def __init__(self, fpgen):
        self.fpgen = fpgen

    def fingerprint_from_smiles(self, smiles, count=False):
        """Compute fingerprint from SMILES using the generator attribute.
        
        Parameters:
        smiles (str): The SMILES string of the molecule.
        count (bool): If True, returns the count fingerprint, else the regular fingerprint.

        Returns:
        np.array: The fingerprint as a NumPy array, or None if there's an error.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if count:
                return self.fpgen.GetCountFingerprintAsNumPy(mol)
            return self.fpgen.GetFingerprintAsNumPy(mol)
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            return None


class SparseFingerprintGenerator:
    def __init__(self, fpgen):
        self.fpgen = fpgen

    def fingerprint_from_smiles(self, smiles: str, count: bool = False, bit_weighing: dict = None):
        """Compute sparse fingerprint from SMILES using the generator attribute.
        
        Parameters:
        smiles: 
            The SMILES string of the molecule.
        count: 
            If True, returns the count fingerprint, else the regular fingerprint.
        bit_weighing:
            Optional. When a dictionary of shape {bit: value} is given, the respective bits will be multiplied
            by the respective given value. Fingerprint bits not in this dictionary will be multiplied
            by one, so it is generally advisable to use normalized bit weights.

        Returns:
        dict: A dictionary where keys are bit indices and values are counts (for count fingerprints)
              or a list of indices for regular sparse fingerprints.
        """
        if (bit_weighing is not None) and not count:
            raise NotImplementedError("Weighing is currently only implemented for count vectors.")
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if count:
                fp_dict = self.fpgen.GetSparseCountFingerprint(mol).GetNonzeroElements()
                return (prepare_sparse_vector(fp_dict, bit_weighing))
            return np.array(sorted(self.fpgen.GetSparseCountFingerprint(mol).GetNonzeroElements().keys()), dtype=np.int64)
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            return None


def prepare_sparse_vector(sparse_fp_dict: dict, bit_weighing: dict = None):
    """Convert dictionaries to sorted arrays.
    """
    keys = np.array(sorted(sparse_fp_dict.keys()), dtype=np.int64)
    if bit_weighing is None:
        values = np.array([sparse_fp_dict[k] for k in keys], dtype=np.int32)
    else:
        values = np.array([sparse_fp_dict[k] * bit_weighing.get(k, 1) for k in keys], dtype=np.float32)
    return keys, values


def compute_fingerprints(compounds, fpgen, count=True, sparse=True):
    if sparse:
        fp_generator = SparseFingerprintGenerator(fpgen)
    else:
        fp_generator = FingerprintGenerator(fpgen)
    
    fingerprints = []
    for inchikey, row in tqdm(compounds.iterrows(), total=len(compounds)):
        fp = fp_generator.fingerprint_from_smiles(row.smiles, count)
        if fp is None:
            print(f"Missing fingerprint for {inchikey}: {row.smiles}")
        else:
            fingerprints.append(fp)
    return fingerprints


def fingerprint_from_smiles_wrapper(smiles, fpgen, count=False):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if count:
            return fpgen.GetCountFingerprintAsNumPy(mol)
        return fpgen.GetFingerprintAsNumPy(mol)
    except:
        return None


@numba.njit
def count_fingerprint_keys(fingerprints, max_keys: int = 10**7):
    """
    Count the occurrences of keys across all sparse fingerprints.

    Parameters:
    fingerprints (list of tuples): 
        Each tuple contains two numpy arrays: (keys, values) for a fingerprint.
    max_keys:
        Maximum number of unique bits that can be counted.

    Returns:
        A tuple of 3 Numpy arrays (unique_keys, counts, first_instances).
    """

    unique_keys = np.zeros(max_keys, dtype=np.int64)
    counts = np.zeros(max_keys, dtype=np.int32)
    first_instances = np.zeros(max_keys, dtype=np.int16)  # Store first fingerprint where the respective bit occurred (for later analysis)
    num_unique = 0
    reached_max_keys = False

    for idx, (keys, _) in enumerate(fingerprints):
        for key in keys:
            # Check if the key is already in unique_keys
            found = False
            for i in range(num_unique):
                if unique_keys[i] == key:
                    counts[i] += 1
                    found = True
                    break
            # If the key is new, add it
            if not found:
                if (num_unique >= max_keys):
                    if not reached_max_keys:
                        print(f"Maximum number of keys was reached at fingerprint number {idx}.")
                        print("Consider raising the max_keys argument.")
                        reached_max_keys = True
                    continue
                unique_keys[num_unique] = key
                counts[num_unique] = 1
                first_instances[num_unique] = idx
                num_unique += 1

    # Trim arrays to the actual size and sort by key
    bit_order = np.argsort(unique_keys[:num_unique])
    return unique_keys[bit_order], counts[bit_order], first_instances[bit_order]


import numpy as np
from scipy.stats import rankdata


def percentile_scores(similarities: np.ndarray) -> np.ndarray:
    """
    Computes percentile ranks (0..100) for the unique upper-triangular (k=1) entries
    of a symmetrical similarity matrix using scipy's rankdata with average ranks.
    The results are then mirrored to the lower triangle so that the returned matrix
    is again symmetric.
    
    The diagonal is left as-is (default: 0.0), but you can adjust it if needed.

    Parameters
    ----------
    similarities : np.ndarray
        2D symmetric similarity matrix, shape (N, N).

    Returns
    -------
    np.ndarray
        A new matrix of the same shape, where the upper-triangular entries (and their
        mirrored lower-triangular counterparts) have been replaced by percentile ranks.
        The diagonal is untouched (default = 0).
    """
    # Step 1: Extract the upper-triangular entries (excluding diagonal)
    iu1 = np.triu_indices(similarities.shape[0], k=1)
    arr = similarities[iu1]

    # Step 2: Rank them using 'average' method (duplicates get same rank)
    ranks = rankdata(arr, method="average")  # Ranks in [1..len(arr)]
    
    # Step 3: Convert ranks to percentile range [0..100]
    percentiles = (ranks - 1) / (len(ranks) - 1) * 100 if len(ranks) > 1 else np.zeros_like(ranks)

    # Step 4: Create a new matrix for output, same shape and dtype
    percentile_matrix = np.zeros_like(similarities, dtype=float)
    
    # Step 5: Place the percentile scores back into the upper triangle
    percentile_matrix[iu1] = percentiles

    # Step 6: Mirror to the lower triangle
    percentile_matrix = percentile_matrix + percentile_matrix.T

    # Optionally, keep or modify the diagonal; default = 0.0
    # If you'd like to keep the original diagonal, uncomment this line:
    # np.fill_diagonal(percentile_matrix, np.diag(similarities))

    return percentile_matrix


def compute_idf(vector_array):
    """Compute inverse document frequency (IDF).duplicates
    """
    N = vector_array.shape[0]
    return np.log(N / (vector_array > 0).sum(axis=0))


def remove_diagonal(matrix):
    """Removes the diagonal from a matrix.

    Meant for removing matches of spectra against itself.
    """
    # Get the number of rows and columns
    nr_of_rows, nr_of_cols = matrix.shape
    if nr_of_rows != nr_of_cols:
        raise ValueError("Expected predictions against itself")

    # Create a mask for the diagonal elements
    diagonal_mask = np.eye(nr_of_rows, dtype=bool)

    # Use the mask to remove the diagonal elements
    matrix_without_diagonal = matrix[~diagonal_mask].reshape(nr_of_rows, nr_of_cols - 1)
    return matrix_without_diagonal

def download_dataset(url: str, output_dir: str = None):
    if output_dir is None:
        output_dir = Path(os.getcwd()).parents[0]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fn = Path(url).name

    urllib.request.urlretrieve(url, os.path.join(output_dir, fn))
    print(f"File {fn} was downloaded successfully.")
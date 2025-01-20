import numpy as np
from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem


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

    def fingerprint_from_smiles(self, smiles, count=False):
        """Compute sparse fingerprint from SMILES using the generator attribute.
        
        Parameters:
        smiles (str): The SMILES string of the molecule.
        count (bool): If True, returns the count fingerprint, else the regular fingerprint.

        Returns:
        dict: A dictionary where keys are bit indices and values are counts (for count fingerprints)
              or a list of indices for regular sparse fingerprints.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if count:
                fp_dict = self.fpgen.GetSparseCountFingerprint(mol).GetNonzeroElements()
                return (prepare_sparse_vector(fp_dict))
            return list(self.fpgen.GetSparseCountFingerprint(mol).GetNonzeroElements().keys())
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            return None


def prepare_sparse_vector(sparse_fp_dict):
    """Convert dictionaries to sorted arrays.
    """
    keys = np.array(sorted(sparse_fp_dict.keys()), dtype=np.int64)
    values = np.array([sparse_fp_dict[k] for k in keys], dtype=np.int32)
    return keys, values


def compute_all_fingerprints(compounds, fpgen, count):
    valid_smiles = []
    valid_compounds = []
    fingerprints = []
    
    for inchikey, row in tqdm(compounds.iterrows(), total=len(compounds)):
        fp = fingerprint_from_smiles_wrapper(row.smiles, fpgen, count)
        if fp is None:
            print(f"Missing fingerprint for {inchikey}: {row.smiles}")
        else:
            fingerprints.append(fp)
            valid_smiles.append(row.smiles)
            valid_compounds.append(inchikey)
    
    # Convert the list of fingerprints to a 2D NumPy array
    fingerprints = np.vstack(fingerprints)
    return fingerprints, valid_smiles, valid_compounds


def fingerprint_from_smiles_wrapper(smiles, fpgen, count=False):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if count:
            return fpgen.GetCountFingerprintAsNumPy(mol)
        return fpgen.GetFingerprintAsNumPy(mol)
    except:
        return None


def compute_idf(vector_array):
    """Compute inverse document frequenccy (IDF).duplicates
    """
    N = vector_array.shape[0]
    return np.log(N / (vector_array > 0).sum(axis=0))


def remove_diagonal(matrix):
    """Removes the diagonal from a matrix

    meant for removing matches of spectra against itself. """
    # Get the number of rows and columns
    nr_of_rows, nr_of_cols = matrix.shape
    if nr_of_rows != nr_of_cols:
        raise ValueError("Expected predictions against itself")

    # Create a mask for the diagonal elements
    diagonal_mask = np.eye(nr_of_rows, dtype=bool)

    # Use the mask to remove the diagonal elements
    matrix_without_diagonal = matrix[~diagonal_mask].reshape(nr_of_rows, nr_of_cols - 1)
    return matrix_without_diagonal

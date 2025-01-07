from rdkit.Chem import rdFingerprintGenerator


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

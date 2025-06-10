import argparse
import importlib.util
import json
import os
from typing import Callable, Optional

import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem
from tqdm import tqdm


def read_data(input_data: str):
    # input is a file path for csv
    if os.path.isfile(input_data) and input_data.endswith(".csv"):
        data = pd.read_csv(input_data)
        if isinstance(data, pd.DataFrame):
            if "smiles" in data.columns:
                return list(data.smiles)

    # input is a list of SMILES strings
    if isinstance(input_data, str) and not os.path.isfile(input_data):
        return [x.strip() for x in input_data.split(",")]

    return None


def _safe_get_mol_from_smiles(smi: str):
    try:
        return Chem.MolFromSmiles(smi)
    except Exception as e:
        return None


def get_mols_from_smiles(smiles: list):
    mols = Parallel(n_jobs=-1, backend="threading")(
        delayed(_safe_get_mol_from_smiles)(smi)
        for smi in tqdm(
            smiles, total=len(smiles), desc="Generate Molecules from SMILES"
        )
    )
    return [mol for mol in mols if mol is not None]


def get_function_object(path_to_pyfile: str, funcname: str) -> Callable:
    spec = importlib.util.spec_from_file_location("tmp_module", path_to_pyfile)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from path: {path_to_pyfile}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ImportError(f"Failed to execute module {path_to_pyfile}: {e}")

    if not hasattr(module, funcname):
        raise AttributeError(
            f"Function '{funcname}' not found in module '{path_to_pyfile}'."
        )

    function = getattr(module, funcname)
    if not callable(function):
        raise ValueError(f"Object '{funcname}' in '{path_to_pyfile}' is not callable.")

    return function


def _parse_fpgen_args(fpgen_args: Optional[str]) -> dict:
    if not fpgen_args:
        return {}
    try:
        fpgen_kwargs = json.loads(fpgen_args)
        if not isinstance(fpgen_kwargs, dict):
            raise ValueError("Parsed fingerprint arguments must be a dictionary.")
        return fpgen_kwargs
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON string for fingerprint generator arguments: {e}")


def main(smiles: list, fpgen_path: str, fpgen_func: str, fpgen_args: str = None, output_path: str = None):
    if not isinstance(smiles, list):
        raise ValueError("SMILES must be a list")

    mols = get_mols_from_smiles(smiles)
    if not mols:
        raise ValueError("No valid molecules found")

    # Get fingerprint generator function and arguments
    function = get_function_object(fpgen_path, funcname=fpgen_func)
    fpgen_kwargs = _parse_fpgen_args(fpgen_args)

    # Assuming the function takes a list of RDKit Mol objects and returns fingerprints
    fingerprints = [
        function(mol, **fpgen_kwargs)
        for mol in tqdm(mols, total=len(mols), desc="Generate fingerprints from mols.")
    ]

    # TODO: save fingerprints to output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Molecular Fingerprint Comparison CLI Tool",
        description="What the program does",
        epilog="Text at the bottom of help",
    )

    parser.add_argument(
        "smiles",
        type=read_data,
        help="SMILES to process. Can be a file path to a CSV file with a 'smiles' column or a comma-separated list of SMILES strings.",
    )
    parser.add_argument(
        "fpgen_path",
        help="Module path to the fingerprint generator function. Must be a valid Python file path.",
    )
    parser.add_argument(
        "fpgen_func",
        help="Function to call on the fingerprint generator object. Must be a valid function that takes a RDKit mol and returns a fingerprint.",
    )
    parser.add_argument(
        "output_path",
    )
    parser.add_argument("-fpgen_args", "--fpgen_args")

    args = parser.parse_args()

    main(
        smiles=args.smiles,
        fpgen_path=args.fpgen_path,
        fpgen_func=args.fpgen_func,
        fpgen_args=args.fpgen_args,
        output_path=args.output_path,
    )

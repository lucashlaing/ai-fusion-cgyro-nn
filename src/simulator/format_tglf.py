import argparse
import shutil
import os
from typing import List
from pathlib import Path

TGLF_KEYS_FOR_REMOVAL = [
    'BT_EXP',
    'NN_MAX_ERROR',
    'VNS_SHEAR_1',
    'VNS_SHEAR_2',
    'RLNP_CUTOFF',
    'VTS_SHEAR_3',
    'DAMP_SIG',
    'VTS_SHEAR_2',
    'RHO_E',
    'VTS_SHEAR_1',
    'B_UNIT',
    'SAT_GEO0_OUT',
    'VNS_SHEAR_3',
    'WDIA_TRAPPED',
    'SHAPE_COS0',
    'SHAPE_COS1',
    'SHAPE_C0S2',
    'SHAPE_S_COS0',
    'SHAPE_S_COS1',
    'SHAPE_S_COS2',
    'KY',
    'N_MODES'
]

TGLF_KEYS_TO_REPLACE = {
    'USE_MHD_RULE': 'T',
    'USE_BPAR': 'F',
    'WRITE_WAVEFUNCTION_FLAG': 1,
    'NKY': 24
}

TGLF_KEYS_TO_ADD = {
    'NMODES': 5
}

def get_child_directories_at_depth_1(path_str):
    parent_path = Path(path_str)
    return [item.name for item in parent_path.iterdir() if item.is_dir()]

def get_all_files_recursively(root_dir):
    file_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            file_paths.append(os.path.join(dirpath, filename))
    return file_paths

def refactor_tglf_file(filepath: str, prefixes_to_remove: List[str], to_replace: dict, to_add: dict):
    if not os.path.exists(filepath):
        print(f"Error: File not found at path: {filepath}")
        return

    filtered_lines = []

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        for line in lines:
            stripped_line = line.lstrip().upper()

            should_remove = any(stripped_line.startswith(prefix.upper()) for prefix in prefixes_to_remove)
            r_key, r_val = None, None
            should_replace = False
            for prefix in to_replace.keys():
                if stripped_line.startswith(prefix.upper()):
                    should_replace = True
                    r_key, r_val = prefix, to_replace[prefix]
                    break

            if not should_remove and not should_replace:
                filtered_lines.append(line)
            elif should_replace:
                filtered_lines.append(f"{r_key} = {r_val}\n")

        for key in to_add.keys():
            filtered_lines.append(f"{key} = {to_add[key]}\n")

        with open(filepath, 'w') as f:
            f.writelines(filtered_lines)

    except IOError as e:
        print(f"An error occurred while reading or writing the file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def process_tglf_batches(source_dir: str, destination_dir: str):
    """
    Copies and refactors TGLF input files from the source directory to the destination directory.

    Args:
        source_dir (str): Root source directory containing batch subdirectories.
        destination_dir (str): Root destination directory where refactored files will be placed.
    """
    os.makedirs(destination_dir, exist_ok=True)
    batch_dirs = get_child_directories_at_depth_1(source_dir)

    for batch_dir in batch_dirs:
        batch_name = batch_dir.split('/')[-1]
        tglf_batch_dir = os.path.join(source_dir, batch_dir, 'tglf')

        try:
            files = get_all_files_recursively(tglf_batch_dir)
            if not files:
                print(f"No files found in {tglf_batch_dir}, skipping.")
                continue

            dest_dir = os.path.join(destination_dir, batch_name)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, 'input.tglf')

            shutil.copy(files[0], dest_path)
            refactor_tglf_file(dest_path, TGLF_KEYS_FOR_REMOVAL, TGLF_KEYS_TO_REPLACE, TGLF_KEYS_TO_ADD)
            print(f"Refactored TGLF file at {dest_path}")

        except FileExistsError:
            print(f"Error: Destination directory '{dest_dir}' already exists.")
        except Exception as e:
            print(f"An error occurred while processing batch '{batch_name}': {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_dir", required=True)
    parser.add_argument("-d", "--destination_dir", required=True)
    args = parser.parse_args()

    process_tglf_batches(args.source_dir, args.destination_dir)

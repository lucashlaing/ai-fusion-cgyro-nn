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
    """
    Returns a list of immediate child directories (depth 1) for a given path.

    Args:
        path_str (str): The path to the parent directory.

    Returns:
        list: A list of Path objects representing the child directories.
    """
    parent_path = Path(path_str)
    child_directories = [
        item.name for item in parent_path.iterdir() if item.is_dir()
    ]
    return child_directories

def get_all_files_recursively(root_dir):
    """
    Recursively retrieves a list of all file paths within a given directory.

    Args:
        root_dir (str): The path to the root directory to start the search from.

    Returns:
        list: A list of absolute paths to all files found.
    """
    file_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            file_paths.append(full_path)
    return file_paths

def refactor_tglf_file(filepath: str, prefixes_to_remove: List[str], to_replace: dict, to_add: dict):
    """
    Reads TGLF file, removes any lines that start with a string found in
    the prefixes_to_remove list, replaces entries with keys in the to_replace_list, 
    and writes the modified content back to the original file path.

    Args:
        filepath (str): The path to the file to be processed.
        prefixes_to_remove (List[str]): A list of string prefixes. Any line
                                        starting with one of these prefixes
                                        (not case-sensitive) will be removed.
        to_replace (dict): A dictionary containing prefixes to be replaced as keys and values with the replacement
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at path: {filepath}")
        return

    filtered_lines = []

    try:
        # 1. Read all lines from the file
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # 2. Filter the lines
        for line in lines:
            # Strip leading/trailing whitespace (including newline) for accurate prefix checking
            # but keep the original line to write back, preserving its newline character.
            stripped_line = line.lstrip().upper()

            # Check if the stripped line starts with any of the prefixes
            should_remove = any(stripped_line.startswith(prefix.upper()) for prefix in prefixes_to_remove)
            r_key= None
            r_val = None
            for prefix in to_replace.keys():
                should_replace = stripped_line.startswith(prefix.upper())
                if should_replace:
                    r_key = prefix
                    r_val = to_replace[prefix]
                    break

            if not should_remove and not should_replace:
                filtered_lines.append(line)
            
            elif should_replace:
                filtered_lines.append(r_key + " = " + str(r_val) + '\n')
        
        for key in to_add.keys():
            filtered_lines.append(key + " = " + str(to_add[key]) + '\n')
        # 3. Write the filtered content back to the original file (overwriting it)
        with open(filepath, 'w') as f:
            f.writelines(filtered_lines)

    except IOError as e:
        print(f"An error occurred while reading or writing the file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_dir")
    parser.add_argument("-d", "--destination_dir")

    args = parser.parse_args()
    source = args.source_dir
    dest = args.destination_dir

    os.makedirs(dest, exist_ok=True)

    # Source directory structure is expected to be as follows:
    # source/
    #   - batch-000/
    #       - cgyro/
    #       - tglf/
    #           - input-000/
    #               - input.tglf
    #           - input-001/ 
    #               - input.tglf
    #           - ... (per ky)
    #    - batch-001/
    #    - ... (per physical condition)

    # Destination directory structure will be as follows on success:
    # dest/
    #   - batch-000/
    #       - input.tglf
    #   - batch-001/
    #       - input.tglf
    #   - ... (per physical condition)

    # You can then run "bash run_simulation.sh tglf dest" and this SHOULD work

    batch_dirs = get_child_directories_at_depth_1(source)
    for batch_dir in batch_dirs:
        batch_name = batch_dir.split('/')[-1]
        tglf_batch_dir = os.path.join(source, os.path.join(batch_dir, 'tglf'))
        try:
            files = get_all_files_recursively(tglf_batch_dir)
            # Copy the file
            dest_dir = os.path.join(dest, batch_name)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, 'input.tglf')
            shutil.copy(files[0], dest_path)
            refactor_tglf_file(dest_path, TGLF_KEYS_FOR_REMOVAL, TGLF_KEYS_TO_REPLACE, TGLF_KEYS_TO_ADD)
            print(f'Refactored TGLF file at {dest_path}')
        except FileExistsError:
            print(f"Error: Destination directory '{dest}' already exists.")
        except Exception as e:
            print(f"An error occurred: {e}")
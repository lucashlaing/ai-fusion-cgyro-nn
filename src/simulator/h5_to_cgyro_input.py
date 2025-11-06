import os
import h5py
from datetime import datetime
from pyrokinetics import Pyro
import numpy as np
import textwrap
import tempfile

import os
import h5py
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm

input_keys = [
    "RLTS_3", "KAPPA_LOC", "ZETA_LOC", "TAUS_3", "VPAR_1", "Q_LOC", "RLNS_1",
    "TAUS_2", "Q_PRIME_LOC", "P_PRIME_LOC", "ZMAJ_LOC", "VPAR_SHEAR_1",
    "RLTS_2", "S_DELTA_LOC", "RLTS_1", "RMIN_LOC", "DRMAJDX_LOC", "AS_3",
    "RLNS_3", "DZMAJDX_LOC", "DELTA_LOC", "S_KAPPA_LOC", "ZEFF", "VEXB_SHEAR",
    "RMAJ_LOC", "AS_2", "RLNS_2", "S_ZETA_LOC", "BETAE_log10", "XNUE_log10", "DEBYE_log10"
]
log_ops_keys = {"BETAE_log10", "XNUE_log10", "DEBYE_log10"}

FIXED_TRAILER_TEMPLATE = textwrap.dedent("""\
    GEOMETRY_FLAG = 1
    SIGN_BT=-1.00000E+00
    SIGN_IT=+1.00000E+00

    #----------Additional Parameters----------
    # Species
    NS=3
    N_MODES=5
    # Questionable forced defaults:
    DRMINDX_LOC=1.0
    NKY=1
    USE_BPER=True
    USE_BPAR=True
    USE_AVE_ION_GRID=True
    USE_MHD_RULE=False
    ALPHA_ZF=-1
    KYGRID_MODEL=0
    KY={ky_val}
    SAT_RULE=2
    NBASIS_MAX=6
    UNITS=CGYRO
    VPAR_2 = 0.0
    VPAR_3 = 0.0
    BT_EXP=1.0
    VPAR_SHEAR_2 = 0.0
    VPAR_SHEAR_3 = 0.0

    #Confirmed with Tom 7/19
    AS_1=+1.0
    TAUS_1=+1.0
    MASS_1=0.0002723125672605524
    ZS_1=-1
    MASS_2=+1.0
    ZS_2=1
    MASS_3=+6.0
    ZS_3=6.0

    #----------Appended Missing Constants----------
    NN_MAX_ERROR = -1.0
    THETA0_SA = 0.0
    NXGRID = 16
    XWELL_SA = 0.0
    VPAR_SHEAR_MODEL = 1
    VNS_SHEAR_1 = 0.0
    VNS_SHEAR_2 = 0.0
    SHAT_SA = 1.0
    RLNP_CUTOFF = 18.0
    WIDTH_MIN = 0.3
    NBASIS_MIN = 2
    FT_MODEL_SA = 1
    WIDTH = 1.65
    VEXB = 0.0
    VPAR_MODEL = 0
    VTS_SHEAR_3 = 0.0
    FIND_WIDTH = True
    FILTER = 2.0
    DAMP_SIG = 0.0
    LINSKER_FACTOR = 0.0
    ALPHA_QUENCH = 0
    WRITE_WAVEFUNCTION_FLAG = 0
    THETA_TRAPPED = 0.7
    GHAT = 1.0
    VTS_SHEAR_2 = 0.0
    WD_ZERO = 0.1
    rho_e = 0.01650189586867377
    NEW_EIKONAL = True
    ADIABATIC_ELEC = False
    DEBYE_FACTOR = 1.0
    VTS_SHEAR_1 = 0.0
    B_unit = 1.0
    ALPHA_SA = 0.0
    GRADB_FACTOR = 0.0
    RMIN_SA = 0.5
    ALPHA_MACH = 0.0
    NWIDTH = 21
    SAT_geo0_out = 1.0
    GCHAT = 1.0
    RMAJ_SA = 3.0
    IBRANCH = -1
    USE_TRANSPORT_MODEL = True
    VNS_SHEAR_3 = 0.0
    WDIA_TRAPPED = 0.0
    KX0_LOC = 0.0
    Q_SA = 2.0
    B_MODEL_SA = 1
    ETG_FACTOR = 1.25
    IFLUX = True
    PARK = 1.0
    ALPHA_E = 1.0
    ALPHA_P = 1.0
    XNU_FACTOR = 1.0
""")

import re

CGYRO_CONSTANTS = {
    'N_ENERGY': 8,
    'N_XI': 24,
    'N_THETA': 24,
    'N_RADIAL': 16,
    'N_TOROIDAL': 1,
    'NONLINEAR_FLAG': 0,
    'BOX_SIZE': 1,
    'DELTA_T': 0.005,   # will be overwritten based on KY
    'MAX_TIME': 1000.0,
    'DELTA_T_METHOD': 1,
    'PRINT_STEP': 100,  # will be overwritten based on KY
    'THETA_PLOT': 1,
}


def load_cgyro_file_as_dict(filepath):
    """Parse input.cgyro into a dictionary."""
    config = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.match(r"(\w+)\s*=\s*([^\n#]+)", line)
            if match:
                key, val = match.groups()
                try:
                    config[key] = float(val)
                except ValueError:
                    config[key] = val  # In case of non-numeric constants
    return config


def write_cgyro_file_from_dict(config, filepath):
    """Write dictionary back to input.cgyro file."""
    with open(filepath, "w") as f:
        for key, val in config.items():
            if isinstance(val, float):
                f.write(f"{key} = {val:.8g}\n")
            else:
                f.write(f"{key} = {val}\n")


def rotate_species_in_dict(config):
    """Rotate _1 → _3, _2 → _1, _3 → _2"""
    keys_by_index = {1: {}, 2: {}, 3: {}}
    pattern = re.compile(r"(.*)_([123])$")

    for key in list(config.keys()):
        match = pattern.match(key)
        if match:
            base, idx = match.groups()
            idx = int(idx)
            keys_by_index[idx][base] = config.pop(key)

    # Apply rotation
    for new_idx, old_idx in [(1, 2), (2, 3), (3, 1)]:
        for base, val in keys_by_index[old_idx].items():
            config[f"{base}_{new_idx}"] = val

    return config


def apply_cgyro_constants(config, ky_val):
    """Update with CGYRO_CONSTANTS + fine-tune DELTA_T/PRINT_STEP"""
    config.update(CGYRO_CONSTANTS)

    if 1 < ky_val <= 10:
        config["DELTA_T"] = 0.001
        config["PRINT_STEP"] = 250
    elif ky_val > 10:
        config["DELTA_T"] = 0.0005
        config["PRINT_STEP"] = 500

    return config


def process_cgyro_file(filepath):
    config = load_cgyro_file_as_dict(filepath)

    ky_val = config.get("KY", None)
    if ky_val is None:
        raise ValueError(f"No KY found in {filepath}")

    config = rotate_species_in_dict(config)
    config = apply_cgyro_constants(config, ky_val)

    write_cgyro_file_from_dict(config, filepath)


def write_input_tglf(f, sample_idx, ky_idx, out_path):
    with open(out_path, "w") as f_out:
        f_out.write("# Geometry (Miller) and Parameters\n")
        for key in input_keys:
            val = f[key][sample_idx]
            if key in log_ops_keys:
                val = 10 ** val
                key_out = key.replace("_log10", "")
            else:
                key_out = key
            f_out.write(f"{key_out}={val:+.5E}\n")

        ky_val = f["ky"][sample_idx, ky_idx]
        trailer = FIXED_TRAILER_TEMPLATE.format(ky_val=f"{ky_val:+.5E}")
        f_out.write("\n" + trailer + "\n")



def generate_tglf_and_cgyro(f, sample_idx, ky_idx, tglf_out_path, cgyro_out_path):
    # Step 1: Create a temp .tglf file just to load into Pyro
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".tglf") as tmp:
        tmp.write("# Geometry (Miller) and Parameters\n")
        for key in input_keys:
            val = f[key][sample_idx]
            if key in log_ops_keys:
                val = 10 ** val
                key_out = key.replace("_log10", "")
            else:
                key_out = key
            tmp.write(f"{key_out}={val:+.5E}\n")

        ky_val = f["ky"][sample_idx, ky_idx]
        trailer = FIXED_TRAILER_TEMPLATE.format(ky_val=f"{ky_val:+.5E}")
        tmp.write("\n" + trailer + "\n")

        tmp_path = tmp.name  # Save path before file closes

    # Step 2: Load temp into Pyro and enforce quasineutrality
    pyro = Pyro(gk_file=tmp_path)
    pyro.local_species.enforce_quasineutrality("ion1")

    # Step 3: Write TGLF and CGYRO from Pyro
    pyro.write_gk_file(tglf_out_path, gk_code="TGLF", enforce_quasineutrality=True)
    pyro.write_gk_file(cgyro_out_path, gk_code="CGYRO", enforce_quasineutrality=True)

    # Step 4: Postprocess CGYRO file
    process_cgyro_file(cgyro_out_path)

    # Optional: Delete temp file
    os.remove(tmp_path)



def convert_h5_to_batch_dir(h5_path, out_root="all_batches"):
    os.makedirs(out_root, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        n_samples, n_ky = f["ky"].shape

        for sample_idx in range(n_samples):
            batch_name = f"batch-{sample_idx:03d}"
            batch_dir = os.path.join(out_root, batch_name)
            tglf_base = os.path.join(batch_dir, "tglf")
            cgyro_base = os.path.join(batch_dir, "cgyro")

            os.makedirs(tglf_base, exist_ok=True)
            os.makedirs(cgyro_base, exist_ok=True)

            for ky_idx in range(n_ky):
                input_name = f"input-{ky_idx:03d}"
                tglf_dir = os.path.join(tglf_base, input_name)
                cgyro_dir = os.path.join(cgyro_base, input_name)

                os.makedirs(tglf_dir, exist_ok=True)
                os.makedirs(cgyro_dir, exist_ok=True)

                tglf_out_path = os.path.join(tglf_dir, "input.tglf")
                cgyro_out_path = os.path.join(cgyro_dir, "input.cgyro")

                generate_tglf_and_cgyro(f, sample_idx, ky_idx, tglf_out_path, cgyro_out_path)

    print(f"✅ Done. TGLF/CGYRO inputs written to subdirs in: {out_root}")



# --- add/replace below in your script ---


# (keep your previous imports and functions: Pyro, input_keys, log_ops_keys,
# FIXED_TRAILER_TEMPLATE, CGYRO_CONSTANTS, load_cgyro_file_as_dict, write_cgyro_file_from_dict,
# rotate_species_in_dict, apply_cgyro_constants, process_cgyro_file, write_input_tglf,
# generate_tglf_and_cgyro)


def _worker_task(h5_path, sample_idx, ky_idx, out_root):
    """
    A single unit of work: generate one pair of input.tglf and input.cgyro
    for (sample_idx, ky_idx). Runs in its own process.
    """
    try:
        # open HDF5 read-only inside this process
        with h5py.File(h5_path, "r") as f:
            batch_name = f"batch-{sample_idx:03d}"
            batch_dir = os.path.join(out_root, batch_name)
            tglf_base = os.path.join(batch_dir, "tglf")
            cgyro_base = os.path.join(batch_dir, "cgyro")

            input_name = f"input-{ky_idx:03d}"
            tglf_dir = os.path.join(tglf_base, input_name)
            cgyro_dir = os.path.join(cgyro_base, input_name)

            os.makedirs(tglf_dir, exist_ok=True)
            os.makedirs(cgyro_dir, exist_ok=True)

            tglf_out_path = os.path.join(tglf_dir, "input.tglf")
            cgyro_out_path = os.path.join(cgyro_dir, "input.cgyro")

            generate_tglf_and_cgyro(f, sample_idx, ky_idx, tglf_out_path, cgyro_out_path)

        # success
        return (sample_idx, ky_idx, None)
    except Exception as e:
        # return error info to the parent so we can report it
        return (sample_idx, ky_idx, repr(e))


def convert_h5_to_batch_dir_parallel(
    h5_path,
    out_root="all_batches",
    max_workers=None,
    chunksize=1
):
    """
    Parallel version. One (sample_idx, ky_idx) per task.
    - max_workers: defaults to os.cpu_count() - 1 (or 1 if that would be 0)
    - chunksize: task submission chunk size (fine to leave at 1)
    """
    os.makedirs(out_root, exist_ok=True)

    # probe dimensions
    with h5py.File(h5_path, "r") as f:
        n_samples, n_ky = f["ky"].shape

    # sensible default for CPU count
    if max_workers is None:
        cpu = os.cpu_count() or 2
        max_workers = max(1, cpu - 1)

    tasks = [(s, k) for s in range(n_samples) for k in range(n_ky)]
    total = len(tasks)

    worker = partial(_worker_task, h5_path, out_root=out_root)

    errors = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, s, k) for (s, k) in tasks]

        for fut in tqdm(as_completed(futures), total=total, desc="Writing TGLF/CGYRO"):
            s, k, err = fut.result()
            if err is not None:
                errors.append((s, k, err))

    if errors:
        print("⚠️ Some tasks failed:")
        for s, k, err in errors[:20]:  # cap output
            print(f"  sample {s}, ky {k}: {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
    else:
        print(f"✅ Done. All {total} inputs written under: {out_root}")


# === Run (parallel) ===
if __name__ == "__main__":
    h5_file = "/Users/wesleyliu/Documents/Github/gacode-docker/tglf_data_sampled_42_50.h5"  # <- your path
    output_dir = "./cgyro_inputs"
    convert_h5_to_batch_dir_parallel(h5_file, output_dir)


# # === Run ===
# h5_file = "/Users/wesleyliu/Documents/Github/gacode-docker/out_52_300_minmax_norm.h5"  # Replace with your actual file path
# output_dir = "./cgyro_inputs"
# convert_h5_to_batch_dir(h5_file, output_dir)

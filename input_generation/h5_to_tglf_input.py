import os
import h5py
import numpy as np
import textwrap



input_keys = [
    "RLTS_3", "KAPPA_LOC", "ZETA_LOC", "TAUS_3", "VPAR_1", "Q_LOC", "RLNS_1",
    "TAUS_2", "Q_PRIME_LOC", "P_PRIME_LOC", "ZMAJ_LOC", "VPAR_SHEAR_1",
    "RLTS_2", "S_DELTA_LOC", "RLTS_1", "RMIN_LOC", "DRMAJDX_LOC", "AS_3",
    "RLNS_3", "DZMAJDX_LOC", "DELTA_LOC", "S_KAPPA_LOC", "ZEFF", "VEXB_SHEAR",
    "RMAJ_LOC", "AS_2", "RLNS_2", "S_ZETA_LOC", "BETAE_log10", "XNUE_log10", "DEBYE_log10"
]
log_ops_keys = {"BETAE_log10", "XNUE_log10", "DEBYE_log10"}

FIXED_TRAILER = textwrap.dedent("""\
    GEOMETRY_FLAG = 1
    SIGN_BT=-1.00000E+00
    SIGN_IT=+1.00000E+00

    #----------Additional Parameters----------
    NS=3
    N_MODES=5
    DRMINDX_LOC=1.0
    NKY=12
    USE_BPER=True
    USE_BPAR=True
    USE_AVE_ION_GRID=True
    USE_MHD_RULE=False
    ALPHA_ZF=-1
    KYGRID_MODEL=4
    KY=+3.00000E-01
    SAT_RULE=3
    NBASIS_MAX=6
    UNITS=CGYRO
    VPAR_2=0.0
    VPAR_3=0.0
    BT_EXP=1.0
    VPAR_SHEAR_2=0.0
    VPAR_SHEAR_3=0.0
    AS_1=+1.0
    TAUS_1=+1.0
    MASS_1=0.0002723125672605524
    ZS_1=-1
    MASS_2=+1.0
    ZS_2=1
    MASS_3=+6.0
    ZS_3=6.0
""")

def write_tglf_file(f, sample_idx, input_dir):
    os.makedirs(input_dir, exist_ok=True)
    out_path = os.path.join(input_dir, "input.tglf")

    with open(out_path, "w") as f_out:
        f_out.write("# Geometry (Miller) and Parameters\n")
        for key in input_keys:
            val = f[key][sample_idx]
            if key in log_ops_keys:
                val = 10 ** val
                key = key.replace("_log10", "")
            f_out.write(f"{key}={val:+.5E}\n")

        f_out.write("\n" + FIXED_TRAILER + "\n")

    print(f"✅ Written {out_path}")

def generate_all_tglf_files_batched(h5_path, out_root, batch_size=100):
    os.makedirs(out_root, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        n_samples = f[input_keys[0]].shape[0]

        for sample_idx in range(n_samples):
            batch_id = sample_idx // batch_size
            input_id = sample_idx % batch_size
            batch_dir = os.path.join(out_root, f"batch-{batch_id:03d}", "tglf", f"input-{input_id:03d}")
            write_tglf_file(f, sample_idx, batch_dir)

# === Run ===
if __name__ == "__main__":
    # === Config ===
    H5_PATH = "input_generation/sampled_output_file.h5"  # Replace with your HDF5 file
    OUTPUT_ROOT = "tglf_inputs_batched"
    BATCH_SIZE = 1  # Change as needed
    generate_all_tglf_files_batched(H5_PATH, OUTPUT_ROOT, BATCH_SIZE)

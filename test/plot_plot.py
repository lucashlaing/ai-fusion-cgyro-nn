import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- USER CONFIGURATION PANEL ---
RUNS_TO_PLOT = {
    "20260316-191511_BAL_res_uni_ran": "Deviation-Uniform-Random",
    "20260316-192244_BAL_strat_uni_ran": "UniBin-Uniform-Random",
    "20260316-192842_BAL_gaussian": "Deviation-Weighted-Gaussian",
    "20260316-193003_BAL_direct": "Uniform-Uniform-Direct",
    "20260316-193541_BAL_eig_stratified": "UniBin-Weighted-EIG",
    "20260316-193542_BAL_strat_uni_gaus": "Uniform-Uniform-Gaussian",
    "20260316-193851_BAL_eig": "Uniform-Uniform-EIG",
    "20260316-194249_BAL_random": "Uniform-Uniform-Random",   
}

# Configuration for the starting point (Baseline)
INITIAL_LOSS = 0.058932207640497454
INITIAL_SAMPLES = 20000
# --------------------------------

df_loss = pd.read_csv(r'test\redone-test-loss.csv')
df_samples = pd.read_csv(r'test\redone-num-samples.csv')

df_loss.columns = df_loss.columns.str.replace('"', '').str.strip()
df_samples.columns = df_samples.columns.str.replace('"', '').str.strip()

df = pd.merge(df_samples, df_loss, on='BAL/iteration')

plt.figure(figsize=(12, 7))
found_any = False

for run_id, label in RUNS_TO_PLOT.items():
    s_col = f"{run_id} - BAL/num_samples"
    l_col = f"{run_id} - BAL/test_loss"
    
    if s_col in df.columns and l_col in df.columns:
        # Get data, drop NaNs, and ensure it's sorted by iteration
        run_data = df[['BAL/iteration', s_col, l_col]].dropna().sort_values('BAL/iteration')
        
        if not run_data.empty:
            # 1. Get the loss values
            y_vals = pd.to_numeric(run_data[l_col]).values
            
            # 2. Get the sample increments (the number added at each step)
            increments = pd.to_numeric(run_data[s_col]).values
            
            # 3. Calculate Cumulative Sum: 
            # x_coords = [20k + iter0_samples, 20k + iter0 + iter1, ...]
            x_coords = INITIAL_SAMPLES + increments.cumsum()
            
            # --- NEW: Prepend the initial baseline point ---
            x_coords = np.insert(x_coords, 0, INITIAL_SAMPLES)
            y_vals = np.insert(y_vals, 0, INITIAL_LOSS)
            # -----------------------------------------------
            
            plt.plot(x_coords, y_vals, label=label, marker='o', markersize=4)
            found_any = True
    else:
        print(f"{s_col} not found in data")

if found_any:
    plt.xlabel('Total Cumulative Samples (Starting from 20k Baseline)')
    plt.ylabel('Test Loss')
    plt.title('Different Acq Functions in Active Learning')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.axhline(y= 0.009)
    plt.legend()
    plt.tight_layout()
    output_path = 'weight_decay_comparison.png'
    plt.savefig(output_path, dpi=300)
    plt.show()
else:
    print("Check column names: No data matched.")
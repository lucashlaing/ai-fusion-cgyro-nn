import pandas as pd
import matplotlib.pyplot as plt

# --- USER CONFIGURATION PANEL ---
RUNS_TO_PLOT = {
    "20260217-224946_BAL_random_end_lr-1e-5_peak_lr-1e-3": "end_lr-1e-5_peak_lr-1e-3",
    "20260217-002855_BAL_random-end_lr-1e-6_peak_lr-5e-4": "end_lr-1e-6_peak_lr-5e-4",
    "20260216-220903_BAL_random-end-lr-1e-6": "end_lr-1e-6_peak_lr-1e-5",
    "20260209-013107_BAL_random": "end_lr-5e-7_peak_lr-1e-5",
}
# --------------------------------

df_loss = pd.read_csv(r'test\lr-test-loss.csv')
df_samples = pd.read_csv(r'test\lr-num-samples.csv')

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
            # We start at 20,000 and add the samples logged at each iteration
            # x_coords = [20k + iter0_samples, 20k + iter0 + iter1, ...]
            x_coords = 20000 + increments.cumsum()
            
            # Note: If your first logged 'num_samples' is the full 30k (baseline + first batch), 
            # use the previous logic but ensure we force floats:
            # x_coords = 20000 + (increments - increments[0])
            
            plt.plot(x_coords, y_vals, label=label, marker='o', markersize=4)
            found_any = True

if found_any:
    plt.xlabel('Total Cumulative Samples (Starting from 20k Baseline)')
    plt.ylabel('Test Loss')
    plt.title('Learning Rates in Active Learning')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    output_path = 'weight_decay_comparison.png'
    plt.savefig(output_path, dpi=300)
    plt.show()
else:
    print("Check column names: No data matched.")
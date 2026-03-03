import os
import h5py

def analyze_h5_folders(root_path, target_key):
    """
    Crawls 'train' and 'test' folders to compare file sizes and entry counts.
    """
    results = {}

    # Define the subfolders we are looking for
    subfolders = ['train', 'test']
    print("tglf-sinn-data")
    print(f"{'Folder':<10} | {'Files':<10} | {'Total Size (MB)':<15} | {'Total Entries':<15}")
    print("-" * 60)

    for folder in subfolders:
        folder_path = os.path.join(root_path, folder)
        
        if not os.path.exists(folder_path):
            print(f"Directory '{folder}' not found in {root_path}")
            continue

        total_size_bytes = 0
        total_entries = 0
        file_count = 0

        # Loop through all files in the subfolder
        for filename in os.listdir(folder_path):
            if filename.endswith('.h5'):
                file_path = os.path.join(folder_path, filename)
                
                # 1. Get File Size
                total_size_bytes += os.path.getsize(file_path)
                
                # 2. Get Entry Count
                try:
                    with h5py.File(file_path, 'r') as f:
                        if target_key in f:
                            # Assumes entries are the first dimension
                            total_entries += f[target_key].shape[0]
                            file_count += 1
                        else:
                            print(f"Warning: Key '{target_key}' not found in {filename}")
                except Exception as e:
                    print(f"Error reading {filename}: {e}")

        # Convert bytes to MB for readability
        total_size_mb = total_size_bytes / (1024 * 1024)
        
        results[folder] = {
            'size_mb': total_size_mb,
            'entries': total_entries,
            'count': file_count
        }

        print(f"{folder:<10} | {file_count:<10} | {total_size_mb:<15.2f} | {total_entries:<15,}")

    return results

# --- CONFIGURATION ---
# Replace 'path_to_your_folder' with the actual path to "some anem"
# Replace 'your_dataset_key' with the specific key (e.g., 'data' or 'labels')
path_to_data = '../../../data/lucas_work/tglf-sinn-data' 
key_to_check = 'RLTS_3' 

if __name__ == "__main__":
    analyze_h5_folders(path_to_data, key_to_check)
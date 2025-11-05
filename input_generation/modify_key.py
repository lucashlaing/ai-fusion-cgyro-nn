import os

def update_key_value_line(line, key, new_value):
    stripped = line.strip()

    # Format: key = value
    if stripped.startswith(f"{key}"):
        parts = stripped.split("=")
        if len(parts) == 2 and parts[1].strip() != str(new_value):
            return f"{key} = {new_value}\n", True
        return line, False

    # Format: value  key
    elif key in stripped and stripped.endswith(key):
        parts = stripped.split()
        if len(parts) == 2 and parts[0] != str(new_value):
            return f"{new_value}  {key}\n", True
        return line, False

    return line, False

def update_flags_in_file(file_path, keys_to_update):
    updated_lines = []
    changed = False
    keys_found = set()

    with open(file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        modified_line = line
        for key, new_value in keys_to_update.items():
            modified_line, was_changed = update_key_value_line(modified_line, key, new_value)
            if was_changed:
                changed = True
                keys_found.add(key)
                break  # Only modify once per line
        updated_lines.append(modified_line)

    # # Add missing keys at the end
    # for key, new_value in keys_to_update.items():
    #     if key not in keys_found:
    #         updated_lines.append(f"{key} = {new_value}\n")
    #         changed = True
    #         print(f"➕ Appended missing key: {key} = {new_value} in {file_path}")

    if changed:
        with open(file_path, "w") as f:
            f.writelines(updated_lines)
        print(f"✅ Updated: {file_path}")
    else:
        print(f"⏭️  Skipped (already correct): {file_path}")


def walk_and_update_inputs(base_dir, filenames, keys):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file in filenames:
                update_flags_in_file(os.path.join(root, file), keys)




if __name__ == "__main__":
    root_directory = "/Users/wesleyliu/Documents/Github/gacode-docker/cgyro_inputs"  # Now apply to all subdirs
    keys_to_update = {
        "NONLINEAR_FLAG": 0,
    }
    filenames = {"input.cgyro", "input.cgyro.gen"}

    for subdir in sorted(os.listdir(root_directory)):
        subdir_path = os.path.join(root_directory, subdir)
        if os.path.isdir(subdir_path):
            walk_and_update_inputs(subdir_path, filenames, keys_to_update)


import subprocess
import os

# cloud related helper functions

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "https://s3-west.nrp-nautilus.io")

def upload_to_s3(prefix, local_path):
    """
    Upload a file or all files in a directory to S3 bucket using s5cmd.

    Parameters:
    - bucket (str): The S3 bucket name
    - prefix (str): The prefix/folder inside the bucket
    - local_dir (str): The single fizle or local directory containing files to upload
    """
    s3_destination = f"s3://ai-fusion-ga/{prefix}"
    if os.path.isdir(local_path):
        # Upload all files in the directory
        if not local_path.endswith('/'):
            local_path += '/'
        command = f's5cmd --endpoint-url "{S3_ENDPOINT_URL}" cp "{local_path}*" "{s3_destination}/"'
    else:
        # Upload a single file (include filename in destination)
        filename = os.path.basename(local_path)
        s3_file_destination = s3_destination.rstrip("/") + f"/{filename}"
        command = f's5cmd --endpoint-url "{S3_ENDPOINT_URL}" cp "{local_path}" "{s3_file_destination}"'

    print(f"Uploading from {local_path} to {s3_destination}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(result.stdout)
    
    # Check for errors
    if result.returncode != 0:
        print(f"Error during upload: {result.stderr}")
        raise Exception(f"Failed to upload to S3: {result.stderr}")
    
    return result.returncode == 0

def download_from_s3(prefix, target_path):
    """
    Download files from S3 bucket to a local file or directory using s5cmd.

    Parameters:
    - prefix (str): The prefix/folder inside the bucket
    - target_path (str): The local file path or directory to download files to
    """
    # Check if target_path is a directory or file
    if os.path.isdir(target_path) or target_path.endswith('/'):
        # Ensure target_path ends with a slash for directory downloads
        if not target_path.endswith('/'):
            target_path += '/'
        if not prefix.endswith('/'):
            prefix += '/'
        s3_source = f"s3://ai-fusion-ga/ersp_res/{prefix}*"
        command = f's5cmd --endpoint-url "{S3_ENDPOINT_URL}" cp "{s3_source}" "{target_path}"'
    else:
        # Download a single file
        s3_source = f"s3://ai-fusion-ga/ersp_res/{prefix}"
        command = f's5cmd --endpoint-url "{S3_ENDPOINT_URL}" cp "{s3_source}" "{target_path}"'

    # Run the command
    print(f"Downloading from {s3_source} to {target_path}...")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Print output for logging
    print(result.stdout)
    print(result.stderr)

    # Check for errors
    if result.returncode != 0:
        raise Exception(f"Failed to download from {s3_source} to {target_path}: {result.stderr}")

    return result.returncode == 0
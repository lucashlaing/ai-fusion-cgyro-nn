#!/bin/bash

# Exit on error
set -e

# Step 1: Create or update conda environment
echo "📦 Creating or updating conda environment from environment.yml..."
conda env create -f environment.yml || conda env update -f environment.yml

# Extract env name
ENV_NAME=$(awk '/name:/ {print $2}' environment.yml)

# Step 2: Activate the conda environment
echo "🔁 Activating conda environment: $ENV_NAME"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "📦 Installing dependencies using Poetry..."
poetry install

#Pip install as the poetry install is broken due to their dependancy issues
conda install -c conda-forge netcdf4 h5py
pip install pyrokinetics

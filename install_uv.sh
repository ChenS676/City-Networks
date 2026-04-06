# 1. Create and activate a virtual environment
uv venv .venv --python 3.11
source .venv/bin/activate

# 2. PyTorch (CUDA 12.4 wheel)
uv pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
  --index-url https://download.pytorch.org/whl/cu124

# 3. PyG core
uv pip install torch_geometric

# 4. PyG C++ extensions (pyg_lib, scatter, sparse, cluster, spline)
uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  --find-links https://data.pyg.org/whl/torch-2.5.0+cu124.html

# 5. Remaining packages
uv pip install networkx osmnx publib scikit-learn pyyaml tqdm

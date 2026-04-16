uv run python - <<'EOF'
import torch, torch_geometric, torch_scatter, torch_cluster
print("torch     :", torch.__version__)
print("cuda ok   :", torch.cuda.is_available())
print("pyg       :", torch_geometric.__version__)
print("scatter   :", torch_scatter.__version__)
print("cluster   :", torch_cluster.__version__)
import graphbench; print("graphbench: ok")
from muon import SingleDeviceMuonWithAuxAdam; print("muon      : ok")
EOF

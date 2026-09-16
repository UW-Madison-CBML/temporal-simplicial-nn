#!/usr/bin/env bash
# Install PyG scatter/sparse wheels (not available on default PyPI).
# Usage: bash scripts/install_pyg_extensions.sh [cpu|cu121]

set -euo pipefail

MODE="${1:-cu121}"

TORCH_VERSION="$(python -c 'import torch; v=torch.__version__.split("+")[0]; print(v)')"
MAJOR_MINOR="$(echo "$TORCH_VERSION" | cut -d. -f1,2)"

case "$MODE" in
  cpu)
    WHEEL_TAG="cpu"
    ;;
  cu121|cuda|gpu)
    WHEEL_TAG="cu121"
    ;;
  *)
    echo "Unknown mode: $MODE (use cpu or cu121)" >&2
    exit 1
    ;;
esac

URL="https://data.pyg.org/whl/torch-${MAJOR_MINOR}.0+${WHEEL_TAG}.html"
echo "Installing PyG extensions from ${URL}"

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f "$URL"

python -c "
import torch
import torch_geometric
import tgb
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('torch_geometric', torch_geometric.__version__)
print('tgb', tgb.__version__ if hasattr(tgb, '__version__') else 'ok')
"

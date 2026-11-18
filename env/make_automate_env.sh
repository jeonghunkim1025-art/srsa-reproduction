#!/bin/bash
# Env for training AutoMate specialists: clone the srsa env and replace rl_games with upstream 1.6.1
set -e
unset LD_PRELOAD
source $HOME/miniforge3/etc/profile.d/conda.sh
echo "[$(date)] clone srsa -> automate"
conda create -y -n automate --clone srsa
conda activate automate
pip uninstall -y rl_games
pip install rl-games==1.6.1
pip install psutil==5.9.8   # keep the version pinned by isaacsim-kernel
python - <<'PY'
import rl_games, os, inspect
from rl_games import torch_runner
src = inspect.getsource(torch_runner.Runner.run_train)
print("rl_games", rl_games.__file__)
print("run_train uses load_mode:", "load_mode" in src)
PY
python -c "import torch, isaaclab; print('torch', torch.__version__)"
echo "[$(date)] AUTOMATE_ENV_DONE"

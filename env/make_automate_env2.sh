#!/bin/bash
# Install the rl_games version required by Isaac Lab isaaclab_rl (isaac-sim fork, python3.11 branch) into the (already cloned) automate env
set -e
unset LD_PRELOAD
export PATH=$HOME/miniforge3/bin:$PATH   # git-lfs may be needed for pip installs from git
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate automate
pip uninstall -y rl_games 2>/dev/null || true
pip install "rl-games @ git+https://github.com/isaac-sim/rl_games.git@python3.11"
pip install psutil==5.9.8
python - <<'PY'
import inspect, rl_games
from rl_games import torch_runner
print("rl_games", rl_games.__file__)
print("run_train requires load_mode:", "load_mode" in inspect.getsource(torch_runner.Runner.run_train))
PY
python -c "import torch; print('torch', torch.__version__)"
conda activate srsa; python -c "import inspect; from rl_games import torch_runner; print('srsa env still SIL fork:', 'load_mode' in inspect.getsource(torch_runner.Runner.run_train))"
echo "[$(date)] AUTOMATE_ENV_DONE"

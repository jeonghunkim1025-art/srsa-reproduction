# 1. Software Environment and Installation Issues

## 1.1 Environment used [measured]

| Item | Value |
|---|---|
| GPU | NVIDIA L40 48 GB (driver 595, CUDA 13.2); NVIDIA A100-SXM4 40 GB × 4 (driver 580, CUDA 13.0) |
| OS | Ubuntu 24.04 (L40), Ubuntu 22.04 (A100) |
| Isaac Lab / Isaac Sim | tag `v2.3.0` / 5.1.0 (versions stated in the SRSA README) |
| SRSA | commit `2bed3f7` |
| conda env `srsa` (SRSA fine-tuning) | Python 3.11, isaaclab 0.47.2, torch 2.7.0+cu128, rl_games 1.6.1 (SRSA `rl_games_sil` fork) |
| conda env `automate` (AutoMate training from scratch and evaluation) | clone of `srsa` with the isaac-sim rl_games fork, torch 2.9.0+cu130, numba-cuda[cu13] 0.30.4, conda cuda-nvcc 12.8 |

- Full package lists: `env/*_pip_freeze.txt` and `env/*_conda_explicit.txt`.
- Environment creation scripts for `automate`: `env/make_automate_env*.sh`.
- The Isaac Lab and SRSA source code was not modified. The only exception is the `assembly_id` line of `assembly_tasks_cfg.py`, which `run_w_id.py` overwrites on every run by design.

## 1.2 Installation issues and resolutions [measured]

| Issue | Resolution |
|---|---|
| `isaaclab.sh` calls apt, which requires sudo (not available) | Installed cmake through conda |
| A full install pulled in a different PyTorch version | Installed only the required component (`--install rl_games`) |
| Build failure of `flatdict` | `setuptools<81` with `--no-build-isolation` |
| The VirtualGL `LD_PRELOAD` broke cuDNN loading | `unset LD_PRELOAD` in all run scripts |
| The SRSA rl_games fork is incompatible with Isaac Lab's `train.py` (AutoMate training) | Used two environments; the AutoMate environment uses the isaac-sim rl_games fork (`python3.11` branch) |
| **The AutoMate soft-DTW imitation reward runs on the CPU without any warning** | See Section 1.3 |

## 1.3 Soft-DTW silently running on the CPU

**Symptom.** The trajectory-imitation term of the AutoMate dense reward (soft-DTW) took the CPU path instead of the GPU path.
- One call took 18 ms, and training ran at 18 fps.
- Training otherwise proceeds without errors, so the problem is only visible when throughput is measured.

**Cause.**
- The GPU path relies on a numba CUDA kernel. In the original environment, nvcc was not installed and numba was not compatible with the CUDA 13 driver.
- With numba 0.59, the GPU kernel crashed with a segmentation fault during CUDA context creation. We attribute this to a mismatch between the driver bindings bundled with numba 0.59 and the CUDA 13 driver [inferred].

**Resolution.** We followed the CUDA 13 instructions in the Isaac Lab documentation:
1. Install `cuda-nvcc=12.8` from conda-forge.
2. Run `pip install numba-cuda[cu13] coverage==7.6.1`. Without pinning `coverage`, the error `coverage.types has no attribute 'Tracer'` occurs.
3. Install torch 2.9.0+cu130 and torchvision 0.24.0+cu130. This is the combination the Isaac Lab 2.3.0 guide lists for aarch64.

**Result.**
- One call takes 0.39 ms, and training runs at 76 fps.
- The GPU and CPU paths produce identical values.
- In an environment that must keep torch cu128, `numba-cuda[cu12]==0.30.4` together with `nvidia-cuda-nvcc-cu12==12.8.61` gives the same effect. If these versions are mismatched, an nvJitLink version error occurs.

**How to check.** If training throughput is in the tens of fps or lower, suspect this issue. Verify with `python -c "from numba import cuda; print(cuda.is_available())"`.

## 1.4 Environment check: success rate on the policy's own task (official check) [measured]

| Policy | Measured | Released reference value |
|---|---|---|
| Self-trained 00638 → 00638 (1024 episodes) | 0.980 | 0.970 |
| Released 00783 → 00783 (128 episodes) | 0.984 | 1.0 |
| Released 00426 → 00426 (128 episodes) | 1.000 | 1.0 |
| Released 00638 → 00638 (1024 episodes) | 0.403 | 0.970 |

- The self-trained policy matches the reference value. We therefore concluded that the evaluation pipeline itself is sound.
- Three of the five released checkpoints (00638, 00141, 00211) score well below the reference value even on their own tasks (0.403, 0.025 and 0.110 respectively). The cause is under investigation.

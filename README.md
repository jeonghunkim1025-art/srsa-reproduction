# Reproducing SRSA: Findings on Success Evaluation and Reward Design

This repository documents an independent reproduction of
[SRSA: Skill Retrieval and Adaptation for Robotic Assembly Tasks](https://arxiv.org/abs/2503.04538)
(ICLR 2025; code at [NVlabs/SRSA](https://github.com/NVlabs/SRSA)) using only the publicly released code, data and checkpoints.
It was carried out as an undergraduate capstone project (Department of Computer Science and Engineering, Kyung Hee University).

> **Status.** This repository contains only the parts of the work that have reached a conclusion.
> Skill retrieval and the comparison against the released zero-shot transfer table are still in progress and are not included.

## Summary of findings

| Topic | Finding | Document |
|---|---|---|
| Environment setup | The soft-DTW imitation reward of AutoMate can silently fall back to the CPU. Training then runs at about 18 fps. Matching numba-cuda and PyTorch to the installed CUDA version restores the GPU path, at about 76 fps. | [docs/01](docs/01_environment.md) |
| Official success check | In every task, the success region of `check_plug_inserted_in_socket` is a hemisphere of radius 15 mm centred on the goal position and capped at the disassembly distance d_dis. The document includes a proof and a numerical verification. | [docs/02](docs/02_official_success_check.md) |
| Strict insertion criterion | The strict criterion adds two per-task conditions to the official check: a lateral bound (clearance + 1 mm) and a depth bound (plug bottom below socket top). Across 257 evaluations, 54% of official successes do not meet the strict criterion. | [docs/03](docs/03_strict_criterion.md) |
| Reward design for fine-tuning | With the official sparse reward, fine-tuning reaches true insertion in only 3 of 12 source–target pairs. With the dense reward (unmodified paper code) or a strict sparse reward, 10 of 12 pairs reach true insertion faster than training from scratch. | [docs/04](docs/04_reward_finetuning.md) |

Every document distinguishes measured results (**[measured]**) from inferences that have not been verified (**[inferred]**).

## Repository layout

```
docs/   Analysis documents (see table above)
tools/
  strict_play.py        Modified Isaac Lab play.py. Records Δz, lateral distance, keypoint error and
                        success at every step to an .npz file. The success check itself is unchanged.
  strict_score.py       Computes official and strict success rates from one .npz file
  strict_geom.py        Computes per-task thresholds T (lateral) and D_ins (depth) from the part meshes
  verify_hemisphere.py  Numerical check that the mean keypoint error is at least the origin distance
  fig_judge_*.py        Figures of the success region
  srsa_strict/          Strict sparse-reward environment (subclass of SRSA AssemblySparseEnv)
                        and its training script
data/
  strict_geom.json/.md  Geometry values (T, D_ins, etc.) for all 100 tasks
  strict_results.tsv    Raw results of the 257 re-evaluations
env/    conda and pip package lists, environment creation scripts
figs/   Figures used in the documents
```

## Usage

```bash
# 1) Per-task thresholds (requires SRSA data/mesh, numpy, scipy)
SRSA_MESH=/path/to/SRSA/data/mesh python tools/strict_geom.py

# 2) Evaluate while recording poses (Isaac Lab v2.3.0 AutoMate environment; same arguments as play.py).
#    Set assembly_id in assembly_tasks_cfg.py beforehand, as run_w_id.py does.
python tools/strict_play.py --task=Isaac-AutoMate-Assembly-Direct-v0 --checkpoint=<ckpt.pth> \
    --num_envs=1024 --headless --out=runs/<name>

# 3) Official and strict success rates
python tools/strict_score.py runs/<name>.npz
```

## Experimental conditions

- **Software:** Isaac Lab v2.3.0, Isaac Sim 5.1, SRSA commit `2bed3f7`
- **Hardware:** NVIDIA L40 (48 GB) and NVIDIA A100 (40 GB)
- **Evaluation:** deterministic policy, 1024 parallel environments, 5 s episodes
- **Fine-tuning:** 128 environments × 200 epochs, one seed per pair. This is smaller than the paper's setting (256 environments × 1000 epochs, five seeds).

## License

Original code and documents in this repository are released under the [BSD 3-Clause License](LICENSE).
Files adapted from Isaac Lab and SRSA retain their original BSD 3-Clause copyright notices; see [NOTICE](NOTICE.md).

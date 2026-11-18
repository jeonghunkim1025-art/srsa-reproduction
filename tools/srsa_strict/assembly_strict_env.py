# [E3] Strict sparse reward: subclasses the original SRSA AssemblySparseEnv and only changes the per-step success passed to _update_rew_buf.
#   original: curr = check_plug_inserted_in_socket(...) (0 < Δz < disassembly_dist AND mean keypoint error < 15 mm)
#   strict:   curr_strict = curr AND lateral distance < T AND Δz < D_ins   (T = clearance c + 1 mm, D_ins = insertion depth; see data/strict_geom.json and docs/03_strict_criterion.md)
#   Δz and lateral distance are taken from held_pos − fixed_pos (env-origin frame, same as strict_play.py).
# Reward = ep_succeeded (strict; latched to 1 once true within the episode), the same form as the original sparse reward. SIL uses this reward as well.
# For learning curves, the episode success rate under the original check is also logged as extras["successes_orig"] (not used in the reward).
import json, os
import torch
from SRSA.tasks.direct.srsa.assembly_sparse_env import AssemblySparseEnv

GEO_PATH = os.environ.get("STRICT_GEOM_JSON", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "strict_geom.json"))
GEO = json.load(open(GEO_PATH))


class AssemblyStrictSparseEnv(AssemblySparseEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        g = GEO[str(self.cfg_task.assembly_id)]
        self.strict_thr = g["thr_mm"] * 1e-3
        self.strict_dins = g["d_ins_mm"] * 1e-3
        self.ep_succeeded_orig = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        print(f"STRICT_REWARD assembly_id={self.cfg_task.assembly_id} T={self.strict_thr*1e3:.3f}mm D_ins={self.strict_dins*1e3:.3f}mm")

    def _update_rew_buf(self, curr_successes):
        d = self.held_pos - self.fixed_pos
        lat = torch.linalg.norm(d[:, :2], dim=1)
        strict = torch.logical_and(curr_successes, torch.logical_and(lat < self.strict_thr, d[:, 2] < self.strict_dins))
        self.ep_succeeded_orig = torch.logical_or(self.ep_succeeded_orig, curr_successes.bool())
        if torch.any(self.reset_buf):
            self.extras["successes_orig"] = self.ep_succeeded_orig.float().mean()
            print("SuccessOrig", self.ep_succeeded_orig.float().mean().item())
        return super()._update_rew_buf(strict)

    def _reset_buffers(self, env_ids):
        super()._reset_buffers(env_ids)
        self.ep_succeeded_orig[env_ids] = False

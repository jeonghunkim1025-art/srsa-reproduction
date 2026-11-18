"""Compute the official and strict success rates from one npz file written by strict_play.py.

Usage: python tools/strict_score.py <run>.npz [--geom data/strict_geom.json]

- Official check (per step) = 0 < Δz < disassembly_dist AND mean keypoint error < close_error_thresh
  (same expression as Isaac Lab automate_algo_utils.check_plug_inserted_in_socket)
- Official success = ep_succeeded at the end of the episode (true if the check was true at any step)
- Strict success = at some step: official check AND lateral distance < T AND Δz < D_ins
  (T and D_ins are per-task values from data/strict_geom.json)
"""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("--geom", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "strict_geom.json"))
a = ap.parse_args()

d = np.load(a.npz, allow_pickle=True)
geo = json.load(open(a.geom))
succ = d["succ"]                      # (steps, envs) latched success as reported by the env
lat = d["lat"] * 1e3; dz = d["dz"] * 1e3  # mm
tid = str(d["assembly_id"]); g = geo[tid]
dist = float(d["disassembly_dist"][0]) * 1e3
cur = (dz > 0) & (dz < dist) & (d["kp"] < float(d["close_error_thresh"]))
orig = succ[-1].astype(bool)
strict = (cur & (lat < g["thr_mm"]) & (dz < g["d_ins_mm"])).any(0)
print(f"task {tid}  envs {orig.size}  T {g['thr_mm']:.2f} mm  D_ins {g['d_ins_mm']:.1f} mm  d_dis {dist:.1f} mm")
print(f"original success {orig.mean():.3f}   strict success {strict.mean():.3f}")
print(f"original-judge recompute mismatch {int((cur.any(0) != orig).sum())}")

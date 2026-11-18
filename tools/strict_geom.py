# Per-task geometry for the strict criterion. Input: SRSA/data/mesh/<id>/{plug,socket}.obj (md5-identical to the assets downloaded at run time).
# Assembled state: plug origin = socket origin, axis = z. The overlapping height range (excluding 1 mm at each end) is sliced every 0.5 mm.
# Clearance c = 10th percentile over heights of g(z), the minimum 2D distance between plug and socket surface points at height z (the tightest-fitting section; works for both hole- and pin-type tasks).
#   The tightest section is what constrains lateral motion of an inserted plug. The median is not used because sockets with a wide upper cavity (e.g. 00638, 00426) would yield the cavity width.
#   For reference, r_hole - r_plug based on inscribed radii (minimum distance to the axis) is also stored (negative and unusable for pin-type tasks).
# Strict lateral threshold T = c + 1 mm margin (margin for contact penetration and mesh approximation; an estimate). Output: data/strict_geom.json, .md
import os, json, numpy as np
from scipy.spatial import cKDTree
W = os.environ.get("SRSA_WORK", os.path.expanduser("~/srsa_work")); M = os.environ.get("SRSA_MESH", f"{W}/SRSA/data/mesh")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
def load(p):
    V, F = [], []
    for l in open(p):
        if l.startswith("v "): V.append([float(x) for x in l.split()[1:4]])
        elif l.startswith("f "): F.append([int(x.split("/")[0]) - 1 for x in l.split()[1:]])
    V = np.array(V); tri = []
    for f in F:
        for k in range(1, len(f) - 1): tri.append([f[0], f[k], f[k + 1]])
    return V, np.array(tri)
def sample(V, T, n=400000, rng=np.random.default_rng(0)):
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    ar = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1); idx = rng.choice(len(T), n, p=ar / ar.sum())
    u, v = rng.random(n), rng.random(n); m = u + v > 1; u[m], v[m] = 1 - u[m], 1 - v[m]
    return a[idx] + u[:, None] * (b[idx] - a[idx]) + v[:, None] * (c[idx] - a[idx])
out = {}
for tid in sorted(os.listdir(M)):
    try:
        Pv, Pt = load(f"{M}/{tid}/plug.obj"); Sv, St = load(f"{M}/{tid}/socket.obj")
    except FileNotFoundError: continue
    P, S = sample(Pv, Pt), sample(Sv, St)
    lo, hi = max(P[:, 2].min(), S[:, 2].min()) + 1e-3, min(P[:, 2].max(), S[:, 2].max()) - 1e-3
    rh, rp, gz = [], [], []
    for z in np.arange(lo, hi, 5e-4):
        ps = P[np.abs(P[:, 2] - z) < 2.5e-4]; ss = S[np.abs(S[:, 2] - z) < 2.5e-4]
        if len(ps) < 20 or len(ss) < 20: continue
        rp.append(np.hypot(ps[:, 0], ps[:, 1]).min()); rh.append(np.hypot(ss[:, 0], ss[:, 1]).min())
        gz.append(cKDTree(ss[:, :2]).query(ps[:, :2])[0].min())
    if not rh: out[tid] = None; continue
    r_h, r_p = float(np.median(rh)), float(np.median(rp)); c = float(np.median(gz))
    # Insertion depth D_ins: the Δz at which the plug bottom reaches the socket top near the axis (highest socket point within plug outer radius + 2 mm of the axis).
    #   Δz < D_ins means the plug bottom is below the socket top (i.e. engaged with the hole or pin).
    r_out = np.hypot(P[:, 0], P[:, 1]).max(); near = S[np.hypot(S[:, 0], S[:, 1]) <= r_out + 2e-3]
    top_local = float(near[:, 2].max()) if len(near) else float(S[:, 2].max())
    d_ins = (top_local - float(P[:, 2].min())) * 1e3
    out[tid] = dict(d_ins_mm=d_ins, socket_top_local_mm=top_local * 1e3, plug_min_z_mm=float(P[:, 2].min()) * 1e3, r_hole_mm=r_h * 1e3, r_plug_mm=r_p * 1e3, inscribed_diff_mm=(r_h - r_p) * 1e3, clearance_mm=float(np.percentile(gz, 10)) * 1e3, gap_p10_mm=float(np.percentile(gz, 10)) * 1e3, gap_p90_mm=float(np.percentile(gz, 90)) * 1e3, thr_mm=float(np.percentile(gz, 10)) * 1e3 + 1.0, gap_median_mm=c * 1e3,
                    overlap_mm=(hi - lo) * 1e3, n_slices=len(rh), socket_top_mm=float(S[:, 2].max() * 1e3),
                    plug_outer_mm=float(np.hypot(P[:, 0], P[:, 1]).max() * 1e3), socket_outer_mm=float(np.hypot(S[:, 0], S[:, 1]).max() * 1e3))
json.dump(out, open(os.path.join(OUT, "strict_geom.json"), "w"), indent=1)
L = ["# Per-task geometry and strict-criterion thresholds (`tools/strict_geom.py`)", "",
     "- In the assembled state (plug origin = socket origin), the overlapping height range (excluding 1 mm at each end) is sliced every 0.5 mm. 400k surface points are sampled per mesh.",
     "- Clearance c = 10th percentile of g(z), the minimum 2D distance between plug and socket surfaces at each height (the tightest-fitting section). Strict threshold T = c + 1 mm (margin for contact penetration and mesh approximation; an estimate).",
     "- Values in parentheses are the median to 90th percentile of g(z). For 00638 and 00426 the median is large because the socket has a wide upper cavity.",
     "- Insertion depth D_ins = (highest socket point within plug outer radius + 2 mm of the axis) − (lowest plug point). Δz < D_ins means the plug bottom is below the socket top.",
     "- Reference only: difference of inscribed radii (minimum distance to the axis). It is negative for pin-type tasks (plug surrounds a socket pin) and is therefore not used in the criterion.", "",
     "| Task | Clearance c (median~90%) | Strict threshold T | Insertion depth D_ins | Hole inscribed radius | Plug inscribed radius | Inscribed difference (ref.) | Overlap height | Socket outer radius |", "|---|---|---|---|---|---|---|---|---|"]
for k, v in out.items():
    if v: L.append(f"| {k} | {v['clearance_mm']:.2f} ({v['gap_median_mm']:.2f}~{v['gap_p90_mm']:.2f}) | {v['thr_mm']:.2f} | {v['d_ins_mm']:.1f} | {v['r_hole_mm']:.2f} | {v['r_plug_mm']:.2f} | {v['inscribed_diff_mm']:.2f} | {v['overlap_mm']:.1f} | {v['socket_outer_mm']:.1f} |")
    else: L.append(f"| {k} | not computable | | | | | |")
cs = np.array([v["clearance_mm"] for v in out.values() if v])
L += ["", f"- Computed for {len(cs)} of 100 tasks. Median clearance {np.median(cs):.2f} mm, range {cs.min():.2f} to {cs.max():.2f} mm. {int((cs > 1).sum())} tasks exceed 1 mm."]
open(os.path.join(OUT, "strict_geom.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L[-1:]))
for k in ["00638", "01132", "01053", "01102", "00141", "00211", "01036", "01041", "01079", "01092", "01125", "01129", "01136", "00426", "00783"]:
    v = out[k]; print(k, "D_ins", round(v["d_ins_mm"], 1), "c", round(v["clearance_mm"], 2), "p10-90", round(v["gap_p10_mm"], 2), round(v["gap_p90_mm"], 2), "T", round(v["thr_mm"], 2), "inscr", round(v["inscribed_diff_mm"], 2))

# Numerical check that the mean keypoint error of the official check is >= |p - s| (since Σ offsets = 0). See docs/02_official_success_check.md.
import numpy as np
from scipy.spatial.transform import Rotation as R
rng = np.random.default_rng(0); N = 200000
o = (np.linspace(0, 1, 4) - 0.5) * 0.15
d = rng.normal(size=(N, 3)) * 0.01
Rs = R.random(N, random_state=1)
def kpmean(Rp, Rs, d):
    s = 0
    for z in o:
        v = np.array([0, 0, z]); s = s + np.linalg.norm(d + Rp.apply(v) - Rs.apply(v), axis=1)
    return s / 4
Rp = Rs * R.from_rotvec(rng.normal(size=(N, 3)) * rng.uniform(0, 0.5, (N, 1)))
print('min(mean_kp - |p-s|) [m] =', (kpmean(Rp, Rs, d) - np.linalg.norm(d, axis=1)).min())
Ry = Rs * R.from_rotvec(np.c_[np.zeros(N), np.zeros(N), rng.uniform(-np.pi, np.pi, N)])
print('yaw-only max |mean_kp - |p-s|| [m] =', np.abs(kpmean(Ry, Rs, d) - np.linalg.norm(d, axis=1)).max())
for deg in [1, 2, 5, 10]:
    t = R.from_rotvec([np.deg2rad(deg), 0, 0])
    print(f'{deg} deg tilt, same origin -> mean kp {1e3*np.mean([np.linalg.norm(t.apply([0,0,z])-[0,0,z]) for z in o]):.2f} mm')

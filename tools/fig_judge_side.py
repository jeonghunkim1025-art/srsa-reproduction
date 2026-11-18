"""Side view of the official success region for a deep task (01132) and a shallow task (01079).

Output: figs/judge_side.png
"""
import os
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs", "judge_side.png")
fig, axs = plt.subplots(1, 2, figsize=(13, 6.3))


def base(a, title):
    th = np.linspace(0, np.pi / 2, 200); a.plot(15 * np.cos(th), 15 * np.sin(th), "k--", lw=2)
    a.axhline(0, color="#888", lw=1)
    a.set_xlim(0, 20); a.set_ylim(-1, 20); a.set_aspect("equal")
    a.set_xlabel("lateral distance of plug origin, d_xy (mm)", fontsize=11)
    a.set_ylabel("height above goal position, Δz (mm)", fontsize=11)
    a.set_title(title, fontsize=12.5)
    a.text(10.9, 11.3, "15 mm", fontsize=11)


# 01132: deep task
a = axs[0]; base(a, "01132 (deep task: insertion depth 56.8 mm)")
z = np.linspace(0, 15, 300)
a.fill_betweenx(z, 0, 1.37, color="#3b82c4", alpha=.85)
zz = np.linspace(0, np.sqrt(225 - 13.9 ** 2), 100); a.fill_betweenx(zz, 13.9, np.sqrt(225 - zz ** 2), color="#d9534f", alpha=.9)
a.fill_betweenx(z, 1.37, np.minimum(13.9, np.sqrt(np.maximum(225 - z ** 2, 0))), color="#ddd", alpha=.8)
a.annotate("true insertion\n(lateral < 1.4 mm)", xy=(0.8, 9), xytext=(3, 16.5), fontsize=10.5, arrowprops=dict(arrowstyle="->"))
a.text(4.2, 6, "overlaps socket wall\n(physically impossible)", fontsize=10.5, color="#444")
a.annotate("(a) beside\nthe socket\n-> judged\nsuccess", xy=(14.4, 2), xytext=(15.6, 6.5), fontsize=10.5, color="#b52b27",
           arrowprops=dict(arrowstyle="->", color="#b52b27"))
a.text(0.4, 18.5, "Within 15 mm of the goal the plug is already >41 mm inside the hole;\n"
                  "passing above the hole is not counted as success.", fontsize=8.5, color="#333", va="center")
# 01079: shallow task
a = axs[1]; base(a, "01079 (shallow task: insertion depth 12.1 mm, d_dis 14 mm)")
z = np.linspace(0, 12.13, 200); a.fill_betweenx(z, 0, 1.34, color="#3b82c4", alpha=.85)
a.fill_betweenx(z, 1.34, np.sqrt(225 - z ** 2), color="#ddd", alpha=.8)
z2 = np.linspace(12.13, 14, 100); a.fill_betweenx(z2, 0, np.sqrt(225 - z2 ** 2), color="#d9534f", alpha=.9)
a.axhline(14, color="#b52b27", lw=1.2, ls=":"); a.text(13.5, 14.4, "height limit d_dis = 14", fontsize=10, color="#b52b27", ha="right")
a.axhline(12.13, color="#555", lw=1.2); a.text(19.8, 12.5, "plug tip reaches socket top", fontsize=9.5, ha="right")
a.annotate("(b) above the hole or resting\non top -> judged success", xy=(4, 13.1), xytext=(6.0, 17), fontsize=10.5, color="#b52b27",
           arrowprops=dict(arrowstyle="->", color="#b52b27"))
a.annotate("true insertion", xy=(0.8, 6), xytext=(3, 3), fontsize=10.5, arrowprops=dict(arrowstyle="->"))
a.text(4.5, 8, "overlaps socket wall\n(physically impossible)", fontsize=10.5, color="#444")
fig.suptitle("Side view of the success region: dashed line = 15 mm radius around the goal position (including Δz)", fontsize=13)
plt.tight_layout(); plt.savefig(OUT, dpi=160)

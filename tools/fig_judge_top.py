"""Top view of the official success region for task 01132 (schematic from measured dimensions).

Output: figs/judge_top.png
"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs", "judge_top.png")
fig, a = plt.subplots(figsize=(7.5, 7.5))
a.add_patch(Rectangle((-30, -9.4), 60, 40, color="#c9ced6"))
a.text(-27, 24, "socket body", fontsize=13, color="#333")
a.add_patch(Circle((0, 0), 4.86, color="white", ec="#555", lw=1.5))
a.add_patch(Circle((0, 0), 4.44, color="#3b82c4", alpha=.9))
a.add_patch(Circle((0, -14.1), 4.5, color="#d9534f", alpha=.9))
a.add_patch(Circle((0, 0), 15, fill=False, ls="--", lw=2.2, color="#111"))
a.annotate("", xy=(0, -14.1), xytext=(0, 0), arrowprops=dict(arrowstyle="<->", lw=1.3))
a.text(1.0, -8.5, "14.1 mm", fontsize=12)
a.annotate("", xy=(10.6, 10.6), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1.3, ls=":"))
a.text(7, 7.5, "15 mm", fontsize=12)
a.annotate("hole (goal position)", xy=(-3.6, 3.3), xytext=(-28, 8), fontsize=11, arrowprops=dict(arrowstyle="->", lw=1))
a.text(-29, -29, "Blue: true insertion (lateral error 0.4 mm) -> success\n"
                 "Red: plug standing against the outer socket wall (14.1 mm) -> success\n"
                 "Dashed circle: region accepted by the official check",
       fontsize=11, va="bottom")
a.set_xlim(-30, 30); a.set_ylim(-30, 30); a.set_aspect("equal")
a.set_title("Top view - task 01132 (schematic from measured dimensions)", fontsize=13)
a.set_xlabel("mm"); a.set_ylabel("mm")
plt.tight_layout(); plt.savefig(OUT, dpi=160)

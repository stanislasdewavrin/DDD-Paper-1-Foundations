"""
Paper I — Master figure
========================

Single-panel summary of the four internal-consistency tests of Paper I.
Loads the saved JSON outputs of tests 1-5 and produces a 2x2 figure.

Run after tests 01..05.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HERE     = Path(__file__).resolve().parent.parent
DATA_DIR = HERE / "data"
FIG_DIR  = HERE / "figures"

# Load
with open(DATA_DIR / "conservation_results.json")  as f: cons = json.load(f)
with open(DATA_DIR / "identity_results.json")      as f: idy  = json.load(f)
with open(DATA_DIR / "relaxation_results.json")    as f: relx = json.load(f)
with open(DATA_DIR / "sr_recovery_results.json")   as f: srs  = json.load(f)

fig, axes = plt.subplots(2, 2, figsize=(11, 9))

# --- Top-left: conservation residual (E0=1.0)
e0_key = "E0=1.0"
clip_pct = cons[e0_key]["clip_pct"]
flux_resid = cons[e0_key]["max_flux_resid"]
labels = ["E0=0.2", "E0=0.5", "E0=1.0", "E0=2.0"]
clips = [cons[k]["clip_pct"] for k in labels]
ax = axes[0, 0]
ax.bar(labels, clips, color="C0")
ax.set_ylabel("clip / R_total_init  (%)")
ax.set_title("Test 1: Conservation\n"
             rf"flux zero-sum: $\leq{flux_resid:.0e}$, clip $\sim$ structural")
ax.grid(True, alpha=0.3, axis="y")

# --- Top-right: identity preservation under drainage
hist = idy["2b_history"]
ticks = [h["tick"] for h in hist]
res   = [abs(h["residual"]) + 1e-20 for h in hist]
ax = axes[0, 1]
ax.semilogy(ticks, res, "b-")
ax.axhline(1e-15, color="grey", lw=0.5, ls="--", label="machine precision")
ax.set_xlabel("tick")
ax.set_ylabel(r"$|T^2+I^2-\chi|$")
ax.set_title(r"Test 2: Identity $T^2+I^2=\chi$ preserved under drainage")
ax.legend()
ax.grid(True, alpha=0.3)

# --- Bottom-left: relaxation phi_inf agreement
b4 = relx["4b_full"]
sims = [c["phi_inf_sim"] for c in b4]
ths  = [c["phi_inf_th"]  for c in b4]
ax = axes[1, 0]
ax.plot([0, np.pi / 2], [0, np.pi / 2], "k--", lw=0.7, label="y=x")
ax.plot(ths, sims, "o", markersize=10, color="C2")
ax.set_xlabel(r"$\arctan(\lambda/\mu)$ analytic")
ax.set_ylabel(r"$\phi_\infty$ simulated")
ax.set_title("Test 4: Inertial relaxation\n"
             r"$\phi_\infty = \arctan(\lambda/\mu)$ to $\sim 10^{-6}$")
ax.legend()
ax.grid(True, alpha=0.3)

# --- Bottom-right: SR recovery
sr_data = srs["results"]
betas = [r["beta"] for r in sr_data]
dt_trame = [r["dtau_trame"] for r in sr_data]
dt_sr    = [r["dtau_sr"]    for r in sr_data]
ax = axes[1, 1]
bs = np.linspace(0.0, 0.99, 100)
ax.plot(bs, np.sqrt(1.0 - bs**2), "k-", lw=2,  label="SR analytic")
ax.plot(betas, dt_trame, "ro", markersize=8, label="DDD (chi=1)")
ax.set_xlabel(r"$\beta = v/c$")
ax.set_ylabel(r"$d\tau/dt$")
ax.set_title("Test 5: SR recovered at chi=1 (identity)")
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle("Paper I — Internal-consistency tests", fontsize=14, y=1.00)
fig.tight_layout()

fig_path = FIG_DIR / "fig06_master.pdf"
fig.savefig(fig_path, bbox_inches="tight")
fig.savefig(str(fig_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"Saved: {fig_path}")

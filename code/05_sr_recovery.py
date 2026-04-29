"""
Paper I — Test 5: Recovery of special-relativistic kinematics at chi=1
========================================================================

Internal-consistency identity (NOT a prediction): for chi=1,
T = sin(phi) and I = cos(phi), with beta = sin(phi), 1/gamma = cos(phi).
Then beta^2 + 1/gamma^2 = 1 trivially, dtau/dt = I = cos(phi) = sqrt(1-beta^2),
and the dispersion relation E^2 = p^2 c^2 + m^2 c^4 follows by definition
of (E, p, m).

This script verifies the numerical match across betas and demonstrates
that the residual against SR is exactly zero by construction.

Outputs:
    data/sr_recovery_results.json
    figures/fig05_sr.pdf
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HERE     = Path(__file__).resolve().parent.parent
DATA_DIR = HERE / "data"
FIG_DIR  = HERE / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


def trame_at_chi1(beta):
    phi = np.arcsin(beta)
    T = np.sin(phi)
    I = np.cos(phi)
    return phi, T, I


def sr_check(beta):
    phi, T, I = trame_at_chi1(beta)
    sumTI       = T**2 + I**2
    gamma_trame = 1.0 / I if I > 1e-30 else np.inf
    gamma_sr    = 1.0 / np.sqrt(1.0 - beta**2)
    dtau_trame  = I
    dtau_sr     = np.sqrt(1.0 - beta**2)
    return {
        "beta":       float(beta),
        "phi":        float(phi),
        "T":          float(T),
        "I":          float(I),
        "T2_plus_I2": float(sumTI),
        "gamma_trame": float(gamma_trame),
        "gamma_sr":    float(gamma_sr),
        "dtau_trame":  float(dtau_trame),
        "dtau_sr":     float(dtau_sr),
        "abs_dev_dtau": float(abs(dtau_trame - dtau_sr)),
    }


if __name__ == "__main__":
    print("Paper I — Test 5: SR recovery at chi=1")
    print("=" * 60)
    betas = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.866, 0.9, 0.99])
    results = [sr_check(b) for b in betas]
    print(f"{'beta':>5} {'gamma_sim':>10} {'gamma_SR':>10} {'dtau_sim':>10} {'dtau_SR':>10} {'dev':>9}")
    for r in results:
        print(f"{r['beta']:>5.3f} {r['gamma_trame']:>10.6f} {r['gamma_sr']:>10.6f} "
              f"{r['dtau_trame']:>10.6f} {r['dtau_sr']:>10.6f} {r['abs_dev_dtau']:>9.2e}")

    out_path = DATA_DIR / "sr_recovery_results.json"
    with open(out_path, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\nSaved: {out_path}")

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    bs = np.linspace(0.0, 0.99, 100)
    dt_trame = np.cos(np.arcsin(bs))
    dt_sr    = np.sqrt(1.0 - bs**2)
    ax.plot(bs, dt_sr,    "k-", lw=2, label="SR analytic")
    ax.plot(bs, dt_trame, "r--", lw=1.5, label=r"DDD $I=\cos\phi$")
    ax.set_xlabel(r"$\beta = v/c$")
    ax.set_ylabel(r"$d\tau/dt = 1/\gamma$")
    ax.set_title("SR kinematics recovered exactly at chi=1 (identity)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = FIG_DIR / "fig05_sr.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(str(fig_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {fig_path}")

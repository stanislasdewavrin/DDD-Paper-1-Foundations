"""
Paper I — Test 2: Interface identity T^2 + I^2 = chi
=====================================================

Three checks:

(2a) Algebraic identity T = sqrt(chi) sin(phi), I = sqrt(chi) cos(phi)
     => T^2 + I^2 = chi exactly, on a grid (chi, phi) in [0,1] x [0, pi/2].

(2b) Preservation under dynamical drainage tick by tick.

(2c) Proportional contraction: T/I = tan(phi) constant under drainage,
     for several velocities (beta = sin(phi)).

Outputs:
    data/identity_results.json
    figures/fig02_identity.pdf
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

R0    = 1.0
R_MIN = 0.005
KAPPA = 0.015
SEED  = 2024
np.random.seed(SEED)

HERE     = Path(__file__).resolve().parent.parent
DATA_DIR = HERE / "data"
FIG_DIR  = HERE / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


def TI_of(chi, phi):
    return np.sqrt(chi) * np.sin(phi), np.sqrt(chi) * np.cos(phi)


# ---------------------------------------------- 2a algebraic identity
def test_2a():
    chi_grid = np.array([0.005, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    beta_grid = np.array([0.0, 0.5, 0.707, 0.866, 1.0])
    max_resid = 0.0
    table = []
    for chi in chi_grid:
        row = [chi]
        for beta in beta_grid:
            phi = np.arcsin(beta)
            T, I = TI_of(chi, phi)
            r = T**2 + I**2 - chi
            row.append(r)
            max_resid = max(max_resid, abs(r))
        table.append(row)
    return max_resid, table


# ---------------------------------------------- 2b dynamical preservation
def test_2b(phi=np.pi / 4, n_ticks=100):
    """phi fixed, chi decays under drainage. Check identity at every tick."""
    R = np.array([R0])
    E = np.array([1.0])
    history = []
    max_resid = 0.0
    for t in range(n_ticks):
        chi = max(0.0, min(1.0, (R[0] - R_MIN) / (R0 - R_MIN)))
        T, I = TI_of(chi, phi)
        r = T**2 + I**2 - chi
        max_resid = max(max_resid, abs(r))
        history.append({
            "tick": t, "R": float(R[0]), "chi": chi,
            "T": float(T), "I": float(I), "T2_I2": float(T**2 + I**2),
            "residual": float(r),
        })
        # drainage step
        R = np.maximum(R_MIN, R - KAPPA * E * (chi ** 2))
    return max_resid, history


# ---------------------------------------------- 2c proportional contraction
def test_2c():
    chis = np.array([1.0, 0.929, 0.867, 0.766, 0.567, 0.397])
    betas = np.array([0.0, 0.3, 0.5, 0.7, 0.9])
    rows = []
    for beta in betas:
        phi = np.arcsin(beta)
        ratio_expected = np.tan(phi) if np.cos(phi) > 1e-12 else np.inf
        ratios = []
        I_values = []
        for chi in chis:
            T, I = TI_of(chi, phi)
            ratios.append(T / I if I > 1e-12 else np.inf)
            I_values.append(I)
        rows.append({
            "beta": float(beta),
            "tan_phi": float(ratio_expected),
            "T_over_I_per_chi": [float(r) for r in ratios],
            "I_per_chi":        [float(v) for v in I_values],
            "ratio_constancy_max_dev": float(np.std(ratios) / (np.mean(ratios) + 1e-30)),
        })
    return rows


# ---------------------------------------------- main
if __name__ == "__main__":
    print("Paper I — Test 2: interface identity T^2+I^2=chi")
    print("=" * 60)

    max_2a, table_2a = test_2a()
    print(f"(2a) Max residual on grid (chi, phi): {max_2a:.2e}")
    assert max_2a < 1e-14

    max_2b, hist_2b = test_2b()
    print(f"(2b) Max residual under dynamical drainage (100 ticks): {max_2b:.2e}")
    assert max_2b < 1e-14

    rows_2c = test_2c()
    print(f"(2c) Proportional contraction T/I = tan(phi):")
    for r in rows_2c:
        print(f"     beta={r['beta']:.2f}: tan(phi)={r['tan_phi']:.4f}, "
              f"max relative dev={r['ratio_constancy_max_dev']:.2e}")

    results = {
        "2a_max_residual_grid":      max_2a,
        "2a_table":                  table_2a,
        "2b_max_residual_drainage":  max_2b,
        "2b_history":                hist_2b,
        "2c_proportional":           rows_2c,
    }
    out_path = DATA_DIR / "identity_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Figure: residual vs tick under drainage; ratio T/I per chi for several beta
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    ticks = [h["tick"] for h in hist_2b]
    res   = [abs(h["residual"]) + 1e-20 for h in hist_2b]
    ax[0].semilogy(ticks, res, "b-")
    ax[0].axhline(1e-15, color="grey", lw=0.5, ls="--", label="machine precision")
    ax[0].set_xlabel("tick")
    ax[0].set_ylabel(r"$|T^2+I^2-\chi|$")
    ax[0].set_title("Identity preservation under drainage")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    chis = np.array([1.0, 0.929, 0.867, 0.766, 0.567, 0.397])
    for r in rows_2c:
        ratios = r["T_over_I_per_chi"]
        # Replace inf
        ratios = [v if np.isfinite(v) else None for v in ratios]
        if r["beta"] == 0.0:
            continue
        ax[1].plot(chis, ratios, "o-", label=fr"$\beta={r['beta']:.2f}$")
    ax[1].set_xlabel(r"$\chi$")
    ax[1].set_ylabel(r"$T/I$")
    ax[1].set_title("Proportional contraction: T/I = tan(phi) constant")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = FIG_DIR / "fig02_identity.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(str(fig_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {fig_path}")

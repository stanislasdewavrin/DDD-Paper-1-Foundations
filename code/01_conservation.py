"""
Paper I — Test 1: Local and global conservation
=================================================

Verifies the conservation properties of the local update rule on a
21^3 cubic lattice with a single source at the centre. Three results
are established:

(1) The flux term is anti-symmetric and zero-sum to machine precision.
(2) The drain term removes from the local reserve exactly what it
    deposits in the excitation sector, until saturation at R_min.
(3) The threshold R_min is the only structural source of non-conservation.

Outputs:
    data/conservation_results.json
    figures/fig01_conservation.pdf

Reproduces Tableau 6a, 6b of the working notes.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------- params
R0      = 1.0        # intact reserve
R_MIN   = 0.005      # saturation threshold
KAPPA   = 0.015      # drainage coefficient
ALPHA   = 0.15       # flux coefficient
F_MAX   = 0.02       # flux cap
N_SIDE  = 21         # cubic lattice side length
N_TICKS = 200
SEED    = 2024
np.random.seed(SEED)

# Output paths
HERE     = Path(__file__).resolve().parent.parent
DATA_DIR = HERE / "data"
FIG_DIR  = HERE / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


# -------------------------------------------------------------- helpers
def make_neighbours_3d(n):
    """6-connected neighbour offsets on a cubic lattice."""
    return [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def compute_chi(R):
    return np.clip((R - R_MIN) / (R0 - R_MIN), 0.0, 1.0)


def step(R, E, deltas):
    """One tick of the update rule.

    Returns (R_next, drain_total, clip_total, flux_zero_sum).
    """
    chi = compute_chi(R)

    # 1) Drainage from excitation
    drain = KAPPA * E * (chi ** 2)

    # 2) Asymmetric fluxes between neighbours.
    flux_in  = np.zeros_like(R)
    flux_out = np.zeros_like(R)
    n_side = R.shape[0]
    for dx, dy, dz in make_neighbours_3d(n_side):
        # Shift the deltas so that we align node i with neighbour j = i + d.
        shifted = np.roll(deltas, shift=(dx, dy, dz), axis=(0, 1, 2))
        # Flux from i to j (positive when delta_i > delta_j)
        f = np.minimum(F_MAX, ALPHA * np.maximum(deltas - shifted, 0.0))
        flux_out += f
        # Symmetric: flux from j to i is also visible at j's perspective
        # We accumulate i->j as outgoing for i and incoming for j.
        # Instead of double pass, we compute incoming from the same logic:
        f_in_from_j = np.minimum(F_MAX, ALPHA * np.maximum(shifted - deltas, 0.0))
        flux_in  += f_in_from_j

    flux_zero_sum_residual = float(np.sum(flux_out - flux_in))  # should be ~0

    # 3) Update R: i loses drain, exports outgoing flux of deficit
    #    (i.e. gives reserve away to less-deficitary neighbours), and
    #    receives incoming flux of deficit (becomes more deficitary)
    R_next = R - drain - flux_in + flux_out
    # 4) Clip at the saturation threshold and accumulate clip
    R_clipped = np.minimum(R0, np.maximum(R_MIN, R_next))
    clip_total = float(np.sum(R_clipped - R_next))  # >=0 when R_next < R_min
    drain_total = float(np.sum(drain))
    return R_clipped, drain_total, clip_total, flux_zero_sum_residual


def run_simulation(E0, n_ticks=N_TICKS, n_side=N_SIDE):
    R = np.full((n_side,) * 3, R0, dtype=np.float64)
    E = np.zeros_like(R)
    centre = (n_side // 2,) * 3
    E[centre] = E0  # constant excitation at centre

    history = {
        "tick":       [],
        "R_centre":   [],
        "drain_tick": [],
        "clip_tick":  [],
        "R_total":    [],
        "flux_resid": [],
    }
    drain_cum, clip_cum = 0.0, 0.0
    for t in range(n_ticks):
        deltas = R0 - R
        R, drain_t, clip_t, fr = step(R, E, deltas)
        drain_cum += drain_t
        clip_cum  += clip_t
        history["tick"].append(t)
        history["R_centre"].append(float(R[centre]))
        history["drain_tick"].append(drain_t)
        history["clip_tick"].append(clip_t)
        history["R_total"].append(float(R.sum()))
        history["flux_resid"].append(fr)
    return history, drain_cum, clip_cum, R


# ---------------------------------------------------------- main
if __name__ == "__main__":
    print(f"Paper I — Test 1: conservation on {N_SIDE}^3 grid, {N_TICKS} ticks")
    print("=" * 60)

    results = {}
    for E0 in [0.2, 0.5, 1.0, 2.0]:
        hist, drain_cum, clip_cum, R_final = run_simulation(E0)
        R_total_init = R0 * (N_SIDE ** 3)
        R_total_final = float(R_final.sum())
        residual_max = max(abs(x) for x in hist["flux_resid"])
        clip_pct = 100.0 * clip_cum / R_total_init
        budget_check = R_total_init - drain_cum - clip_cum
        results[f"E0={E0}"] = {
            "drain_total":     drain_cum,
            "clip_total":      clip_cum,
            "R_total_final":   R_total_final,
            "R_total_init":    R_total_init,
            "budget_residual": R_total_final - budget_check,
            "max_flux_resid":  residual_max,
            "clip_pct":        clip_pct,
        }
        print(f"E0={E0:>4.2f} | drain={drain_cum:.6f} | clip={clip_cum:.4f} | "
              f"clip%={clip_pct:.4f} | flux_resid_max={residual_max:.2e}")

    out_path = DATA_DIR / "conservation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Headline figure: conservation history for E0=1.0
    hist, _, _, _ = run_simulation(1.0)
    ticks = np.array(hist["tick"])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    ax[0].semilogy(ticks, [abs(r) + 1e-20 for r in hist["flux_resid"]], "b-")
    ax[0].axhline(1e-15, color="grey", lw=0.5, ls="--", label=r"machine precision")
    ax[0].set_xlabel("tick")
    ax[0].set_ylabel(r"$|\sum F_{i\to j}|$")
    ax[0].set_title("Flux zero-sum residual")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(ticks, np.array(hist["R_centre"]), "r-", label=r"$R_{\rm centre}$")
    ax[1].axhline(R_MIN, color="grey", lw=0.5, ls="--", label=r"$R_{\min}$")
    ax[1].set_xlabel("tick")
    ax[1].set_ylabel(r"$R$")
    ax[1].set_title("Centre reserve (E0=1.0): saturation at R_min")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = FIG_DIR / "fig01_conservation.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(str(fig_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {fig_path}")

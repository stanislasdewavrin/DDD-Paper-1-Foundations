"""
Paper I - Test 11: Two-source superposition and finite-resource breakdown
==========================================================================

Sanity check that demonstrates:

  (A) In the linear regime (mu = 1, weak sinks), the two-source
      stationary deficit field equals the sum of two independent
      one-source deficit fields, up to lattice and boundary errors.

  (B) When the sinks are strong enough that the local reserve is
      driven near the floor R_min, superposition fails locally
      because the rate-limiter beta_i and the floor jointly cap the
      delivered drainage.

The figure overlays equipotentials (white contour lines) on the
heatmaps so the reader can see field geometry and superposition
visually.

Outputs
-------
    data/11_two_source.json
    figures/fig11_two_source.pdf
    figures/fig11_two_source.png
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)
FIG  = HERE / "figures"; FIG.mkdir(exist_ok=True)

R0      = 1.0
R_MIN   = 0.0
KAPPA   = 1.0
ALPHA   = 1.0
L       = 41


def laplacian(R):
    s = np.zeros_like(R)
    s[1:, :, :]  += R[:-1, :, :]
    s[:-1, :, :] += R[1:, :, :]
    s[:, 1:, :]  += R[:, :-1, :]
    s[:, :-1, :] += R[:, 1:, :]
    s[:, :, 1:]  += R[:, :, :-1]
    s[:, :, :-1] += R[:, :, 1:]
    n_inside = 6 * np.ones_like(R)
    n_inside[0, :, :]  -= 1
    n_inside[-1, :, :] -= 1
    n_inside[:, 0, :]  -= 1
    n_inside[:, -1, :] -= 1
    n_inside[:, :, 0]  -= 1
    n_inside[:, :, -1] -= 1
    s += (6 - n_inside) * R0
    return s - 6 * R


def solve_steady(sinks, tol=1e-7, max_iter=20000, saturate=False):
    R = R0 * np.ones_like(sinks)
    omega = 0.5
    for it in range(max_iter):
        if saturate:
            beta = np.clip(R / R0, 0.0, 1.0)
        else:
            beta = np.ones_like(R)
        s_neighbours = laplacian(R) + 6 * R
        beta_safe = np.where(beta > 1e-3, beta, 1e-3)
        target = (s_neighbours / 6.0) - KAPPA * sinks / (6.0 * ALPHA * beta_safe)
        target = np.maximum(target, 0.0)
        R_new = (1 - omega) * R + omega * target
        diff = np.max(np.abs(R_new - R))
        R = R_new
        if diff < tol:
            break
    deficit = R0 - R
    return R, deficit, it


def make_sinks(positions, strength):
    s = np.zeros((L, L, L))
    for (i, j, k) in positions:
        s[i, j, k] = strength
    return s


def main():
    cx = L // 2
    sep = 8
    p1 = (cx - sep // 2, cx, cx)
    p2 = (cx + sep // 2, cx, cx)

    out = {"L": L, "sep": sep,
           "p1": list(p1), "p2": list(p2),
           "regimes": []}

    for label, strength, saturate in [
        ("A_linear", 0.001, False),
        ("B_saturating", 12.0, True),
    ]:
        s_left = make_sinks([p1], strength)
        s_right = make_sinks([p2], strength)
        s_both = make_sinks([p1, p2], strength)

        _, def_left, _  = solve_steady(s_left,  saturate=saturate)
        _, def_right, _ = solve_steady(s_right, saturate=saturate)
        R_both, def_both, _ = solve_steady(s_both, saturate=saturate)

        def_super = def_left + def_right
        err_pointwise = def_both - def_super
        rel_err_max = float(np.max(np.abs(err_pointwise))
                            / max(np.max(np.abs(def_both)), 1e-15))
        out["regimes"].append({
            "label": label,
            "strength": strength,
            "saturate": saturate,
            "def_max": float(def_both.max()),
            "rel_err_max": rel_err_max,
            "min_reserve": float(R_both.min()),
        })

    with open(DATA / "11_two_source.json", "w") as f:
        json.dump(out, f, indent=2)
    print("data ->", DATA / "11_two_source.json")
    for r in out["regimes"]:
        print(f"  {r['label']:<14} strength={r['strength']:.3g}, "
              f"def_max={r['def_max']:.3g}, "
              f"rel_err_max={r['rel_err_max']:.2e}, "
              f"min_reserve={r['min_reserve']:.3g}")

    # ------------------------------------------------------------------
    # Figure: 2x3 grid with equipotentials overlay
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.0))

    for row, (label, strength, saturate) in enumerate([
        ("A_linear", 0.001, False),
        ("B_saturating", 12.0, True),
    ]):
        s_left = make_sinks([p1], strength)
        s_right = make_sinks([p2], strength)
        s_both = make_sinks([p1, p2], strength)
        _, def_left, _  = solve_steady(s_left,  saturate=saturate)
        _, def_right, _ = solve_steady(s_right, saturate=saturate)
        _, def_both, _  = solve_steady(s_both, saturate=saturate)
        def_super = def_left + def_right
        residual = def_both - def_super

        z_slice = cx
        d_both = def_both[:, :, z_slice]
        d_super = def_super[:, :, z_slice]
        resid = residual[:, :, z_slice]

        vmax = max(d_both.max(), d_super.max())
        vmin = 0.0

        im0 = axes[row, 0].imshow(d_both.T, origin="lower",
                                   vmin=vmin, vmax=vmax, cmap="viridis")
        im1 = axes[row, 1].imshow(d_super.T, origin="lower",
                                   vmin=vmin, vmax=vmax, cmap="viridis")
        rm = max(1e-15, np.max(np.abs(resid)))
        im2 = axes[row, 2].imshow(resid.T, origin="lower",
                                   vmin=-rm, vmax=+rm, cmap="RdBu_r")

        # Equipotentials overlay (log-spaced)
        if vmax > 0:
            lev_low = max(vmax * 5e-3, 1e-12)
            levels = np.geomspace(lev_low, vmax * 0.9, 8)
            axes[row, 0].contour(d_both.T, levels=levels,
                                  colors="white", linewidths=0.7,
                                  alpha=0.6)
            axes[row, 1].contour(d_super.T, levels=levels,
                                  colors="white", linewidths=0.7,
                                  alpha=0.6)
            res_levels = [-0.5*rm, 0.0, 0.5*rm]
            axes[row, 2].contour(resid.T, levels=res_levels,
                                  colors=["blue", "black", "red"],
                                  linewidths=[0.6, 1.0, 0.6],
                                  alpha=0.8)

        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])
            ax.scatter([p1[0], p2[0]], [p1[1], p2[1]],
                       facecolors="none", edgecolors="yellow",
                       linewidths=1.6, s=80, zorder=5)

        regime_label = ("(A) Linear regime ($\\mu = 1$, weak sinks)"
                        if not saturate else
                        "(B) Saturating regime (rate-limiter active)")
        axes[row, 0].set_ylabel(regime_label, fontsize=10.5,
                                rotation=90, labelpad=12)

        plt.colorbar(im0, ax=axes[row, 0], fraction=0.045, pad=0.03)
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.045, pad=0.03)
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.045, pad=0.03)

        if row == 0:
            axes[row, 0].set_title("two-source deficit $\\delta_{12}$\n"
                                    "(heatmap + equipotentials)")
            axes[row, 1].set_title("$\\delta_1 + \\delta_2$ (sum of single-source)\n"
                                    "(heatmap + equipotentials)")
            axes[row, 2].set_title("residual $\\delta_{12} - (\\delta_1 + \\delta_2)$")

        rel_max = rm / max(d_both.max(), 1e-15) * 100
        axes[row, 2].text(0.04, 0.94,
                          f"max|residual|/max($\\delta_{{12}}$) = {rel_max:.1f}\\%",
                          transform=axes[row, 2].transAxes,
                          fontsize=9, color="black",
                          bbox=dict(facecolor="white", alpha=0.85,
                                    edgecolor="none"))

    fig.suptitle("Two-source superposition: heatmap + equipotentials, "
                 "linear vs finite-resource regime",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(FIG / "fig11_two_source.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig11_two_source.png", dpi=180, bbox_inches="tight")
    print("figure ->", FIG / "fig11_two_source.pdf")


if __name__ == "__main__":
    main()

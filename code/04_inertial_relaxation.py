"""
Paper I — Test 4: Inertial relaxation of phi
==============================================

(4a) Simple relaxation:  dphi/dt = (phi_eq - phi)/tau_ret.
     Number of ticks to reach |phi - phi_eq|/|phi_0 - phi_eq| <= eps
     is t(eps) = tau_ret * ln(1/eps).

(4b) Full balance:  dphi/dt = lambda*cos(phi) - mu*sin(phi).
     Stationary phi_inf = arctan(lambda/mu),
     beta_inf = lambda / sqrt(lambda^2 + mu^2).

Outputs:
    data/relaxation_results.json
    figures/fig04_relaxation.pdf
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SEED = 2024
np.random.seed(SEED)

HERE     = Path(__file__).resolve().parent.parent
DATA_DIR = HERE / "data"
FIG_DIR  = HERE / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


def relax_simple(phi0, phi_eq, tau, n_ticks=2000):
    """Simple relaxation. Returns trajectory and tick to reach 10/5/1%."""
    phi = phi0
    traj = [phi]
    threshold_ticks = {0.10: None, 0.05: None, 0.01: None}
    delta0 = abs(phi_eq - phi0)
    for t in range(1, n_ticks + 1):
        phi += (phi_eq - phi) / tau
        traj.append(phi)
        rel = abs(phi - phi_eq) / (delta0 + 1e-30)
        for eps in threshold_ticks:
            if threshold_ticks[eps] is None and rel <= eps:
                threshold_ticks[eps] = t
    return np.array(traj), threshold_ticks


def relax_full(lam, mu, phi0=0.0, n_ticks=2000, tol=1e-8):
    """dphi/dt = lambda cos(phi) - mu sin(phi). Iterate until convergence."""
    phi = phi0
    traj = [phi]
    for t in range(1, n_ticks + 1):
        dphi = lam * np.cos(phi) - mu * np.sin(phi)
        phi += dphi
        # clamp to [0, pi/2]
        phi = max(0.0, min(np.pi / 2, phi))
        traj.append(phi)
        if abs(dphi) < tol:
            return np.array(traj), t, phi
    return np.array(traj), n_ticks, phi


# ------------------------------------------------------ tests
def test_4a():
    """Two transitions: rest -> very fast, moderate -> fast"""
    cases = []
    for transition_label, beta0, beta_eq in [
        ("rest -> very fast", 0.0, 0.9),
        ("moderate -> fast",  0.3, 0.7),
    ]:
        phi0   = np.arcsin(beta0)
        phi_eq = np.arcsin(beta_eq)
        for tau in [3, 5, 10, 20, 50]:
            traj, ticks = relax_simple(phi0, phi_eq, tau)
            cases.append({
                "transition":  transition_label,
                "beta0":       beta0,
                "beta_eq":     beta_eq,
                "tau":         tau,
                "ticks_eps10": ticks[0.10],
                "ticks_eps5":  ticks[0.05],
                "ticks_eps1":  ticks[0.01],
                "phi_eq":      float(phi_eq),
                "phi_final":   float(traj[-1]),
                "residual_pct": 100.0 * abs(traj[-1] - phi_eq) / phi_eq,
            })
    return cases


def test_4b():
    """Stationary phi_inf vs arctan(lambda/mu) for various (lambda, mu)."""
    cases = []
    for label, lam, mu in [
        ("lambda dominates",   0.030, 0.010),
        ("balance",            0.030, 0.030),
        ("strong confinement", 0.030, 0.100),
        ("very strong conf.",  0.010, 0.030),
        ("intermediate",       0.050, 0.020),
        ("high speed",         0.100, 0.050),
    ]:
        traj, ticks_conv, phi_inf = relax_full(lam, mu)
        phi_inf_th = np.arctan(lam / mu) if mu > 0 else np.pi / 2
        beta_inf_th = lam / np.sqrt(lam**2 + mu**2)
        beta_inf    = np.sin(phi_inf)
        cases.append({
            "label":       label,
            "lambda":      lam,
            "mu":          mu,
            "phi_inf_sim": float(phi_inf),
            "phi_inf_th":  float(phi_inf_th),
            "beta_inf_sim": float(beta_inf),
            "beta_inf_th":  float(beta_inf_th),
            "ticks_conv":  ticks_conv,
            "abs_dev":     float(abs(phi_inf - phi_inf_th)),
        })
    return cases


# ------------------------------------------------------ main
if __name__ == "__main__":
    print("Paper I — Test 4: inertial relaxation of phi")
    print("=" * 60)
    cases_4a = test_4a()
    print("\n(4a) Simple relaxation: ticks vs tau * ln(1/eps)")
    print(f"{'transition':<22} {'tau':>4} {'eps10':>6} {'eps5':>6} {'eps1':>6} {'theory ln(1/0.01)':>20}")
    for c in cases_4a:
        th = c["tau"] * np.log(100.0)
        print(f"{c['transition']:<22} {c['tau']:>4} {c['ticks_eps10']:>6} "
              f"{c['ticks_eps5']:>6} {c['ticks_eps1']:>6} {th:>20.1f}")

    cases_4b = test_4b()
    print("\n(4b) Stationary phi_inf vs arctan(lambda/mu)")
    print(f"{'label':<22} {'lambda':>7} {'mu':>7} {'phi_sim':>10} {'phi_th':>10} {'dev':>10}")
    for c in cases_4b:
        print(f"{c['label']:<22} {c['lambda']:>7.4f} {c['mu']:>7.4f} "
              f"{c['phi_inf_sim']:>10.6f} {c['phi_inf_th']:>10.6f} {c['abs_dev']:>10.2e}")

    out = {"4a_simple": cases_4a, "4b_full": cases_4b}
    out_path = DATA_DIR / "relaxation_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Figure
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    # Left: trajectories for various tau
    for tau in [3, 10, 50]:
        traj, _ = relax_simple(0.0, np.arcsin(0.9), tau, n_ticks=300)
        t = np.arange(len(traj))
        ax[0].plot(t, traj, label=fr"$\tau_{{\rm ret}}={tau}$")
    ax[0].axhline(np.arcsin(0.9), color="grey", lw=0.5, ls="--",
                  label=r"$\phi_{\rm eq}=\arcsin(0.9)$")
    ax[0].set_xlabel("tick")
    ax[0].set_ylabel(r"$\phi$ (rad)")
    ax[0].set_title("Simple relaxation")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # Right: phi_inf vs theory across (lambda, mu) cases
    cases = test_4b()
    sims = [c["phi_inf_sim"] for c in cases]
    ths  = [c["phi_inf_th"]  for c in cases]
    ax[1].plot([0, np.pi / 2], [0, np.pi / 2], "k--", lw=0.5, label="y=x")
    ax[1].plot(ths, sims, "o", markersize=8, color="C0")
    ax[1].set_xlabel(r"$\arctan(\lambda/\mu)$ analytic")
    ax[1].set_ylabel(r"$\phi_\infty$ simulated")
    ax[1].set_title("Stationary equilibrium")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = FIG_DIR / "fig04_relaxation.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(str(fig_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {fig_path}")

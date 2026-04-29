"""
Explicit time-stepped DDD rule (v7 convention)
================================================

Implements the rule of Paper I v7 *literally*:
- Reserve R_i in [0, infinity), rest value R_0 (NOT a ceiling)
- Directional fluxes F_{i->j} = beta_i * alpha * (R_i - R_j)_+
- Rate-limiter beta_i computed per node (no clipping)
- External sink E_i scaled by the same beta_i
- Update at every tick of duration tau

In the linear regime (no node reaches R_i = 0), beta_i = 1
everywhere and the rule reduces to explicit Euler integration of
dR/dt = alpha * Delta R - kappa * E. The stationary 1/r profile
should match the Jacobi-solver result of G_measure_standalone.py.

This script is the "physical" simulation; G_measure_standalone is
the "fast" linearised solver. They should agree on the 1/r
coefficient.

USAGE
    python v7_drainage_rule.py --L 32 --tau 0.05 --n_ticks 30000

Pure numpy, no scipy.
"""
import argparse
import json
import time
import sys
import numpy as np


def laplacian_cubic(R):
    """Discrete 6-point Laplacian on a cubic lattice with periodic
    or reflective handling at the edges. Implemented with np.roll
    + boundary correction for Dirichlet (R outside = R_0)."""
    s = np.zeros_like(R)
    s[1:, :, :]  += R[:-1, :, :]
    s[:-1, :, :] += R[1:, :, :]
    s[:, 1:, :]  += R[:, :-1, :]
    s[:, :-1, :] += R[:, 1:, :]
    s[:, :, 1:]  += R[:, :, :-1]
    s[:, :, :-1] += R[:, :, 1:]
    return s - 6.0 * R


def directional_flux_sums(R, alpha):
    """Compute, for each node, the desired total outflow O^des_i
    and the desired total inflow S^in_i, summed over the 6 cubic
    neighbours, using F^des_{i->j} = alpha * (R_i - R_j)_+ .

    Returns (O_des, S_in) arrays of shape R.shape."""
    O = np.zeros_like(R)
    S = np.zeros_like(R)
    for axis in range(3):
        for shift in (-1, +1):
            Rn = np.roll(R, shift, axis=axis)
            # Out from i to its neighbour in (axis, shift) direction
            out = alpha * np.maximum(R - Rn, 0.0)
            # In to i from that same neighbour: use -shift
            Rn2 = np.roll(R, -shift, axis=axis)
            inc = alpha * np.maximum(Rn2 - R, 0.0)
            O += out
            S += inc
    # Dirichlet correction: at the boundary, neighbours outside
    # are at R_0 (rest level). They contribute as if R_outside = R_0.
    return O, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--n_ticks", type=int, default=20000)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--R0", type=float, default=1.0)
    ap.add_argument("--E0", type=float, default=0.5,
                    help="External sink strength at the centre. "
                         "Choose < R_0 / tau / kappa to keep beta=1.")
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    L = args.L
    cx, cy, cz = L // 2, L // 2, L // 2
    tau = args.tau
    alpha = args.alpha
    kappa = args.kappa
    R0 = args.R0
    E0 = args.E0

    print("=" * 72)
    print("DDD v7 explicit rate-limited rule")
    print("=" * 72)
    print(f"L = {L}, tau = {tau}, alpha = {alpha}, kappa = {kappa}")
    print(f"R_0 = {R0}, E_0 = {E0} (single point sink at centre)")
    print(f"n_ticks = {args.n_ticks}")

    # Initialize at rest
    R = np.full((L, L, L), R0, dtype=np.float64)

    # External sink: only at the central node
    E = np.zeros((L, L, L))
    E[cx, cy, cz] = E0

    # Stability advisory
    stab = 1.0 / (6.0 * alpha)
    if tau > stab:
        print(f"WARNING tau = {tau} > stability bound {stab:.4f}; "
              f"may diverge.")

    log_every = max(1, args.n_ticks // 20)
    beta_min_log = []

    t0 = time.time()
    for it in range(args.n_ticks):
        # Compute desired flux totals per node
        O_des, S_in = directional_flux_sums(R, alpha)
        # Available local budget
        A = R + tau * S_in
        # Demand
        D = tau * O_des + tau * kappa * E
        # Rate-limiter (handle D == 0 case)
        beta = np.where(D > 1e-15, np.minimum(1.0, A / np.maximum(D, 1e-15)), 1.0)
        # Update
        R = R + tau * S_in - beta * tau * O_des - beta * tau * kappa * E
        # Numerical hygiene: clamp tiny negatives from float roundoff
        R = np.maximum(R, 0.0)
        # Apply Dirichlet at boundary: tie outermost layer to R_0
        R[0, :, :] = R0; R[-1, :, :] = R0
        R[:, 0, :] = R0; R[:, -1, :] = R0
        R[:, :, 0] = R0; R[:, :, -1] = R0

        if it % log_every == 0 or it == args.n_ticks - 1:
            d_center = R0 - float(R[cx, cy, cz])
            d_5 = R0 - float(R[cx + 5, cy, cz]) if cx + 5 < L else 0
            d_10 = R0 - float(R[cx + 10, cy, cz]) if cx + 10 < L else 0
            min_R = float(R.min())
            beta_min = float(beta.min())
            beta_min_log.append(beta_min)
            print(f"  it={it:7d}  d0={d_center:.5f}  d5={d_5:.5f}  "
                  f"d10={d_10:.5f}  min(R)={min_R:.5f}  "
                  f"min(beta)={beta_min:.4f}")
    elapsed = time.time() - t0
    print(f"Total time: {elapsed:.1f} s")

    # Radial profile of deficit delta = R_0 - R
    delta = R0 - R
    xs = np.arange(L) - cx
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing='ij')
    r = np.sqrt(X**2 + Y**2 + Z**2)
    r_bins = np.arange(0.5, L * 0.4, 0.25)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    profile = np.zeros(len(r_centers))
    err = np.zeros(len(r_centers))
    counts = np.zeros(len(r_centers), dtype=int)
    for i in range(len(r_centers)):
        mask = (r >= r_bins[i]) & (r < r_bins[i + 1])
        n = int(mask.sum())
        counts[i] = n
        if n > 1:
            profile[i] = float(delta[mask].mean())
            err[i] = float(delta[mask].std() / np.sqrt(n))

    # Newton fit: delta = A/r + B
    r_min_fit = 3.0
    r_max_fit = 0.30 * L
    mask_fit = ((r_centers > r_min_fit) & (r_centers < r_max_fit)
                & (counts > 5) & (profile > 0))
    n_bins = int(mask_fit.sum())
    if n_bins < 5:
        print("FAIL: insufficient bins for fit.")
        return 1
    x_fit = 1.0 / r_centers[mask_fit]
    y_fit = profile[mask_fit]
    e_fit = np.maximum(err[mask_fit], 1e-15)
    w = 1.0 / (e_fit * e_fit)
    Sw = np.sum(w); Swx = np.sum(w * x_fit); Swy = np.sum(w * y_fit)
    Swxx = np.sum(w * x_fit * x_fit); Swxy = np.sum(w * x_fit * y_fit)
    det = Sw * Swxx - Swx * Swx
    A = (Sw * Swxy - Swx * Swy) / det
    B = (Swxx * Swy - Swx * Swxy) / det
    A_err = float(np.sqrt(Sw / det))

    # Theoretical: -alpha Delta delta = kappa E => delta = M/(4 pi r)
    # with M = kappa E_0 / alpha
    M_eff = kappa * E0 / alpha
    A_continuum = M_eff / (4.0 * np.pi)

    print()
    print("--- Radial fit ---")
    print(f"  fit range: [{r_min_fit}, {r_max_fit:.1f}], n_bins = {n_bins}")
    print(f"  A_fit       = {A:.7f} +/- {A_err:.2e}")
    print(f"  A_continuum = M/(4 pi) = {A_continuum:.7f} (with M = kappa E_0 / alpha = {M_eff})")
    bias_pct = (A - A_continuum) / A_continuum * 100
    print(f"  bias        = {bias_pct:.3f} %")
    print(f"  rel. precision A: {A_err/abs(A):.2e}")

    print()
    print("--- Diagnostics ---")
    print(f"  min(R) over run: never went below 0 (rate-limiter active "
          f"if min(beta) < 1)")
    print(f"  min(beta) at end: {beta.min():.6f}")
    if beta.min() > 0.999:
        print("  beta = 1 everywhere => linear regime confirmed.")
    else:
        print(f"  beta_min = {beta.min():.4f} => rate-limiter activated "
              f"somewhere; result is non-linear.")

    # Save
    if args.output is None:
        out = f"v7_rule_L{L}_tau{tau}_E{E0}_n{args.n_ticks}.json"
    else:
        out = args.output
    summary = {
        "L": L, "tau": tau, "alpha": alpha, "kappa": kappa,
        "R0": R0, "E0": E0, "n_ticks": args.n_ticks,
        "A_fit": float(A), "A_err": float(A_err),
        "A_continuum": float(A_continuum),
        "bias_pct": float(bias_pct),
        "rel_precision_A": float(A_err / abs(A)),
        "min_beta_final": float(beta.min()),
        "min_R_final": float(R.min()),
        "elapsed_seconds": float(elapsed),
        "linear_regime": bool(beta.min() > 0.999),
    }
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

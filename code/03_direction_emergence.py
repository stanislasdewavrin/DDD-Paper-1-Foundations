"""
Paper I — Test 3: Emergence of direction n_i from gradient of retention
=========================================================================

Four configurations:

    Case 1 — Linear gradient R(i,j) = 1 + 0.05*i      => n_i = (+1, 0)
    Case 2 — Radial profile R(r) = 1 - 0.3/r          => n_i radial out
    Case 3 — Diagonal gradient R(i,j) = 1 + 0.03*(i+j) => n_i = (.707, .707)
    Case 4 — Double symmetric source                  => |A_i|=0 at centre

Outputs:
    data/direction_results.json
    figures/fig03_direction.pdf
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

LAMBDA_R = 1.0
N = 11
SEED = 2024
np.random.seed(SEED)

HERE     = Path(__file__).resolve().parent.parent
DATA_DIR = HERE / "data"
FIG_DIR  = HERE / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


def compute_n(R):
    """Compute n_i and |A_i| at every interior node, 4-connected 2D.

    A_i = sum_{j ~ i} (C_j - C_i) e_{i->j}, with C_j = lambda_R * R_j.
    Returns (n_x, n_y, |A|).
    """
    C = LAMBDA_R * R
    H, W = R.shape
    n_x = np.zeros_like(R)
    n_y = np.zeros_like(R)
    A_norm = np.zeros_like(R)
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            Ax = (C[i + 1, j] - C[i, j]) * 1 + (C[i - 1, j] - C[i, j]) * (-1)
            Ay = (C[i, j + 1] - C[i, j]) * 1 + (C[i, j - 1] - C[i, j]) * (-1)
            norm = np.hypot(Ax, Ay)
            A_norm[i, j] = norm
            if norm > 1e-15:
                n_x[i, j] = Ax / norm
                n_y[i, j] = Ay / norm
    return n_x, n_y, A_norm


def case1_linear():
    R = np.array([[1.0 + 0.05 * i for j in range(N)] for i in range(N)])
    return R, "Linear gradient (+x)"


def case2_radial():
    R = np.zeros((N, N))
    cx, cy = N // 2, N // 2
    for i in range(N):
        for j in range(N):
            r = max(0.5, np.hypot(i - cx, j - cy))
            R[i, j] = 1.0 - 0.3 / r
    return R, "Radial source at centre"


def case3_diagonal():
    R = np.array([[1.0 + 0.03 * (i + j) for j in range(N)] for i in range(N)])
    return R, "Diagonal gradient (+x,+y)"


def case4_double():
    R = np.full((N, N), 1.0)
    cx1, cy1 = 3, N // 2
    cx2, cy2 = 7, N // 2
    for i in range(N):
        for j in range(N):
            d1 = max(0.5, np.hypot(i - cx1, j - cy1))
            d2 = max(0.5, np.hypot(i - cx2, j - cy2))
            R[i, j] = 1.0 - 0.2 / d1 - 0.2 / d2
    return R, "Double symmetric source"


def angle_of(nx, ny):
    return np.degrees(np.arctan2(ny, nx))


if __name__ == "__main__":
    print("Paper I — Test 3: emergence of direction n_i")
    print("=" * 60)
    cases = [case1_linear(), case2_radial(), case3_diagonal(), case4_double()]
    out = {}
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    for idx, ((R, label), ax) in enumerate(zip(cases, axes.flat)):
        nx, ny, A_norm = compute_n(R)
        # Sample a few interior nodes for the table
        samples = [(2, N // 2), (N // 2, N // 2), (N - 3, N // 2),
                   (N // 2, 2), (N // 2, N - 3),
                   (3, 3), (N - 4, N - 4)]
        case_data = []
        for (i, j) in samples:
            case_data.append({
                "i": i, "j": j,
                "R":      float(R[i, j]),
                "A_norm": float(A_norm[i, j]),
                "n_x":    float(nx[i, j]),
                "n_y":    float(ny[i, j]),
                "angle":  float(angle_of(nx[i, j], ny[i, j])),
            })
        out[f"case{idx+1}_{label}"] = case_data
        print(f"\n{label}")
        for d in case_data:
            print(f"  ({d['i']},{d['j']}): n=({d['n_x']:+.4f}, {d['n_y']:+.4f}) "
                  f"angle={d['angle']:+.2f}°  |A|={d['A_norm']:.4f}")

        # Vector field plot
        X, Y = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        ax.imshow(R.T, origin="lower", cmap="viridis", alpha=0.6)
        # Skip border for arrows
        skip = 1
        ax.quiver(X[1:-1:skip, 1:-1:skip], Y[1:-1:skip, 1:-1:skip],
                  nx[1:-1:skip, 1:-1:skip], ny[1:-1:skip, 1:-1:skip],
                  color="red", scale_units="xy", scale=2, width=0.005)
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])

    out_path = DATA_DIR / "direction_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    fig.tight_layout()
    fig_path = FIG_DIR / "fig03_direction.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(str(fig_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {fig_path}")

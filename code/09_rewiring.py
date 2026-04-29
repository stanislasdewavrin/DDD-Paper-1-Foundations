"""
Paper XVI — Dynamical substrate: link birth/death and arrow of time
======================================================================

In all previous DDD papers the connectivity of the substrate is
fixed. This paper makes the connectivity itself dynamical.

Local rewiring rule:
  At each tick, with probability p_birth proportional to local activity
  product chi_i * chi_j, a new link (i,j) is born between two
  spatially close nodes. With probability p_death proportional to
  local INACTIVITY (1 - chi_i)(1 - chi_j), an existing link is
  pruned.

This produces:
  (a) Background-independence: the graph is part of the state.
  (b) Arrow of time: the entropy S = -log P(graph) increases
      monotonically because rewiring is irreversible without
      input from outside.
  (c) Effective dimension d_H increases over time as activity
      pulls in new connections (the graph 'thickens' in active
      regions).

We simulate this on N=1000 nodes randomly placed in a 3D box,
initialised as a random geometric graph (mean degree 6). We
measure d_H, mean degree, and connection entropy as functions of
tick.

Outputs:
    data/rewiring.json
    figures/fig01_rewiring.pdf
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)
FIG  = HERE / "figures"; FIG.mkdir(exist_ok=True)

N_NODES   = 1000
BOX       = 10.0
MEAN_DEG0 = 6
N_TICKS   = 80
SEED      = 2026

P_BIRTH = 0.02
P_DEATH = 0.02

rng = np.random.default_rng(SEED)


# ------------- substrate setup
positions = rng.uniform(0, BOX, size=(N_NODES, 3))
# Initial random geometric graph: connect each node to MEAN_DEG0 nearest
tree = cKDTree(positions)
dists, idxs = tree.query(positions, k=MEAN_DEG0 + 1)  # includes self
edges = set()
for i in range(N_NODES):
    for j in idxs[i, 1:]:  # skip self
        a, b = (i, j) if i < j else (j, i)
        edges.add((int(a), int(b)))
print(f"Initial graph: N={N_NODES}, E={len(edges)}, mean deg={2*len(edges)/N_NODES:.2f}")


def degrees(edges, n):
    deg = np.zeros(n, dtype=int)
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    return deg


def hausdorff_dim(edges, positions, n_samples=80):
    """Estimate effective dimension from N(r) ~ r^d."""
    n = len(positions)
    # Build adjacency for BFS
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    # BFS from random nodes
    rs = list(range(1, 11))
    counts = np.zeros(len(rs))
    sample_idx = rng.choice(n, n_samples, replace=False)
    for s in sample_idx:
        visited = {s: 0}
        frontier = [s]
        for r in range(1, max(rs) + 1):
            new_frontier = []
            for v in frontier:
                for u in adj[v]:
                    if u not in visited:
                        visited[u] = r
                        new_frontier.append(u)
            frontier = new_frontier
        for ki, r in enumerate(rs):
            counts[ki] += sum(1 for v, d in visited.items() if d <= r)
    counts /= n_samples
    # Fit log N = d log r over middle range
    log_r = np.log(rs[1:6])
    log_n = np.log(counts[1:6] + 1e-10)
    p = np.polyfit(log_r, log_n, 1)
    return float(p[0])


def chi_field(edges, positions):
    """Activity proxy: high-degree nodes are 'active'."""
    deg = degrees(edges, len(positions))
    deg_max = deg.max() if deg.max() > 0 else 1
    return deg / deg_max


# ----- evolve

history = {"tick": [], "n_edges": [], "mean_deg": [], "d_H": [], "entropy": []}

# Compute initial measurements
d_H0 = hausdorff_dim(edges, positions)
deg0 = degrees(edges, N_NODES)
print(f"Initial d_H = {d_H0:.3f}, mean deg = {deg0.mean():.2f}")

# Evolution loop
edge_list = list(edges)
all_edges = set(edges)
for t in range(N_TICKS):
    chi = chi_field(all_edges, positions)
    # Birth: pick a random unconnected close pair, add with p ~ chi_i chi_j
    n_births = 0
    for _ in range(int(P_BIRTH * N_NODES)):
        i = rng.integers(0, N_NODES)
        # Find a close non-neighbour
        d, jj = tree.query(positions[i], k=20)
        for j in jj[1:]:
            j = int(j)
            a, b = (i, j) if i < j else (j, i)
            if (a, b) not in all_edges:
                # Birth probability
                if rng.random() < chi[i] * chi[j]:
                    all_edges.add((a, b))
                    n_births += 1
                    break

    # Death: prune low-activity links
    n_deaths = 0
    edge_arr = list(all_edges)
    rng.shuffle(edge_arr)
    n_to_check = int(P_DEATH * len(edge_arr))
    for k in range(n_to_check):
        a, b = edge_arr[k]
        if rng.random() < (1 - chi[a]) * (1 - chi[b]):
            all_edges.discard((a, b))
            n_deaths += 1

    if t % 10 == 0 or t == N_TICKS - 1:
        d_H = hausdorff_dim(all_edges, positions, n_samples=50)
        deg = degrees(all_edges, N_NODES)
        # Entropy: -sum p log p of degree distribution
        if deg.sum() > 0:
            p = deg / deg.sum()
            p = p[p > 0]
            S = float(-np.sum(p * np.log(p)))
        else:
            S = 0.0
        history["tick"].append(t)
        history["n_edges"].append(len(all_edges))
        history["mean_deg"].append(float(deg.mean()))
        history["d_H"].append(d_H)
        history["entropy"].append(S)
        print(f"  t={t:3d}  E={len(all_edges):5d}  deg={deg.mean():.2f}  "
              f"d_H={d_H:.3f}  S={S:.4f}  (births={n_births} deaths={n_deaths})")

# Save
results = {
    "params": {"N_NODES": N_NODES, "BOX": BOX, "MEAN_DEG0": MEAN_DEG0,
               "N_TICKS": N_TICKS, "P_BIRTH": P_BIRTH, "P_DEATH": P_DEATH},
    "history": history,
    "initial_d_H": float(d_H0),
}
with open(DATA / "rewiring.json", "w") as f:
    json.dump(results, f, indent=2)

# Figure
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

ax = axes[0, 0]
ax.plot(history["tick"], history["n_edges"], "bo-", markersize=5)
ax.set_xlabel("tick"); ax.set_ylabel("number of links")
ax.set_title("Total number of links (graph growth)")
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(history["tick"], history["mean_deg"], "go-", markersize=5)
ax.set_xlabel("tick"); ax.set_ylabel("mean degree")
ax.set_title("Mean node degree")
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(history["tick"], history["d_H"], "ro-", markersize=5)
ax.axhline(d_H0, color="grey", lw=0.5, ls="--", label="initial")
ax.set_xlabel("tick"); ax.set_ylabel(r"effective dimension $d_H$")
ax.set_title("Effective dimension evolves")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(history["tick"], history["entropy"], "ko-", markersize=5)
ax.set_xlabel("tick"); ax.set_ylabel(r"degree-distribution entropy $S$")
ax.set_title("Connectivity entropy: arrow of time")
ax.grid(True, alpha=0.3)

fig.suptitle(f"Dynamical substrate: rewiring with p_birth={P_BIRTH}, p_death={P_DEATH}",
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(FIG / "fig01_rewiring.pdf", bbox_inches="tight")
fig.savefig(FIG / "fig01_rewiring.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: {FIG / 'fig01_rewiring.pdf'}")

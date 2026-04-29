# Paper I — Discrete Drainage Dynamics: Foundations

Manuscript and reproducible simulation code for **Paper I** of the DDD
series:

> *Discrete Drainage Dynamics: A Bounded, Conservative Local Rule for
> Newtonian-Like Potentials*

This paper defines a deterministic, parallel, two-step local rule on a
discrete graph and shows, by direct numerical simulation, that its
linearised steady state reproduces a $1/r$ static potential at the
percent level. The contribution of the paper is the rule, not the
$1/r$ law.

## Contents

```
paperI_foundations/
├── README.md                   ← This file
├── OUTLINE.md                  ← Working notes on structure
├── Makefile                    ← Build helpers
├── paper.tex                   ← LaTeX source
├── paper.pdf                   ← Compiled manuscript (9 pages)
├── references.bib              ← BibTeX bibliography
├── .gitignore
├── code/
│   ├── G_measure_standalone.py    Linearised Jacobi solver for 1/r
│   │                              (Table 2 in the paper)
│   ├── v7_drainage_rule.py        Explicit rate-limited rule (full
│   │                              non-linear simulation, both regimes)
│   └── spectral_dimension.py      Heat-kernel d_s(t) on cubic and BCC
│                                  (Table 3 in the paper)
├── data/                       Numerical outputs (JSON)
└── figures/                    Generated figures (PDF + PNG)
```

## What this paper claims

Three claims (Section 1 of the paper):

1. **The rule.** A deterministic, parallel, two-step local update on a
   discrete graph: a Link Update computes desired directional fluxes
   $F^{\rm des}_{i \to j} = \alpha (R_i - R_j)_+$, a Node Update
   advances the reserve $R_i$, and a per-node rate-limiter
   $\beta_i \in [0, 1]$ ensures the reserve stays non-negative
   without clipping.
2. **The $1/r$ result.** Run as a stationary linearised simulation on
   cubic lattices of side $L \in \{64, 128, 160\}$, the rule produces
   a radial profile compatible with $1/r$, with a coefficient that
   matches the discrete continuum prediction $M/(4\pi r)$ at the
   percent level.
3. **Conservation.** The internal flux-exchange rule is exactly
   conservative: $\beta_i$ scales outflows and external sinks in the
   same proportion, so no mass is created or destroyed by the
   internal rule. External sinks enter as explicit bookkeeping.

## What this paper does not claim

- Newton's constant $G$ in physical units (calibrated, not derived).
- General relativity or any post-Newtonian observable.
- Any property of the Standard Model.
- Uniqueness of the rule.
- Novelty of the $1/r$ result itself: any 3D discrete diffusion
  produces it. The novelty is the rule, including the rate-limited
  finite-resource structure that distinguishes it from a plain
  discrete heat equation.

## Quick start

Reproduce the numerical results with pure-numpy scripts (no scipy
required):

```bash
# Linearised steady-state Jacobi solver (fast):
python code/G_measure_standalone.py --L 128 --n-iter 10000

# Explicit rate-limited rule, linear regime (slower but exact):
python code/v7_drainage_rule.py --L 24 --n_ticks 5000 --E0 0.3

# Explicit rate-limited rule, saturated non-linear regime:
python code/v7_drainage_rule.py --L 24 --n_ticks 5000 --E0 5.0

# Spectral-dimension comparison cubic vs BCC:
python code/spectral_dimension.py
```

The Jacobi solver and the explicit rate-limited rule agree to seven
significant digits on $A_{\rm fit}$ in the linear regime
($\beta_i = 1$ everywhere), validating the equivalence stated in
Sec. 6.1 of the paper.

## Build the manuscript

```bash
pdflatex paper
bibtex paper
pdflatex paper
pdflatex paper
```

## Required LaTeX packages

Standard TeX Live: amsmath, amssymb, amsthm, graphicx, hyperref,
xcolor, booktabs, authblk, geometry, cite.

## Reproducibility

All scripts are pure-numpy and depend on no external simulation
framework. The Jacobi steady-state results in Tables 2 and 3 of the
paper are deterministic given the lattice size $L$, the source
strength $M$, and the iteration count.

## License

- Simulation code in `code/`: MIT.
- Manuscript text and figures: standard arXiv terms.

## Contact

Stanislas Dewavrin — Independent Researcher — `dewavrin.iphone@gmail.com`

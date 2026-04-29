# Paper I — Outline

**Working title.** *Discrete Drainage Dynamics: Foundations of a Procedural Lattice Framework*

**Author.** Stanislas Dewavrin (Independent Researcher)

**Companion papers.**
- Paper II (in preparation): *Emergent Gravity from Discrete Drainage Dynamics*
- Paper III (in preparation): *Cosmogenesis from a Discrete Lattice* (DESI DR2)

## Strategic position

Paper I is the **foundation pillar** of the DDD series. It must:

1. Commit to a single, sharp ontology — charges on links, nodes as local operators — and never depart from it.
2. Establish the language discipline that prevents micro/macro confusion (a frequent failure mode of emergent-spacetime papers).
3. Derive the central interface identity $T^2 + I^2 = \chi$ from the local rule, not postulate it.
4. Recover the kinematics of special relativity in the regime $\chi = 1$ as an internal-consistency identity, not a prediction.
5. Defer everything else to companion papers, with explicit epistemic labels.

Paper I makes **zero new physical predictions**. Its job is to make Papers II and III possible.

## Section plan

### 1. Introduction
- Motivation: emergent-spacetime programs (Verlinde, Wolfram, causal sets, loop QG) and what DDD adds.
- The core claim: a procedural, local, discrete substrate from which spacetime, gravity and matter emerge as macroscopic regimes.
- Structure of the paper.
- What is **not** in this paper.

### 2. Scales and language: a notational discipline
- Eight levels (ontological / microscopic / local-rule / simulational / mesoscopic / macroscopic-emergent / interpretive / programmatic).
- Notation table separating micro objects (nodes, links, ticks, charges) from emergent objects (positions, instants, distances, fields).
- Convention: tick $t$ vs emergent time $\tau$; topological neighbourhood $\mathcal{N}(i)$ vs spatial neighbourhood; discrete gradient $\hat\nabla$ vs continuous $\nabla$.

### 3. The discrete substrate
- Graph $G = (V, E)$ as primitive object. No embedding space.
- Connectivity (random geometric, mean degree $\bar{D} \simeq 6$ — to be tuned in companion papers).
- **Charges on links.** Each link $\ell_{ij}$ carries a charge $q_{ij} \in \mathbb{Z}$ (or $\mathbb{Z} \times U(1)$ if oriented sector activated — flagged for Paper VI).
- Node $i$ as local operator: reads incident link charges, applies routing rule, writes new charges back.
- Tick: discrete update unit. Time emerges from tick counting.

### 4. Local update rule
- Family-A rule (node-centred): node reads in, applies rule, writes out.
- Determinism in asymmetric configurations; weighted draw at strict symmetries.
- The draw at symmetric bifurcations is **forced** by locality + indivisibility, not postulated. (This is the seed for Paper VII on quantum phenomena.)
- Bounded propagation: at most one node per tick → emergent maximal speed $c$.

### 5. Conservation
- Local: charge conservation at every node, every tick.
- Global: total charge invariant.
- Effective "energy" as a derived quantity; not all of total charge is "free" — bound configurations carry localized excess.
- Numerical check (from v2.md Tableau 6a): zero-sum to machine precision; clip on $R_{\min}$ documented as the only structural source of non-conservation.

### 6. Emergent node variables
- Local reserve $R_i$ defined as a functional of incident link charges: $R_i = \Lambda(\{q_{ij}\}_{j \in \mathcal{N}(i)})$ — typically a sum or weighted sum.
- Normalized capacity $\chi_i \in [0, 1]$, deficit $\delta_i = R_0 - R_i$.
- Why $R_i$ is derived, not fundamental: this is the link-charge ontology in operation.

### 7. Kinematic sector and the interface identity
- Definition of $T_i$ (translational component) as the net positive directed flux through node $i$.
- Definition of $I_i$ (internal component) as the residual local activity.
- Angular parametrization $T = \sqrt{\chi_i} \sin\phi$, $I = \sqrt{\chi_i} \cos\phi$.
- The identity $T^2 + I^2 = \chi$ as an **algebraic identity of the definition**, not a dynamical claim.
- Numerical check: identity preserved to machine precision under drainage (v2.md Tableau 8b).
- Recovery of SR cinematics in the $\chi = 1$ regime: $\beta = \sin\phi$, $1/\gamma = \cos\phi$, $E^2 = p^2 c^2 + m^2 c^4$ — all by construction, status of internal consistency, not prediction.

### 8. Direction emergence
- Local retention $C_i$ as a non-decreasing function of $R_i$.
- Asymmetry vector $\mathbf{A}_i = \sum_{j \sim i}(C_j - C_i) \mathbf{e}_{i \to j}$.
- Direction $\mathbf{n}_i = \mathbf{A}_i / \|\mathbf{A}_i\|$ when defined; symmetry locus $\|\mathbf{A}_i\| = 0$ as honest indeterminacy (no privileged direction, by symmetry).
- Numerical validation (v2.md Tableau): exact recovery of expected directions on linear, radial, diagonal gradients; documented discretization residual on radial profile (~6° at first off-axis ring, decaying with $r$).

### 9. Inertial relaxation
- $d\phi/dt = (\phi_{\rm eq} - \phi)/\tau_{\rm ret}$ as effective dynamics of $\phi$.
- $\phi_{\rm eq}$ from balance: $\dot\phi = \lambda \cos\phi - \mu \sin\phi$ → $\phi_\infty = \arctan(\lambda/\mu)$.
- Numerical validation (v2.md Tableau 3a, 3b): convergence in $\tau_{\rm ret} \cdot \ln(1/\varepsilon)$ ticks; equilibrium recovered to $10^{-6}$.
- Reading: inertia as **time of lattice reorganization**, not as a property of an object. Mass deferred to Paper V.

### 10. Numerical coherence summary
- Master table summarizing the four internal-consistency checks of this paper:
  - Identity $T^2 + I^2 = \chi$
  - Conservation (zero-sum + clip)
  - Direction emergence
  - Inertial relaxation
- Reproducible from `code/` directory; status: **internal consistency**, not external validation.

### 11. Discussion
- What this paper establishes: the substrate, the rules, the algebra, the internal coherence.
- What this paper does **not** claim: emergent gravity (Paper II), cosmological dynamics (Paper III), Newton's constant (Paper II.5), particles (Paper V), electromagnetism (Paper VI), quantum phenomena (Paper VII), unification (Paper VIII).
- The methodological status of simulation in a procedural theory.

### 12. Conclusion
- Recap of the five committed claims.
- Forward references to Papers II, III.

## Numerical material (already available in v2.md, to be ported into reproducible Python)

| Check | v2.md table | Python script (target) |
|---|---|---|
| Conservation zero-sum + clip | Tableau 6a, 6b | `code/01_conservation.py` |
| Identity $T^2+I^2=\chi$ | Tableau 8a, 8b, 8c | `code/02_interface_identity.py` |
| Direction emergence | Tableau 9.5 (4 cases) | `code/03_direction_emergence.py` |
| Inertial relaxation | Tableau 3a, 3b | `code/04_inertial_relaxation.py` |
| SR recovery at $\chi=1$ | Tableau 8.1 | `code/05_sr_recovery.py` |
| Master figure | (new) | `code/06_master_figure.py` |

## Forbidden territory in Paper I

The following must **not** appear in this paper as established results, only as forward references with explicit "Paper X, in preparation":

- Newton's constant numerical value
- Schwarzschild profile beyond the weak-field identification
- Photon deflection $\Delta\theta$
- DESI DR2 comparison
- Maxwell's equations
- Spin-1/2
- Particle masses
- Born rule
- Hubble dipole

## Length target

- 12–15 pages including figures and bibliography.
- Tone: foundational, pedagogical where useful, terse otherwise. No grandiloquence.

"""
Paper I --- Lorentz time-delay confrontation with Fermi-LAT GRBs
====================================================================

DDD prediction (Paper I, Section "Causal structure"):
  Below the saturation threshold alpha*|nabla delta| < F_max, the
  rule is parabolic with dispersion omega(k) = -i*alpha*k^2.
  In the saturated regime, the lightcone propagates at constant
  c_latt with a quadratic-in-k residual:
     Delta v / c  =  alpha * (k * ell_P)^2
                  ~  alpha * (E_gamma / E_Planck)^2

  with E_Planck ~ 1.22e19 GeV, alpha ~ 0.15 (Paper I).

Existing Fermi-LAT bounds on Lorentz invariance violation come from
gamma-ray burst time-delay analyses. Vasileiou+ 2013 give the
strongest bounds, parametrised in terms of E_QG (the energy scale
where LIV becomes O(1)):
   E_QG_1  >  7.6 * E_Planck   (linear, n=1)
   E_QG_2  >  1.2 * 10^11 GeV  (quadratic, n=2, weaker)

In the DDD framework, the LIV is purely quadratic (n=2) because the
linear regime is parabolic, not just dispersive: there is no n=1
contribution. The relevant bound is therefore the quadratic one:
   E_QG_2 (DDD)  <=>  E_Planck / sqrt(alpha) ~ 2.6 * E_Planck

The Vasileiou bound E_QG_2 > 1.2e11 GeV translates into a constraint
on the DDD coupling alpha. We compute the effective bound here.

We do NOT have access to per-GRB photon time-delay data without
fetching multiple Fermi-LAT spectra; instead we use the published
bounds directly, which already encapsulate the analysis of multiple
GRBs.

Outputs:
    data/lorentz_grb.json
    figures/lorentz_grb.pdf
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)
FIG  = HERE / "figures"; FIG.mkdir(exist_ok=True)

# Constants
E_PLANCK_GEV = 1.22e19       # GeV, Planck energy
ALPHA_DDD    = 0.15          # Paper I XY/flux coupling

# Vasileiou et al 2013 (Fermi-LAT GRB) published 95% CL bounds:
E_QG_1_BOUND = 7.6 * E_PLANCK_GEV  # n=1, linear
E_QG_2_BOUND = 1.2e11              # n=2, quadratic (in GeV)

# DDD prediction: only n=2 contribution, with effective scale
# E_QG_2_DDD = E_Planck / sqrt(alpha)
E_QG_2_DDD = E_PLANCK_GEV / np.sqrt(ALPHA_DDD)

print("=" * 70)
print("DDD vs Fermi-LAT GRB Lorentz invariance bounds")
print("=" * 70)
print(f"\nDDD parameters:")
print(f"  alpha        = {ALPHA_DDD}")
print(f"  E_Planck     = {E_PLANCK_GEV:.2e} GeV")
print(f"  E_QG_2 (DDD) = E_Planck / sqrt(alpha) = {E_QG_2_DDD:.2e} GeV")
print(f"\nFermi-LAT GRB bounds (Vasileiou+ 2013):")
print(f"  E_QG_1 > {E_QG_1_BOUND:.2e} GeV  (n=1, linear) [DDD predicts NO n=1 term]")
print(f"  E_QG_2 > {E_QG_2_BOUND:.2e} GeV  (n=2, quadratic)")

# Compatibility test
ratio = E_QG_2_DDD / E_QG_2_BOUND
print(f"\nCompatibility:")
print(f"  E_QG_2 (DDD) / E_QG_2 (bound) = {ratio:.2e}")
if ratio > 1:
    print(f"  -> DDD predicts E_QG_2 ABOVE the lower bound by {ratio:.2e}")
    print(f"  -> DDD is COMPATIBLE with Fermi-LAT bounds")
else:
    print(f"  -> DDD predicts E_QG_2 BELOW the lower bound -> REFUTED")

# Compute Delta v / c at canonical energies
energies_GeV = np.logspace(0, 11, 200)
delta_v_c = ALPHA_DDD * (energies_GeV / E_PLANCK_GEV) ** 2

# Compare with quadratic LIV bound at the same energies
# Standard QG: Delta v / c = (E / E_QG_2_BOUND)^2 (boundary)
delta_v_c_bound = (energies_GeV / E_QG_2_BOUND) ** 2

# At GRB-relevant energy ~ 30 GeV:
E_GRB = 30.0  # GeV (typical Fermi-LAT GRB photon)
dvc_DDD = ALPHA_DDD * (E_GRB / E_PLANCK_GEV) ** 2
dvc_bound = (E_GRB / E_QG_2_BOUND) ** 2
print(f"\nAt E = {E_GRB} GeV (Fermi-LAT GRB scale):")
print(f"  DDD prediction:       Delta v / c = {dvc_DDD:.2e}")
print(f"  Vasileiou n=2 bound:  Delta v / c < {dvc_bound:.2e}")
print(f"  ratio bound/DDD:      {dvc_bound/dvc_DDD:.2e}")

# Verdict
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
print(f"DDD predicts NO linear-in-E LIV term (n=1 absent by parabolic")
print(f"linear regime). The strongest Fermi-LAT bound (n=1) is therefore")
print(f"automatically satisfied.")
print(f"")
print(f"DDD predicts a quadratic-in-E LIV term at the level of")
print(f"Delta v/c ~ alpha * (E/E_Planck)^2 = {dvc_DDD:.1e} at 30 GeV.")
print(f"The Fermi-LAT n=2 bound is {dvc_bound/dvc_DDD:.1e}x larger,")
print(f"i.e. {np.log10(dvc_bound/dvc_DDD):.1f} orders of magnitude above")
print(f"the DDD prediction. The framework is COMFORTABLY compatible.")

# Save
results = {
    "DDD_alpha":    ALPHA_DDD,
    "E_Planck_GeV": E_PLANCK_GEV,
    "E_QG_2_DDD_GeV":   float(E_QG_2_DDD),
    "Vasileiou_E_QG_1_lower_GeV":  E_QG_1_BOUND,
    "Vasileiou_E_QG_2_lower_GeV":  E_QG_2_BOUND,
    "ratio_DDD_to_bound": float(ratio),
    "compatibility":    "DDD is compatible (E_QG_2_DDD above bound)",
    "delta_v_c_at_30GeV_DDD":   float(dvc_DDD),
    "delta_v_c_at_30GeV_bound": float(dvc_bound),
    "n1_term":          "DDD predicts none (parabolic linear regime)",
    "verdict":          ("Compatible with Fermi-LAT bounds at all "
                          "current energies; falsifiable at 30 GeV "
                          "if Delta v/c reaches ~alpha*(E/E_Planck)^2 "
                          "level."),
}
with open(DATA / "lorentz_grb.json", "w") as f:
    json.dump(results, f, indent=2)

# Figure
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.loglog(energies_GeV, delta_v_c, "b-", lw=2,
           label=r"DDD: $\Delta v/c = \alpha (E/E_P)^2$")
ax.loglog(energies_GeV, delta_v_c_bound, "r--", lw=1.5,
           label=r"Vasileiou+ 2013 (95% CL)")
ax.axvline(E_GRB, color="grey", lw=0.5, ls=":",
            label=f"E = {E_GRB} GeV (GRB)")
ax.fill_between(energies_GeV, delta_v_c_bound,
                  np.ones_like(delta_v_c_bound), color="red",
                  alpha=0.15, label="excluded by Fermi-LAT")
ax.scatter([E_GRB], [dvc_DDD], c="blue", s=120, marker="*", zorder=5,
            label=fr"DDD prediction at GRB: ${dvc_DDD:.0e}$")
ax.set_xlabel("photon energy E (GeV)")
ax.set_ylabel(r"$\Delta v / c$")
ax.set_title("Lorentz invariance: DDD prediction vs Fermi-LAT GRB bound\n"
              f"DDD is compatible by {dvc_bound/dvc_DDD:.0e}x")
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.3, which="both")
ax.set_ylim(1e-30, 1)
fig.tight_layout()
fig.savefig(FIG / "lorentz_grb.pdf", bbox_inches="tight")
fig.savefig(FIG / "lorentz_grb.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: {FIG / 'lorentz_grb.pdf'}")

"""Spectral dimension d_s(t) of cubic Laplacian, multiple L."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)
FIG  = HERE / "figures"; FIG.mkdir(exist_ok=True)


def cubic_spectrum(L):
    n = np.arange(L)
    kx, ky, kz = np.meshgrid(2*np.pi*n/L, 2*np.pi*n/L, 2*np.pi*n/L, indexing='ij')
    return (2*(3 - np.cos(kx) - np.cos(ky) - np.cos(kz))).flatten()


def bcc_spectrum(L):
    n = np.arange(L)
    kx, ky, kz = np.meshgrid(2*np.pi*n/L, 2*np.pi*n/L, 2*np.pi*n/L, indexing='ij')
    return (8*(1 - np.cos(kx/2)*np.cos(ky/2)*np.cos(kz/2))).flatten()


def d_s_curve(spectrum, ts):
    spec = spectrum[spectrum > 1e-12]
    Ps = np.array([np.mean(np.exp(-spec * t)) for t in ts])
    log_t = np.log(ts)
    log_P = np.log(Ps)
    return Ps, -2 * np.gradient(log_P, log_t)


def main():
    Ls = [32, 64, 128]
    ts = np.logspace(-1, 3.5, 80)
    out = {}
    target = 3 + 1/(2*np.pi)

    for L in Ls:
        print("L =", L, "(", L**3, "modes )")
        sc = cubic_spectrum(L)
        sb = bcc_spectrum(L)
        Pc, dc = d_s_curve(sc, ts)
        Pb, db = d_s_curve(sb, ts)
        t_low = 5.0
        t_high = (L/8.0)**2
        m = (ts > t_low) & (ts < t_high)
        if m.sum() < 3:
            m = (ts > t_low) & (ts < L*L/4.0)
        d_min_c = float(dc[m].min())
        d_min_b = float(db[m].min())
        d_avg_c = float(dc[m].mean())
        d_avg_b = float(db[m].mean())
        print("  cubic plateau t in [", t_low, ",", round(t_high,1), "]")
        print("    d_s min  =", round(d_min_c, 5))
        print("    d_s mean =", round(d_avg_c, 5), "+/-", round(float(dc[m].std()), 5))
        print("  bcc plateau:")
        print("    d_s min  =", round(d_min_b, 5))
        print("    d_s mean =", round(d_avg_b, 5))
        print("  target 3+1/(2pi) =", round(target, 5))
        out[str(L)] = {
            "ts": ts.tolist(),
            "Ps_cubic": Pc.tolist(), "ds_cubic": dc.tolist(),
            "Ps_bcc": Pb.tolist(), "ds_bcc": db.tolist(),
            "ds_min_cubic": d_min_c, "ds_min_bcc": d_min_b,
            "ds_avg_cubic": d_avg_c, "ds_avg_bcc": d_avg_b,
            "t_window": [t_low, float(t_high)],
        }

    print("\n=== Convergence in L ===")
    print("  L     ds_min_cubic   ds_min_bcc")
    for L in Ls:
        r = out[str(L)]
        print("  ", L, "   ", round(r["ds_min_cubic"], 5), "   ", round(r["ds_min_bcc"], 5))
    print("  target Newton 3.00000  ; topological 3+1/(2pi) =", round(target,5))

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8,5))
        for L in Ls:
            r = out[str(L)]
            tarr = np.array(r["ts"])
            sel = tarr < L*L
            ax.plot(tarr[sel], np.array(r["ds_cubic"])[sel], lw=1.5, label="cubic L="+str(L))
        ax.axhline(3.0, color='k', ls='--', lw=1, label='d=3 Newton')
        ax.axhline(target, color='r', ls=':', lw=1, label='3+1/(2pi)')
        ax.set_xscale('log'); ax.set_xlabel('t'); ax.set_ylabel('d_s(t)')
        ax.set_ylim(2.5, 4.0); ax.legend(fontsize=9); ax.grid(alpha=0.3)
        ax.set_title('Spectral dimension of cubic Laplacian')
        fig.tight_layout()
        fig.savefig(FIG / "fig_spectral_dim.png", dpi=160)
        plt.close(fig)
        print("\nFigure saved")
    except Exception as e:
        print("plot failed:", e)

    with open(DATA / "spectral_dimension.json", "w") as f:
        json.dump({"by_L": out, "target": target}, f, indent=2)
    print("Data saved")


if __name__ == "__main__":
    main()

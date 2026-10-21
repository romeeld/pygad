#!/usr/bin/env python
"""
Benchmark pygad.analysis.vpfit.fit_profiles_sat on synthetic spectra with
known components (single lines from unsaturated to strongly saturated, and
blends), to measure biases in the recovered logN, b and line centres.

Each case is a noisy spectrum of +-600 km/s (exact Voigt profiles), fitted
with the same settings as scripts/random_los_spectra.py.  Every true component is matched to the
nearest fitted component; extra fitted components with logN > 12.5 are
counted as spurious.

    python tests/vpfit_benchmark.py [--nproc 16] [--seeds 3] [-o results.json]

This is a diagnostic, not a pass/fail test.
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

C_KMS = 299792.458
Z = 2.0
SNR = 50.0
DV = 2.5
VMAX = 600.0

# (name, line, [(logN, b, v), ...])
CASES = (
    [("HI %.1f b%d" % (N, b), "H1215", [(N, b, 0.0)])
     for N in (13.0, 14.0, 15.0, 16.0, 17.0, 18.0) for b in (15, 30)]
    + [
        ("HI blend 14.5+13.5 dv30", "H1215", [(14.5, 20, -15.0), (13.5, 20, 15.0)]),
        ("HI blend 15.5+15.0 dv40", "H1215", [(15.5, 25, -20.0), (15.0, 25, 20.0)]),
        ("HI blend 16.0+14.0 dv60", "H1215", [(16.0, 25, -30.0), (14.0, 20, 30.0)]),
        ("HI triple 14/14.5/13.5", "H1215", [(14.0, 20, -60.0), (14.5, 25, 0.0), (13.5, 15, 50.0)]),
        ("CIV 13.0 b8", "CIV1548", [(13.0, 8, 0.0)]),
        ("CIV 14.0 b8", "CIV1548", [(14.0, 8, 0.0)]),
        ("CIV blend 13.5+13.0 dv20", "CIV1548", [(13.5, 8, -10.0), (13.0, 10, 10.0)]),
    ]
)


def run_case(job):
    name, line, comps, seed = job
    import pygad as pg
    from pygad.analysis import vpfit
    from pygad.analysis.tau_lookup import model_tau_fast

    pg.environment.verbose = pg.environment.VERBOSE_QUIET
    A = pg.analysis.absorption_spectra
    lam0 = float(pg.UnitScalar(A.lines[line]["l"], "Angstrom"))
    lc = lam0 * (1.0 + Z)
    v = np.arange(-VMAX, VMAX, DV)
    l = lc * (1.0 + v / C_KMS)
    p_true = np.ravel([[N, b, lc * (1.0 + vc / C_KMS)] for N, b, vc in comps])
    # exact profiles, not the fitter's lookup table, so its errors show up
    tau = vpfit.model_tau(line, p_true, l, "Voigt")
    rng = np.random.default_rng(seed)
    noise = np.full_like(v, 1.0 / SNR)
    flux = np.exp(-tau) + rng.normal(0.0, 1.0 / SNR, len(v))

    if A.lines[line]["ion"] == "HI":
        logN_bounds = [12.0, 19.0]
    else:
        logN_bounds = [10.0, 17.0]
    b_therm = float(A.thermal_b_param(line, pg.UnitScalar(1e4, "K"), "km/s"))
    b_bounds = [0.25 * b_therm, 150.0]

    t0 = time.time()
    rl, ri = vpfit.find_regions(l, flux, noise, N_sigma=10.0)
    if len(rl) == 0:
        ll = {"N": np.array([]), "b": np.array([]), "l": np.array([])}
    else:
        ll = vpfit.fit_profiles_sat(line, l, flux, noise, rl, ri, chisq_lim=2.0,
                                    chisq_factor=0.95, N_sigma_constr=3.0, max_lines=12,
                                    mode="Voigt", logN_bounds=logN_bounds,
                                    b_bounds=b_bounds)
    dt = time.time() - t0

    N_fit = np.asarray(ll["N"], dtype=float)
    b_fit = np.asarray(ll["b"], dtype=float)
    v_fit = (np.asarray(ll["l"], dtype=float) / lc - 1.0) * C_KMS
    model = np.exp(-model_tau_fast(line, np.ravel(np.c_[N_fit, b_fit, ll["l"]]), l, "Voigt"))
    chisq = float(np.mean(((flux - model) / noise) ** 2))

    matches, used = [], set()
    for N, b, vc in sorted(comps, key=lambda c: -c[0]):
        cand = [i for i in range(len(N_fit)) if i not in used]
        if not cand:
            matches.append(None)
            continue
        i = min(cand, key=lambda i: abs(v_fit[i] - vc))
        if abs(v_fit[i] - vc) > max(b, 15.0):
            matches.append(None)
            continue
        used.add(i)
        matches.append({"N": N, "b": b, "v": vc, "dN": float(N_fit[i] - N),
                        "db_frac": float(b_fit[i] / b - 1.0), "dv": float(v_fit[i] - vc)})
    n_spurious = int(sum(1 for i in range(len(N_fit)) if i not in used and N_fit[i] > 12.5))
    # total column within +-150 km/s of the true components
    vmin, vmax = min(c[2] for c in comps) - 150, max(c[2] for c in comps) + 150
    sel = (v_fit > vmin) & (v_fit < vmax)
    logNtot_fit = float(np.log10(np.sum(10 ** N_fit[sel]))) if sel.any() else 0.0
    logNtot_true = float(np.log10(np.sum([10 ** c[0] for c in comps])))
    return {"name": name, "seed": seed, "time": dt, "chisq": chisq, "n_fit": len(N_fit),
            "n_spurious": n_spurious, "dNtot": logNtot_fit - logNtot_true,
            "matches": matches}


def summarise(results):
    print("%-26s %5s %7s %7s %7s %7s %5s %6s %6s" % (
        "case", "found", "dN", "db/b", "dv", "dNtot", "spur", "chi2", "t[s]"))
    for name, *_ in CASES:
        rs = [r for r in results if r["name"] == name]
        m = [x for r in rs for x in r["matches"][:1]]   # strongest component
        found = [x for x in m if x is not None]
        med = lambda k: np.median([x[k] for x in found]) if found else np.nan
        print("%-26s %2d/%-2d %+7.3f %+7.3f %+7.2f %+7.3f %5.1f %6.2f %6.1f" % (
            name, len(found), len(m), med("dN"), med("db_frac"), med("dv"),
            np.median([r["dNtot"] for r in rs]),
            np.mean([r["n_spurious"] for r in rs]),
            np.median([r["chisq"] for r in rs]),
            np.median([r["time"] for r in rs])))
    allm = [x for r in results for x in r["matches"]]
    ok = [x for x in allm if x is not None]
    print("\nall components: %d/%d matched; median |dN| %.3f, median db/b %+.3f, "
          "spurious/spectrum %.2f, total time %.0f s" % (
              len(ok), len(allm), np.median([abs(x["dN"]) for x in ok]),
              np.median([x["db_frac"] for x in ok]),
              np.mean([r["n_spurious"] for r in results]),
              sum(r["time"] for r in results)))
    print("(dN, db/b, dv: median over seeds for the strongest component; "
          "dNtot: total log N within +-150 km/s)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nproc", type=int, default=8)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("-o", "--output", help="write all results as JSON")
    args = p.parse_args()
    jobs = [(name, line, comps, seed) for name, line, comps in CASES
            for seed in range(args.seeds)]
    with ProcessPoolExecutor(args.nproc) as ex:
        results = list(ex.map(run_case, jobs))
    summarise(results)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()

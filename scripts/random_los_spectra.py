#!/usr/bin/env python
"""
Generate mock absorption spectra along random lines of sight (LOS) through a
periodic simulation volume, and optionally fit Voigt profiles to them.

For each requested line (e.g. H1215, CIV1548, OVI1031) a spectrum spanning the
full box in velocity is computed for every LOS with
pygad.analysis.absorption_spectra (all LOS for a line in one OpenMP-parallel C
call).  The box is periodic along the LOS, so the optical depth is computed over
a window several box lengths wide (enough to contain the fastest gas) and folded
back into [0, vbox]; nothing that scatters out of the
box via peculiar velocities or line wings is lost.  An optional line spread
function and Gaussian noise are then applied.

Voigt profiles are fitted with pygad.analysis.vpfit (find_regions +
fit_profiles_sat).  The spectrum is rolled to start at its least-absorbed
pixel before fitting, so no feature is split across the box edge, and fitted
line centres are mapped back into the box afterwards.

Output layout (OUTDIR defaults to ./spectra_<snapshot name>):

    OUTDIR/run_info.json            arguments + derived quantities of the run
    OUTDIR/los.txt                  LOS index and transverse positions
    OUTDIR/<line>/los_<NNNNN>.h5    one spectrum per line and LOS
    OUTDIR/lines_<line>.h5          catalogue of all fitted components (--fit)
    OUTDIR/spectra.png              overview plot (--plot N)

The per-spectrum HDF5 files use the same dataset names as
pygad.analysis.write_spectrum / write_line_list, so they can be read back by
pygad.analysis.fit_profiles(line, spectrum_file=...).

Examples
--------
    # 100 random LOS for HI, CIV and OVI, then fit them on 16 cores
    python random_los_spectra.py snap_m25n256_078.hdf5 -n 100 \\
        --lines H1215 CIV1548 OVI1031 --fit --nproc 16 --plot 5

    # re-fit spectra that already exist, without touching the snapshot
    python random_los_spectra.py snap_m25n256_078.hdf5 --lines CIV1548 \\
        --fit-only --overwrite-fits --nproc 16

Simba / GIZMO snapshots
-----------------------
pygad's default config/gadget.cfg does not describe GIZMO output: it assumes
`Metallicity` holds element *masses* in the order He,C,Mg,O,Fe,Si,H,...
(GIZMO stores mass *fractions* Z,He,C,N,O,Ne,Mg,Si,S,Ca,Fe) and that
InternalEnergy is in units of a*VELOCITY**2 (GIZMO's is physical (km/s)^2).
Left alone this gives garbage element masses, nH, temperatures and hence ion
columns.
By default (--gizmo check) the script checks X_H, X_He, metallicity, T and
the ion-table blocks after loading and stops with an explanation if they look
wrong.  With --gizmo fix the affected blocks of a GIZMO snapshot are instead
corrected in memory before anything is derived from them (works with the stock
config).
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np

C_KMS = 299792.458
GIZMO_METALS = ["Z", "He", "C", "N", "O", "Ne", "Mg", "Si", "S", "Ca", "Fe"]
AXIS_NAMES = {"x": 0, "y": 1, "z": 2}

_T0 = time.time()


def log(msg):
    print("[%7.1fs] %s" % (time.time() - _T0, msg), flush=True)


# ---------------------------------------------------------------------------
# Snapshot preparation
# ---------------------------------------------------------------------------

def is_gizmo_snapshot(path):
    """GIZMO/Simba: 11-column Metallicity block holding mass fractions."""
    with h5py.File(path, "r") as f:
        if "PartType0/Metallicity" not in f:
            return False
        Z = f["PartType0/Metallicity"]
        if Z.ndim != 2 or Z.shape[1] != len(GIZMO_METALS):
            return False
        He = Z[: min(len(Z), 100000), 1]
    return 0.2 < np.median(He) < 0.4


def fix_gizmo_blocks(s):
    """
    Overwrite the gas blocks that pygad's default gadget.cfg misreads for
    GIZMO snapshots.  Must be called before anything depending on them (temp,
    nH, ion masses, ...) is derived.
    """
    import pygad as pg

    g = s.gas
    # InternalEnergy is physical specific energy in (km/s)^2
    u = g["u"]
    g["u"] = pg.UnitArr(np.asarray(u, dtype=np.float64), "km**2/s**2")

    Z = np.asarray(g["Z"], dtype=np.float64)
    mass = g["mass"]
    m = np.asarray(mass, dtype=np.float64)
    frac = {el: Z[:, i] for i, el in enumerate(GIZMO_METALS)}
    frac["H"] = 1.0 - frac["Z"] - frac["He"]
    # pygad needs the same shape as the 'Z' block, i.e. the first 11 names
    names = pg.gadget.elements[: Z.shape[1]]
    missing = [el for el in names if el not in frac]
    if missing:
        raise RuntimeError("cannot map GIZMO metals onto pygad elements %s" % missing)
    g["elements"]
    g["elements"] = pg.UnitArr(np.column_stack([frac[el] * m for el in names]), mass.units)
    for el in names:
        g[el]
        g[el] = pg.UnitArr(frac[el] * m, mass.units)
    g["metallicity"]
    g["metallicity"] = pg.UnitArr(frac["Z"], "1")


def check_gas_blocks(s, line_names):
    """
    Sanity-check the derived gas blocks that the spectra depend on and stop
    with an explanation if the pygad config does not match the snapshot.
    """
    import pygad as pg
    g = s.gas
    sub = g[:: max(1, len(g) // 200000)]
    m = np.asarray(sub["mass"], dtype=np.float64)
    problems = []
    XH = np.median(np.asarray(sub["H"], dtype=np.float64) / m)
    XHe = np.median(np.asarray(sub["He"], dtype=np.float64) / m)
    if not 0.6 < XH < 0.8:
        problems.append("median H mass fraction is %.3g (expected ~0.75): the 'elements' "
                        "order in gadget.cfg does not match the snapshot's Metallicity "
                        "columns (GIZMO/Simba: Z,He,C,N,O,Ne,Mg,Si,S,Ca,Fe)" % XH)
    if not 0.2 < XHe < 0.3:
        problems.append("median He mass fraction is %.3g (expected ~0.25)" % XHe)
    Z = np.asarray(sub["metallicity"], dtype=np.float64)
    if np.median(Z) > 0.1:
        problems.append("median metallicity is %.3g: Metallicity is probably read as "
                        "masses instead of mass fractions (Z units in gadget.cfg)" % np.median(Z))
    T = np.median(np.asarray(sub["temp"].in_units_of("K", subs=s)))
    if not 3e3 < T < 1e6:
        problems.append("median gas temperature is %.3g K: check the units of U "
                        "(InternalEnergy) in gadget.cfg" % T)
    ions = {pg.analysis.absorption_spectra.lines[ln]["ion"] for ln in line_names}
    for ion in sorted(ions):
        if ion not in g.available_blocks():
            problems.append("ion block '%s' is not available; is [iontable] tabledir in "
                            "derived.cfg (currently %s) correct?"
                            % (ion, pg.snapshot.derived.iontable["tabledir"]))
    if problems:
        raise SystemExit("The gas blocks look wrong for this snapshot:\n  - "
                         + "\n  - ".join(problems)
                         + "\nFix the pygad config (./ or ~/.config/pygad/*.cfg), or rerun "
                           "with --gizmo fix to correct the blocks in memory.")
    log("gas blocks OK: X_H=%.3f X_He=%.3f median T=%.3g K" % (XH, XHe, T))


def los_axes(axis):
    """(xaxis, yaxis) for a LOS along `axis`, keeping a right-handed frame."""
    return {0: (1, 2), 1: (2, 0), 2: (0, 1)}[axis]


# ---------------------------------------------------------------------------
# Spectrum generation
# ---------------------------------------------------------------------------

def fold_periodic(arr, nbins, weights=None):
    """
    Fold an array of shape (Nlos, k*nbins) spanning k box lengths in velocity
    into (Nlos, nbins) spanning [0, vbox].  With `weights`, return the weighted
    mean of `arr` (for column-density-weighted physical quantities).
    """
    nlos = arr.shape[0]
    k = arr.shape[1] // nbins
    if weights is None:
        return arr.reshape(nlos, k, nbins).sum(axis=1)
    w = weights.reshape(nlos, k, nbins)
    wsum = w.sum(axis=1)
    qsum = (arr.reshape(nlos, k, nbins) * w).sum(axis=1)
    out = np.zeros_like(wsum)
    np.divide(qsum, wsum, out=out, where=wsum > 0)
    return out


def gaussian_lsf(flux, fwhm_kms, dv):
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(flux, fwhm_kms / 2.3548200450309493 / dv, mode="wrap")


def compute_spectra(s, line_name, los, vbox, nbins, axis, hi_from_snapshot, nwrap):
    """
    Return a dict of folded per-pixel arrays, each of shape (Nlos, nbins):
    tau, col_density [cm^-2], phys_density [g cm^-3], temperature [K],
    metal_frac, vpec [km/s].
    """
    import pygad as pg
    A = pg.analysis.absorption_spectra

    line = A.lines[line_name]
    ion = line["ion"]
    if hi_from_snapshot and ion == "HI":
        # HI as tracked by the simulation itself (e.g. Simba/Grackle with
        # self-shielding) instead of the pygad ionisation tables
        ion = s.gas["nh"] * s.gas["H"]
    xaxis, yaxis = los_axes(axis)
    taus, col, rho, temp, metal, vpec, _, _ = A.mock_absorption_spectra_multilos(
        s, los, ion,
        l=line["l"], f=line["f"], atomwt=line["atomwt"],
        A_ki=line.get("A_ki", 0.0), element=line["element"],
        vel_extent=pg.UnitArr([-nwrap * vbox, (nwrap + 1) * vbox], "km/s"),
        Nbins=(2 * nwrap + 1) * nbins, xaxis=xaxis, yaxis=yaxis, return_los_phys=True,
    )
    col = np.asarray(col, dtype=np.float64)
    out = {
        "tau": fold_periodic(np.asarray(taus), nbins),
        "col_density": fold_periodic(col, nbins),
    }
    for key, arr in (("phys_density", rho), ("temperature", temp),
                     ("metal_frac", metal), ("vpec", vpec)):
        out[key] = fold_periodic(np.asarray(arr, dtype=np.float64), nbins, weights=col)
    return out


def write_spectrum_file(path, line_name, lambda_rest, z, los_pos, vel, wave, flux,
                        flux_noiseless, noise, data, i, attrs):
    """Same dataset names as pygad.analysis.write_spectrum, plus extras."""
    tmp = path + ".tmp"
    with h5py.File(tmp, "w") as hf:
        lam0 = hf.create_dataset("lambda_rest", data=lambda_rest)
        lam0.attrs["ion_name"] = line_name
        hf.create_dataset("LOS_pos", data=np.asarray(los_pos, dtype=np.float64))
        hf.create_dataset("redshift", data=z)
        hf.create_dataset("velocity", data=vel)
        hf.create_dataset("wavelength", data=wave)
        hf.create_dataset("flux", data=flux)
        hf.create_dataset("flux_noiseless", data=flux_noiseless)
        hf.create_dataset("tau", data=data["tau"][i])
        hf.create_dataset("noise", data=noise)
        hf.create_dataset("col_density", data=data["col_density"][i])
        hf.create_dataset("phys_density", data=data["phys_density"][i])
        hf.create_dataset("temperature", data=data["temperature"][i])
        hf.create_dataset("metallicity",
                          data=np.log10(np.maximum(data["metal_frac"][i], 1e-10)))
        hf.create_dataset("vpec", data=data["vpec"][i])
        units = {"velocity": "km/s", "wavelength": "Angstrom",
                 "col_density": "cm**-2 (per pixel)", "phys_density": "g cm**-3",
                 "temperature": "K", "metallicity": "log10 mass fraction",
                 "vpec": "km/s"}
        for k, u in units.items():
            hf[k].attrs["units"] = u
        for k, v in attrs.items():
            hf.attrs[k] = v
    os.replace(tmp, path)


def generate(args, los, snap_info):
    """Load the snapshot once and write every missing (line, LOS) spectrum."""
    import pygad as pg

    todo = {}
    for line_name in args.lines:
        missing = [i for i in range(len(los))
                   if args.overwrite or not os.path.exists(spec_path(args, line_name, i))]
        if missing:
            todo[line_name] = missing
        else:
            log("%s: all %d spectra exist, skipping (use --overwrite to redo)"
                % (line_name, len(los)))
    if not todo:
        return

    log("loading %s" % args.snapshot)
    s = pg.Snapshot(args.snapshot)
    if snap_info["gizmo_fix"]:
        log("--gizmo fix: correcting u, element and metallicity blocks in memory")
        fix_gizmo_blocks(s)
    check_gas_blocks(s, list(todo))

    z = float(s.redshift)
    H = s.cosmology.H(s.redshift).in_units_of("(km/s)/Mpc", subs=s)
    vbox = float(H * s.boxsize.in_units_of("Mpc", subs=s))
    nbins = round(vbox / args.vpix)
    dv = vbox / nbins
    vel = (np.arange(nbins) + 0.5) * dv
    snap_info.update(redshift=z, vbox_kms=vbox, nbins=nbins, dv_kms=dv,
                     boxsize=float(s.boxsize), pos_units=str(s["pos"].units))
    log("z=%.4f  box=%g %s  vbox=%.1f km/s  %d pixels of %.3f km/s"
        % (z, float(s.boxsize), s["pos"].units, vbox, nbins, dv))

    # extend the velocity window by enough box lengths on each side that no gas
    # (incl. fast winds/jets) + 1000 km/s of line wing falls outside before folding
    vmax = float(np.abs(s.gas["vel"][:, args.axis].in_units_of("km/s", subs=s)).max())
    nwrap = int(np.ceil((vmax + 1000.0) / vbox))
    log("max |v_pec| = %.0f km/s: computing over %d box lengths before folding"
        % (vmax, 2 * nwrap + 1))

    attrs = {"snapshot": os.path.abspath(args.snapshot), "axis": "xyz"[args.axis],
             "vbox": vbox, "dv": dv, "snr": args.snr, "lsf": args.lsf or "none",
             "seed": args.seed, "gizmo_fix": snap_info["gizmo_fix"],
             "pos_units": str(s["pos"].units),
             "hi_from_snapshot": bool(args.hi_from_snapshot)}

    for iline, (line_name, missing) in enumerate(todo.items()):
        lam0 = float(pg.analysis.absorption_spectra.lines[line_name]["l"].split()[0])
        wave = lam0 * (1.0 + z) * (1.0 + vel / C_KMS)
        lsf = check_lsf(args.lsf, wave)
        os.makedirs(os.path.join(args.outdir, line_name), exist_ok=True)
        log("%s: computing %d spectra (lambda_obs ~ %.1f A)"
            % (line_name, len(missing), wave.mean()))
        for b0 in range(0, len(missing), args.batch):
            idx = missing[b0:b0 + args.batch]
            data = compute_spectra(s, line_name, los[idx], vbox, nbins,
                                   args.axis, args.hi_from_snapshot, nwrap)
            for k, i in enumerate(idx):
                flux0 = np.exp(-data["tau"][k])
                if lsf is None:
                    flux_lsf = flux0
                elif lsf[0] == "gauss":
                    flux_lsf = gaussian_lsf(flux0, lsf[1], dv)
                else:
                    flux_lsf, _ = pg.analysis.absorption_spectra.apply_LSF(
                        wave, flux0, np.ones(nbins), grating=lsf[1])
                # independent, reproducible noise for every (line, LOS)
                rng = np.random.default_rng([args.seed, iline, i])
                sigma = np.full(nbins, 1.0 / args.snr)
                flux = flux_lsf + rng.normal(0.0, sigma)
                los_pos = np.zeros(3)
                los_pos[list(los_axes(args.axis))] = los[i]
                los_pos[args.axis] = -1.0
                write_spectrum_file(spec_path(args, line_name, i), line_name, lam0, z,
                                    los_pos, vel, wave, flux, flux_lsf, sigma,
                                    data, k, attrs)
            log("%s: wrote %d/%d" % (line_name, min(b0 + args.batch, len(missing)),
                                     len(missing)))


def check_lsf(lsf, wave):
    """Parse --lsf; refuse a COS grating whose LSF tables don't cover `wave`."""
    if lsf is None or lsf.lower() == "none":
        return None
    if lsf.startswith("gauss:"):
        return ("gauss", float(lsf.split(":", 1)[1]))
    import pygad as pg
    data = pg.analysis.absorption_spectra.LSF_data
    if lsf not in data:
        raise SystemExit("unknown LSF '%s'; choose none, gauss:<FWHM km/s> or one of %s"
                         % (lsf, sorted(data)))
    ch = np.asarray(data[lsf]["channels"], dtype=float)
    if not (ch.min() - 50 <= wave.mean() <= ch.max() + 50):
        raise SystemExit("LSF %s covers %g-%g A but this line is observed at ~%.0f A; "
                         "use --lsf gauss:<FWHM> or none"
                         % (lsf, ch.min(), ch.max(), wave.mean()))
    return ("grating", lsf)


# ---------------------------------------------------------------------------
# Voigt profile fitting
# ---------------------------------------------------------------------------

def default_bounds(line_name, logN_bounds, b_bounds):
    import pygad as pg
    A = pg.analysis.absorption_spectra
    if logN_bounds is None:
        logN_bounds = [12.0, 19.0] if A.lines[line_name]["ion"] == "HI" else [10.0, 17.0]
    if b_bounds is None:
        b_bounds = [0.0, 150.0]
    if b_bounds[0] == 0:
        # as in vpfit.fit_profiles: 1/4 of the thermal width at 1e4 K
        b_therm = float(A.thermal_b_param(line_name, pg.UnitScalar(1e4, "K"), "km/s"))
        b_bounds = [0.25 * b_therm, b_bounds[1]]
    return list(map(float, logN_bounds)), list(map(float, b_bounds))


def fit_spectrum_file(path, opts):
    """
    Fit Voigt profiles to one periodic spectrum file and store the result in
    its 'line_list' group (schema of vpfit.write_line_list, plus 'v').
    Runs in a worker process.
    """
    import pygad as pg
    from pygad.analysis import vpfit
    from pygad.analysis.tau_lookup import model_tau_fast

    pg.environment.verbose = opts["pygad_verbose"]
    with h5py.File(path, "r") as hf:
        line_name = hf["lambda_rest"].attrs["ion_name"]
        if isinstance(line_name, bytes):
            line_name = line_name.decode()
        lam0 = float(hf["lambda_rest"][()])
        z = float(hf["redshift"][()])
        vel = np.array(hf["velocity"])
        flux = np.array(hf["flux"])
        noise = np.array(hf["noise"])
        tau = np.array(hf["tau"]) if "tau" in hf else None
        if "line_list" in hf and not opts["overwrite"]:
            return path, None
    npix = len(vel)
    dv = vel[1] - vel[0]
    vbox = npix * dv

    # start at the least-absorbed pixel so no feature straddles the edge
    istart = int(np.argmin(tau)) if tau is not None else int(np.argmax(flux))
    f_r = np.roll(flux, -istart)
    n_r = np.roll(noise, -istart)
    v_r = vel[istart] + dv * np.arange(npix)       # monotonic, may exceed vbox
    l_r = lam0 * (1.0 + z) * (1.0 + v_r / C_KMS)

    nbuf = opts["nbuffer"]
    l_b = np.concatenate((l_r[0] + (l_r[1] - l_r[0]) * np.arange(-nbuf, 0), l_r,
                          l_r[-1] + (l_r[-1] - l_r[-2]) * np.arange(1, nbuf + 1)))
    f_b = np.concatenate((np.ones(nbuf), f_r, np.ones(nbuf)))
    n_b = np.concatenate((np.full(nbuf, n_r.mean()), n_r, np.full(nbuf, n_r.mean())))

    logN_bounds, b_bounds = default_bounds(line_name, opts["logN_bounds"], opts["b_bounds"])
    regions_l, regions_i = vpfit.find_regions(l_b, f_b, n_b, N_sigma=opts["N_sigma"],
                                              verbose=opts["verbose"])
    if len(regions_l) > 0:
        ll = vpfit.fit_profiles_sat(
            line_name, l_b, f_b, n_b, regions_l, regions_i,
            chisq_lim=opts["chisq_lim"], chisq_factor=opts["chisq_factor"],
            N_sigma_constr=opts["N_sigma_constr"], max_lines=opts["max_lines"],
            logN_bounds=logN_bounds, b_bounds=b_bounds, mode="Voigt",
            verbose=opts["verbose"])
    else:
        ll = {}
    keys = ["region", "N", "dN", "b", "db", "l", "dl", "EW", "Chisq"]
    ll = {k: np.asarray(ll.get(k, []), dtype=float) for k in keys}

    # model on the rolled grid, then rolled back onto the stored grid; uses the
    # same tau lookup as the fitter so model and fit are consistent
    tau_model = np.zeros(npix)
    if len(ll["N"]):
        p = np.column_stack((ll["N"], ll["b"], ll["l"])).ravel()
        tau_model = model_tau_fast(line_name, p, l_r)
    model_flux = np.roll(np.exp(-np.clip(tau_model, 0, 50)), istart)

    def to_box(l_obs):
        v = (C_KMS * (np.asarray(l_obs) / (lam0 * (1.0 + z)) - 1.0)) % vbox
        return v, lam0 * (1.0 + z) * (1.0 + v / C_KMS)

    v_line, l_line = to_box(ll["l"])
    regions_l = np.asarray(regions_l, dtype=float).reshape(-1, 2)
    regions_i = np.asarray(regions_i, dtype=int).reshape(-1, 2)
    _, reg_l = to_box(regions_l) if len(regions_l) else (None, regions_l)
    reg_i = (regions_i - nbuf + istart) % npix
    chisq = (np.sum(((flux - model_flux) / noise) ** 2) / npix)

    with h5py.File(path, "a") as hf:
        for name in ("line_list", "lines"):
            if name in hf:
                del hf[name]
        grp = hf.create_group("line_list")
        grp.create_dataset("region", data=ll["region"].astype(int))
        grp.create_dataset("logN", data=ll["N"])
        grp.create_dataset("dlogN", data=ll["dN"])
        grp.create_dataset("b", data=ll["b"])
        grp.create_dataset("db", data=ll["db"])
        grp.create_dataset("l", data=l_line)
        grp.create_dataset("dl", data=ll["dl"])
        grp.create_dataset("v", data=v_line)
        grp.create_dataset("EW", data=ll["EW"])
        grp.create_dataset("chisq", data=ll["Chisq"])
        grp.create_dataset("model_flux", data=model_flux)
        grp.create_dataset("region_lam_start", data=reg_l[:, 0] if len(reg_l) else [])
        grp.create_dataset("region_lam_end", data=reg_l[:, 1] if len(reg_l) else [])
        grp.create_dataset("region_pix_start", data=reg_i[:, 0] if len(reg_i) else [])
        grp.create_dataset("region_pix_end", data=reg_i[:, 1] if len(reg_i) else [])
        grp.attrs["total_reduced_chisq"] = chisq
        grp.attrs["logN_bounds"] = logN_bounds
        grp.attrs["b_bounds"] = b_bounds
        grp.attrs["note"] = ("regions may wrap: region_*_start > region_*_end "
                             "means the region crosses the box edge")
    return path, len(ll["N"])


def fit_all(args, nlos):
    opts = {"overwrite": args.overwrite_fits, "chisq_lim": args.chisq_lim,
            "chisq_factor": args.chisq_factor, "N_sigma_constr": args.N_sigma_constr,
            "N_sigma": args.N_sigma, "max_lines": args.max_lines,
            "logN_bounds": args.logN_bounds, "b_bounds": args.b_bounds,
            "nbuffer": 50, "verbose": args.verbose >= 2,
            "pygad_verbose": 1 if args.verbose >= 2 else 0}
    paths = [spec_path(args, ln, i) for ln in args.lines for i in range(nlos)]
    paths = [p for p in paths if os.path.exists(p)]
    log("fitting Voigt profiles to %d spectra with %d process(es)" % (len(paths), args.nproc))
    failed = []
    done = 0

    def report(path, result):
        nonlocal done
        done += 1
        if args.verbose or done % max(1, len(paths) // 20) == 0 or done == len(paths):
            what = "kept existing fit" if result is None else "%d components" % result
            log("  [%d/%d] %s: %s" % (done, len(paths), os.path.relpath(path, args.outdir), what))

    if args.nproc <= 1:
        for p in paths:
            try:
                report(*fit_spectrum_file(p, opts))
            except Exception as err:  # keep going; report at the end
                failed.append((p, repr(err)))
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.nproc, mp_context=ctx) as ex:
            futs = {ex.submit(fit_spectrum_file, p, opts): p for p in paths}
            for fut in as_completed(futs):
                try:
                    report(*fut.result())
                except Exception as err:  # keep going; report at the end
                    failed.append((futs[fut], repr(err)))
    for p, err in failed:
        log("FIT FAILED %s: %s" % (p, err))
    for ln in args.lines:
        write_catalogue(args, ln, nlos)
    return failed


def write_catalogue(args, line_name, nlos):
    """Collect all fitted components of one line into OUTDIR/lines_<line>.h5."""
    cols = {k: [] for k in ("los", "logN", "dlogN", "b", "db", "v", "l", "dl", "EW", "chisq")}
    nfit = 0
    for i in range(nlos):
        p = spec_path(args, line_name, i)
        if not os.path.exists(p):
            continue
        with h5py.File(p, "r") as hf:
            if "line_list" not in hf:
                continue
            nfit += 1
            g = hf["line_list"]
            n = len(g["logN"])
            cols["los"].append(np.full(n, i))
            for k in cols:
                if k != "los":
                    cols[k].append(np.array(g[k]))
    out = os.path.join(args.outdir, "lines_%s.h5" % line_name)
    with h5py.File(out, "w") as hf:
        for k, v in cols.items():
            hf.create_dataset(k, data=np.concatenate(v) if v else np.array([]))
        hf.attrs["line"] = line_name
        hf.attrs["n_spectra_fitted"] = nfit
    ncomp = sum(len(v) for v in cols["logN"])
    log("%s: %d components from %d spectra -> %s" % (line_name, ncomp, nfit, out))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_overview(args, nlos):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(args.plot, nlos)
    nl = len(args.lines)
    fig, axes = plt.subplots(n, nl, figsize=(5 * nl, 1.6 * n + 0.6), sharex=True,
                             sharey=True, squeeze=False)
    for i in range(n):
        for j, ln in enumerate(args.lines):
            ax = axes[i, j]
            p = spec_path(args, ln, i)
            if not os.path.exists(p):
                ax.set_axis_off()
                continue
            with h5py.File(p, "r") as hf:
                v = np.array(hf["velocity"])
                ax.plot(v, np.array(hf["flux"]), color="0.55", lw=0.6, drawstyle="steps-mid")
                if "line_list" in hf:
                    g = hf["line_list"]
                    ax.plot(v, np.array(g["model_flux"]), color="C3", lw=1.0)
                    for vc in g["v"]:
                        ax.axvline(vc, ymin=0.92, ymax=1.0, color="C0", lw=0.8)
            ax.set_ylim(-0.1, 1.25)
            ax.set_xlim(v[0], v[-1])
            if i == 0:
                ax.set_title(ln)
            if j == 0:
                ax.set_ylabel("LOS %d" % i)
    for ax in axes[-1]:
        ax.set_xlabel("v [km/s]")
    fig.subplots_adjust(hspace=0, wspace=0.03)
    out = os.path.join(args.outdir, "spectra.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    log("wrote %s" % out)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def spec_path(args, line_name, i):
    return os.path.join(args.outdir, line_name, "los_%05d.h5" % i)


def get_los(args):
    """Random (or user-supplied) transverse LOS positions, in snapshot units."""
    los_file = os.path.join(args.outdir, "los.txt")
    if args.los_file:
        los = np.atleast_2d(np.loadtxt(args.los_file, usecols=(0, 1)))
        src = args.los_file
    elif os.path.exists(los_file) and not args.new_los:
        # reuse, so an interrupted/extended run stays consistent
        los = np.atleast_2d(np.loadtxt(los_file, usecols=(1, 2)))
        if len(los) < args.nlos:
            raise SystemExit("%s has %d LOS but -n %d was requested; use --new-los "
                             "(and --overwrite) to draw a new set" % (los_file, len(los), args.nlos))
        los = los[:args.nlos]
        src = los_file
    else:
        with h5py.File(args.snapshot, "r") as f:
            boxsize = float(f["Header"].attrs["BoxSize"])
        rng = np.random.default_rng(args.seed)
        los = rng.uniform(0.0, boxsize, size=(args.nlos, 2))
        src = "random (seed %d)" % args.seed
    if not args.fit_only:
        with open(los_file, "w") as f:
            f.write("# LOS along axis %s; columns: index, %s, %s (snapshot pos units)\n"
                    % ("xyz"[args.axis], *("xyz"[a] for a in los_axes(args.axis))))
            for i, (a, b) in enumerate(los):
                f.write("%d %.6f %.6f\n" % (i, a, b))
    log("%d LOS from %s" % (len(los), src))
    return los


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("snapshot", help="snapshot file (HDF5)")
    p.add_argument("-n", "--nlos", type=int, default=10, help="number of random LOS (default 10)")
    p.add_argument("--lines", nargs="+", default=["H1215"],
                   help="line names from pygad.analysis.absorption_spectra.lines, "
                        "e.g. H1215 CIV1548 OVI1031 (default H1215)")
    p.add_argument("-o", "--outdir", help="output directory (default ./spectra_<snapshot>)")
    p.add_argument("--axis", choices="xyz", default="z", help="LOS direction (default z)")
    p.add_argument("--seed", type=int, default=12345, help="random seed for LOS and noise")
    p.add_argument("--los-file", help="text file with 'x y' LOS positions (snapshot units) "
                                      "instead of random ones")
    p.add_argument("--new-los", action="store_true",
                   help="draw new random LOS even if OUTDIR/los.txt exists")
    p.add_argument("--vpix", type=float, default=2.5, help="pixel size in km/s (default 2.5)")
    p.add_argument("--snr", type=float, default=50.0, help="S/N per pixel (default 50)")
    p.add_argument("--lsf", default="none",
                   help="none (default), gauss:<FWHM km/s>, or a COS grating from "
                        "absorption_spectra.LSF_data (e.g. COS_G130M)")
    p.add_argument("--hi-from-snapshot", action="store_true",
                   help="for HI use the snapshot's NeutralHydrogenAbundance instead of "
                        "the pygad ionisation tables")
    p.add_argument("--gizmo", choices=("check", "fix"), default="check",
                   help="check (default): stop if the gas blocks look inconsistent with "
                        "the snapshot; fix: correct u/elements/metallicity of a "
                        "GIZMO/Simba snapshot in memory (for pygad's stock gadget.cfg)")
    p.add_argument("--batch", type=int, default=500, help="LOS per C call (memory control)")
    p.add_argument("--overwrite", action="store_true", help="regenerate existing spectra")

    f = p.add_argument_group("Voigt profile fitting")
    f.add_argument("--fit", action="store_true", help="fit Voigt profiles after generating")
    f.add_argument("--fit-only", action="store_true",
                   help="only fit existing spectra in OUTDIR (snapshot is not loaded)")
    f.add_argument("--overwrite-fits", action="store_true", help="refit already-fitted spectra")
    f.add_argument("--nproc", type=int, default=1, help="parallel fitting processes")
    f.add_argument("--logN-bounds", type=float, nargs=2, metavar=("MIN", "MAX"),
                   help="log10 N bounds (default 12 19 for HI, 10 17 for metals)")
    f.add_argument("--b-bounds", type=float, nargs=2, metavar=("MIN", "MAX"),
                   help="b bounds in km/s; MIN=0 means 1/4 of b_thermal(1e4 K) "
                        "(default 0 150)")
    f.add_argument("--chisq-lim", type=float, default=2.0,
                   help="stop adding components below this reduced chi^2 (default 2)")
    f.add_argument("--chisq-factor", type=float, default=0.95)
    f.add_argument("--N-sigma", type=float, default=10.0,
                   help="detection significance for absorption regions (default 10)")
    f.add_argument("--N-sigma-constr", type=float, default=3.0)
    f.add_argument("--max-lines", type=int, default=12, help="max components per region")

    p.add_argument("--plot", type=int, default=0, metavar="N",
                   help="plot the first N LOS for all lines to OUTDIR/spectra.png")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args(argv)
    args.axis = AXIS_NAMES[args.axis]
    if args.outdir is None:
        base = os.path.splitext(os.path.basename(args.snapshot))[0]
        args.outdir = os.path.abspath("spectra_" + base)
    if args.lsf.lower() == "none":
        args.lsf = None
    return args


def main(argv=None):
    args = parse_args(argv)
    import pygad as pg
    pg.environment.verbose = (pg.environment.VERBOSE_NORMAL if args.verbose >= 2
                              else pg.environment.VERBOSE_QUIET)
    unknown = [ln for ln in args.lines if ln not in pg.analysis.absorption_spectra.lines]
    if unknown:
        raise SystemExit("unknown line(s) %s; see pygad.analysis.absorption_spectra.lines"
                         % unknown)
    os.makedirs(args.outdir, exist_ok=True)

    los = get_los(args)
    if not args.fit_only:
        info = {"gizmo_fix": args.gizmo == "fix"}
        if info["gizmo_fix"] and not is_gizmo_snapshot(args.snapshot):
            raise SystemExit("--gizmo fix: %s does not look like a GIZMO snapshot"
                             % args.snapshot)
        generate(args, los, info)
        run = {k: v for k, v in vars(args).items()}
        run.update(info)
        run["nlos"] = len(los)
        with open(os.path.join(args.outdir, "run_info.json"), "w") as f:
            json.dump(run, f, indent=1, default=str)

    failed = []
    if args.fit or args.fit_only:
        failed = fit_all(args, len(los))
    if args.plot:
        plot_overview(args, len(los))
    log("done")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

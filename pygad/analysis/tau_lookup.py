"""
pygad/analysis/tau_lookup.py
────────────────────────────
Fast optical-depth computation via a pre-computed lookup table.

Physics
-------
For any absorption line, optical depth scales linearly with column density N:

    τ(v ; N, b) = N · τ_unit(v ; b)

where τ_unit is the profile at N = 1 cm⁻² and its shape in velocity space
depends only on the ion's atomic constants and the Doppler parameter b —
NOT on the observed (redshifted) central wavelength λ₀.

The TauLookupTable class pre-computes τ_unit on a fine uniform velocity
grid for a log-spaced set of b values.  Evaluating τ for a new (logN, b, λ₀)
triplet then costs only:

  ① 1-D linear interpolation in log(b) → τ_unit on the stored v-grid
  ② np.interp to convert to the actual pixel velocities
  ③ a scalar multiply by N = 10^logN

Typical speedup over the exact Faddeeva-function evaluation: 20–100×.

Public API
----------
  TauLookupTable  – build, query, save/load, and validate a table.
  get_tau_lookup  – cached factory; one shared table per (ion, mode).
  model_tau_fast  – drop-in replacement for vpfit.model_tau().
  benchmark       – wall-clock comparison of both implementations.
"""

from __future__ import annotations

from .absorption_spectra import lines, line_profile
from pygad.physics import c as _c_light
from .. import environment

import numpy as np
import os
import time

__all__ = ["TauLookupTable", "get_tau_lookup", "model_tau_fast", "benchmark"]

# Speed of light in km/s – computed once at import time
_C_KMS: float = float(_c_light.in_units_of("km/s"))

# In-memory cache: (ion_name, mode) → TauLookupTable
_tau_lookup_cache: dict[tuple[str, str], "TauLookupTable"] = {}


# ══════════════════════════════════════════════════════════════════════════ #
class TauLookupTable:
    """
    Pre-computed 2-D lookup table of line-profile optical depths.

    Stores τ_unit(v ; b) on a uniform velocity grid for a log-spaced set of
    b values.  Querying the table for an arbitrary (logN, b, λ₀) absorption
    line requires only simple array interpolation.

    Parameters
    ----------
    ion_name : str
        Line identifier, e.g. 'HI1215', as used in absorption_spectra.lines.
    mode : {'Voigt', 'Gaussian', 'Lorentzian'}
        Profile shape.
    b_min, b_max : float
        Range of Doppler parameters covered (km/s).
        Queries outside this range are silently clamped to the nearest edge.
    n_b : int
        Number of b grid entries (log-spaced).  60 gives < 0.1 % interpolation
        error across the full profile for typical Voigt lines.
    v_max : float
        Half-width of the velocity grid (km/s).
        Rule of thumb: 5 000 km/s for most metal lines;
                       ≥ 30 000 km/s for damped Lyman-alpha systems.
    dv : float
        Velocity pixel size (km/s).  Must satisfy dv ≲ b_min / 10 for
        accurate representation of narrow lines.
        Default 0.5 km/s works well for b_min ≥ 5 km/s.

    Memory footprint
    ----------------
    n_b × (2·v_max/dv + 1) × 8 bytes (float64).
    With defaults → 60 × 80 001 × 8 ≈ 38 MB per ion+mode.
    """

    # Default grid parameters
    _B_MIN: float = 1.0
    _B_MAX: float = 300.0
    _N_B:   int   = 100
    _V_MAX: float = 20_000.0   # km/s — covers damping wings of strong HI lines
    _DV:    float = 0.5        # km/s — accurate for b ≥ 5 km/s

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        ion_name: str,
        mode:     str   = "Voigt",
        b_min:    float = _B_MIN,
        b_max:    float = _B_MAX,
        n_b:      int   = _N_B,
        v_max:    float = _V_MAX,
        dv:       float = _DV,
    ) -> None:
        if ion_name not in lines:
            raise KeyError(
                f"Ion '{ion_name}' not found in absorption_spectra.lines."
            )
        self.ion_name = ion_name
        self.mode     = mode
        self.b_min    = float(b_min)
        self.b_max    = float(b_max)
        self.n_b      = int(n_b)
        self.v_max    = float(v_max)
        self.dv       = float(dv)
        self.c_kms    = _C_KMS          # expose for callers that need it
        self._build()

    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        """
        Fill tau_table[n_b, n_v] by calling line_profile() once per b value.

        The profile shape in velocity space is independent of the observed
        central wavelength, so a single template at l_rest is valid for all
        redshifted lines of the same ion.
        """
        line_data = lines[self.ion_name]
        # Robust extraction of rest wavelength (Å) from float / string / UnitScalar
        l_rest = float(str(line_data["l"]).split()[0])

        # ── uniform velocity grid, symmetric about v = 0 ──────────────
        n_v         = int(round(2.0 * self.v_max / self.dv)) + 1
        self.v_grid = np.linspace(-self.v_max, self.v_max, n_v)

        # ── log-spaced b grid ─────────────────────────────────────────
        self.log_b_grid = np.linspace(
            np.log10(self.b_min), np.log10(self.b_max), self.n_b
        )
        self.b_grid = 10.0 ** self.log_b_grid

        # Template wavelengths: one wavelength per velocity grid point,
        # centred on l_rest.  The profile shape at any other l0 is identical
        # in velocity space, so this template covers all redshifts.
        l_template = l_rest * (1.0 + self.v_grid / _C_KMS)

        # ── fill the lookup table ─────────────────────────────────────
        t0 = time.perf_counter()
        if environment.verbose >= environment.VERBOSE_TACITURN:
            print(
                f"Building tau lookup table for {self.ion_name} ({self.mode}):\n"
                f"  {self.n_b} b values in [{self.b_min}, {self.b_max}] km/s\n"
                f"  velocity grid: {n_v:,} pts over ±{self.v_max:g} km/s "
                f"(Δv = {self.dv} km/s) …",
                flush=True,
            )

        self.tau_table = np.empty((self.n_b, n_v), dtype=np.float64)
        for ib, bpar in enumerate(self.b_grid):
            _, tau = line_profile(
                line_data,
                1.0,                        # N = 1 cm⁻² (unit column density)
                b=float(bpar),
                l0=l_rest,
                l=l_template,
                mode=self.mode,
            )
            self.tau_table[ib] = np.asarray(tau, dtype=np.float64)

        elapsed = time.perf_counter() - t0
        if environment.verbose >= environment.VERBOSE_TACITURN:
            print(
                f"  Done in {elapsed:.2f} s  "
                f"(table: {self.tau_table.nbytes / 1e6:.1f} MB)",
                flush=True,
            )

    # ── Core interpolation ───────────────────────────────────────────── #
    def interp_b(self, b: float) -> np.ndarray:
        """
        Linear interpolation in log(b) → τ_unit profile on self.v_grid.

        b is silently clamped to [b_min, b_max].

        Returns
        -------
        tau_unit : ndarray, shape (n_v,)
            Optical depth per unit column density at each velocity grid point.
        """
        log_b = np.log10(np.clip(b, self.b_min, self.b_max))
        # Index of first grid entry strictly greater than log_b
        ib    = int(np.searchsorted(self.log_b_grid, log_b, side="right"))
        ib    = np.clip(ib, 1, self.n_b - 1)
        t     = (
            (log_b - self.log_b_grid[ib - 1])
            / (self.log_b_grid[ib] - self.log_b_grid[ib - 1])
        )
        return (1.0 - t) * self.tau_table[ib - 1] + t * self.tau_table[ib]

    def interp_v(
        self,
        tau_unit: np.ndarray,
        l:        np.ndarray,
        l0:       float,
    ) -> np.ndarray:
        """
        Interpolate a τ_unit profile (on self.v_grid) to a wavelength grid.

        Separating this from interp_b() lets callers pre-compute interp_b()
        once and reuse it for multiple column densities (see _grow_line
        optimisation in vpfit.py).

        Parameters
        ----------
        tau_unit : ndarray   τ_unit on self.v_grid (output of interp_b).
        l        : ndarray   Output wavelength grid (Å).
        l0       : float     Central (observed) wavelength of the line (Å).

        Returns
        -------
        tau_at_l : ndarray, shape (len(l),)
        """
        v_out = (np.asarray(l, dtype=np.float64) / l0 - 1.0) * _C_KMS
        return np.interp(v_out, self.v_grid, tau_unit, left=0.0, right=0.0)

    # ------------------------------------------------------------------ #
    def compute_tau(
        self,
        logN: float,
        b:    float,
        l0:   float,
        l:    np.ndarray,
    ) -> np.ndarray:
        """
        Full optical-depth profile for one absorption line.

        Equivalent to calling line_profile() but ~20-100× faster.

        Parameters
        ----------
        logN : float      log₁₀ column density (cm⁻²).
        b    : float      Doppler parameter (km/s).
        l0   : float      Central (observed) wavelength, possibly redshifted (Å).
        l    : array-like Output wavelength grid (Å).

        Returns
        -------
        tau : ndarray, shape (len(l),)
        """
        # ① b-interpolation: τ_unit on the stored velocity grid
        tau_unit = self.interp_b(b)

        # ② velocity interpolation: τ_unit at the actual pixel wavelengths
        #    Pixels beyond ±v_max receive τ = 0 (no absorption)
        tau_at_l = self.interp_v(tau_unit, np.asarray(l, dtype=float), l0)

        # ③ scale by column density N = 10^logN
        return (10.0 ** logN) * tau_at_l

    # ── Persistence ─────────────────────────────────────────────────── #
    def save(self, filepath: str) -> None:
        """Save the lookup table to a compressed NumPy archive (.npz)."""
        if not filepath.endswith(".npz"):
            filepath += ".npz"
        np.savez_compressed(
            filepath,
            tau_table    = self.tau_table,
            v_grid       = self.v_grid,
            b_grid       = self.b_grid,
            meta_strings = np.array([self.ion_name, self.mode]),
            meta_floats  = np.array(
                [self.b_min, self.b_max, self.v_max, self.dv], dtype=float
            ),
            meta_ints    = np.array([self.n_b], dtype=int),
        )
        print(f"Saved lookup table → {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "TauLookupTable":
        """
        Restore a lookup table from a .npz file created by save().

        Parameters
        ----------
        filepath : str   Path to file (the .npz extension is optional).
        """
        if not filepath.endswith(".npz"):
            filepath += ".npz"
        data = np.load(filepath, allow_pickle=True)

        obj            = cls.__new__(cls)
        obj.tau_table  = data["tau_table"]
        obj.v_grid     = data["v_grid"]
        obj.b_grid     = data["b_grid"]
        obj.log_b_grid = np.log10(obj.b_grid)
        ms             = data["meta_strings"]
        obj.ion_name   = str(ms[0])
        obj.mode       = str(ms[1])
        mf             = data["meta_floats"]
        obj.b_min, obj.b_max, obj.v_max, obj.dv = (
            float(mf[0]), float(mf[1]), float(mf[2]), float(mf[3])
        )
        obj.n_b    = int(data["meta_ints"][0])
        obj.c_kms  = _C_KMS
        print(
            f"Loaded lookup table for {obj.ion_name} ({obj.mode}) "
            f"from {filepath}"
        )
        return obj

    # ── Accuracy validation ──────────────────────────────────────────── #
    def validate(
        self,
        n_test: int   = 30,
        rtol:   float = 0.02,
        seed:   int   = 42,
    ) -> float:
        """
        Check accuracy of the lookup table against exact line_profile() calls.

        Tests n_test b values drawn uniformly from [b_min, b_max] plus the
        two edge values.  Reports the worst-case relative error at pixels
        where τ_exact > 1 % of the line-centre value.

        Parameters
        ----------
        n_test : int    Number of random b values to test.
        rtol   : float  Pass/fail threshold for relative error (default 2 %).
        seed   : int    RNG seed for reproducibility.

        Returns
        -------
        max_rel_err : float
        """
        line_data = lines[self.ion_name]
        l_rest    = float(str(line_data["l"]).split()[0])
        rng       = np.random.default_rng(seed)

        # Sub-sample the velocity grid (every 10th pt) for speed
        v_sub = self.v_grid[::10]
        l_sub = l_rest * (1.0 + v_sub / _C_KMS)

        # Always include the edge values b_min and b_max
        b_vals = np.concatenate(
            [rng.uniform(self.b_min, self.b_max, n_test),
             [self.b_min, self.b_max]]
        )

        max_err = 0.0
        for b in b_vals:
            _, tau_exact = line_profile(
                line_data, 1.0,
                b=float(b), l0=l_rest,
                l=l_sub, mode=self.mode,
            )
            tau_exact  = np.asarray(tau_exact, dtype=float)
            tau_approx = np.interp(v_sub, self.v_grid, self.interp_b(b))

            peak = tau_exact.max()
            if peak == 0.0:
                continue
            mask = tau_exact > 0.01 * peak
            if mask.any():
                rel_err = np.max(
                    np.abs(tau_approx[mask] - tau_exact[mask]) / tau_exact[mask]
                )
                max_err = max(max_err, float(rel_err))

        passed = max_err <= rtol
        print(
            f"Validation [{self.ion_name}, {self.mode}]: "
            f"max relative error = {100 * max_err:.3f} %  "
            f"[{'PASS ✓' if passed else 'FAIL ✗ — consider reducing dv or increasing n_b'}]"
        )
        return max_err

    # ── Diagnostics ─────────────────────────────────────────────────── #
    def plot(
        self,
        b_values: list[float] | None = None,
        ax=None,
    ) -> None:
        """
        Plot τ_unit(v) for a selection of b values.

        Parameters
        ----------
        b_values : list[float] | None
            b values to plot.  Defaults to 5 log-spaced values across the table.
        ax : matplotlib Axes | None
            Axes to plot into.  Creates a new figure if None.
        """
        import matplotlib.pyplot as plt

        if b_values is None:
            b_values = np.logspace(
                np.log10(self.b_min), np.log10(self.b_max), 5
            )

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        for b in b_values:
            tau_unit = self.interp_b(b)
            # Only plot the non-negligible core
            mask = tau_unit > 1e-6 * tau_unit.max()
            ax.plot(self.v_grid[mask], tau_unit[mask], label=f"b = {b:.0f} km/s")

        ax.set_xlabel("Velocity offset (km/s)")
        ax.set_ylabel(r"$\tau_{\rm unit}(v)$ [N = 1 cm$^{-2}$]")
        ax.set_title(f"{self.ion_name} ({self.mode}) lookup table")
        ax.legend(fontsize=8)
        ax.set_yscale("log")
        plt.tight_layout()
        plt.show()

    # ── Representation ───────────────────────────────────────────────── #
    def __repr__(self) -> str:
        r, c = self.tau_table.shape
        return (
            f"TauLookupTable(ion='{self.ion_name}', mode='{self.mode}', "
            f"b=[{self.b_min}, {self.b_max}] km/s × {self.n_b} pts, "
            f"v_max={self.v_max} km/s, dv={self.dv} km/s, "
            f"shape={r}×{c}, {self.tau_table.nbytes / 1e6:.1f} MB)"
        )


# ══════════════════════════════════════════════════════════════════════════ #
def get_tau_lookup(
    ion_name:  str,
    mode:      str  = "Voigt",
    cache_dir: str  = None,
    **kwargs,
) -> TauLookupTable:
    """
    Cached factory: return the TauLookupTable for (ion_name, mode).

    The first call for a new (ion_name, mode) pair builds the table (≈ 1 s).
    Subsequent calls within the same Python session return the cached object
    with zero overhead.

    If cache_dir is provided, the table is loaded from a .npz file if one
    already exists there; otherwise it is built and saved automatically,
    eliminating the build cost in future sessions.

    Parameters
    ----------
    ion_name  : str   Line identifier, e.g. 'HI1215'.
    mode      : str   Profile shape.
    cache_dir : str   Directory for persistent .npz files (optional).
    **kwargs          Forwarded to TauLookupTable.__init__() when building,
                      e.g. b_min=5, v_max=30000, dv=0.2.

    Returns
    -------
    TauLookupTable
    """
    key = (ion_name, mode)

    if key in _tau_lookup_cache:
        return _tau_lookup_cache[key]

    npz_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        npz_path = os.path.join(cache_dir, f"tau_lookup_{ion_name}_{mode}.npz")
        if os.path.exists(npz_path):
            table = TauLookupTable.load(npz_path)
            _tau_lookup_cache[key] = table
            return table

    table = TauLookupTable(ion_name, mode, **kwargs)

    if npz_path is not None:
        table.save(npz_path)

    _tau_lookup_cache[key] = table
    return table


# ══════════════════════════════════════════════════════════════════════════ #
def model_tau_fast(
    ion_name: str,
    p:        np.ndarray,
    l:        np.ndarray,
    mode:     str                    = "Voigt",
    lookup:   TauLookupTable | None  = None,
) -> np.ndarray:
    """
    Drop-in replacement for vpfit.model_tau() using lookup-table interpolation.

    Computes the summed optical depth of all lines encoded in p on the
    wavelength grid l.  The table is built once and cached automatically.

    Parameters
    ----------
    ion_name : str           Line identifier, e.g. 'HI1215'.
    p        : array-like    Flat parameter array [logN₁, b₁, λ₁, logN₂, …].
    l        : array-like    Wavelength grid (Å).
    mode     : str           Profile shape.
    lookup   : TauLookupTable | None
        Pre-built table to use; auto-built from cache if None.

    Returns
    -------
    total_tau : ndarray, shape (len(l),)
    """
    p     = np.asarray(p, dtype=float)
    l_arr = np.asarray(l, dtype=float)

    total_tau = np.zeros(len(l_arr), dtype=float)
    if p.size == 0:
        return total_tau

    if lookup is None:
        lookup = get_tau_lookup(ion_name, mode)

    for ip in range(p.size // 3):
        total_tau += lookup.compute_tau(
            logN = p[ip * 3],
            b    = p[ip * 3 + 1],
            l0   = p[ip * 3 + 2],
            l    = l_arr,
        )
    return total_tau


# ══════════════════════════════════════════════════════════════════════════ #
def benchmark(
    ion_name:  str = "HI1215",
    mode:      str = "Voigt",
    n_pixels:  int = 1_000,
    n_lines:   int = 5,
    n_trials:  int = 200,
    cache_dir: str = None,
) -> dict:
    """
    Wall-clock comparison of model_tau vs model_tau_fast.

    Parameters
    ----------
    ion_name  : Line to benchmark (must exist in absorption_spectra.lines).
    n_pixels  : Wavelength pixels per call.
    n_lines   : Absorption lines per call.
    n_trials  : Number of calls to average for timing.
    cache_dir : Passed to get_tau_lookup().

    Returns
    -------
    dict with keys 'exact_ms', 'fast_ms', 'speedup'.
    """
    from .vpfit import model_tau   # lazy import avoids circular dependency

    line_data = lines[ion_name]
    l_rest    = float(str(line_data["l"]).split()[0])
    lookup    = get_tau_lookup(ion_name, mode, cache_dir=cache_dir)

    rng = np.random.default_rng(0)
    l   = np.linspace(l_rest * 0.99, l_rest * 1.01, n_pixels)

    def _rand_p() -> np.ndarray:
        logN = rng.uniform(12, 18, n_lines)
        b    = rng.uniform(lookup.b_min, lookup.b_max, n_lines)
        l0   = rng.uniform(l_rest * 0.995, l_rest * 1.005, n_lines)
        return np.column_stack([logN, b, l0]).ravel()

    # warm-up (not timed)
    model_tau(ion_name, _rand_p(), l, mode)
    model_tau_fast(ion_name, _rand_p(), l, mode, lookup=lookup)

    t0 = time.perf_counter()
    for _ in range(n_trials):
        model_tau(ion_name, _rand_p(), l, mode)
    exact_ms = (time.perf_counter() - t0) * 1e3 / n_trials

    t0 = time.perf_counter()
    for _ in range(n_trials):
        model_tau_fast(ion_name, _rand_p(), l, mode, lookup=lookup)
    fast_ms = (time.perf_counter() - t0) * 1e3 / n_trials

    speedup = exact_ms / fast_ms
    print(
        f"\nBenchmark  [{ion_name}, {n_lines} lines, {n_pixels} px, "
        f"{n_trials} trials]\n"
        f"  model_tau       : {exact_ms:8.3f} ms / call\n"
        f"  model_tau_fast  : {fast_ms:8.3f} ms / call\n"
        f"  Speed-up        : {speedup:8.1f}×"
    )
    return {"exact_ms": exact_ms, "fast_ms": fast_ms, "speedup": speedup}


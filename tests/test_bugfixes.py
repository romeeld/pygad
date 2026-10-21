"""Regression checks for pygad bug fixes.

Run with the project venv:

    .venv/bin/python tests/test_bugfixes.py
"""
import inspect
import os

import pygad
from pygad.snapshot.snapshot import Snapshot
from pygad.units.unit_arr import UnitArr, _div_units
from pygad.units.units import define, define_from_cfg, defined_units


def check_fix1():
    # FIX 1: _host_derive_block gained a `cache` parameter so the non-cached
    # derive path (called with a second positional arg) no longer crashes.
    params = inspect.signature(Snapshot._host_derive_block).parameters
    assert 'cache' in params, "missing 'cache' parameter"
    assert params['cache'].default is True, "cache default should be True"


def check_fix2():
    # FIX 2: define_from_cfg's `undefine_old` flag is now honoured; passing
    # undefine_old=False preserves previously defined units.
    # NOTE: allow_redef=True is required here because `import pygad` already
    # auto-loads units.cfg (152 units incl. 'm'); without it the re-definition
    # of 'm' would raise UnitError independently of this fix. allow_redef lets
    # the call run so we can assert zzzcustom survives (it would be wiped if
    # the bug were still present).
    define('zzzcustom', 'm')
    assert 'zzzcustom' in defined_units()
    cfg = os.path.join(os.path.dirname(pygad.__file__), 'config', 'units.cfg')
    define_from_cfg([cfg], undefine_old=False, allow_redef=True)
    assert 'zzzcustom' in defined_units()   # preserved now


def check_fix3a():
    # FIX 3a: _div_units no longer evaluates `1 / None` when both operands are
    # unitless; it returns None instead.
    assert _div_units(UnitArr([1., 2.]), UnitArr([2., 4.])) is None


def check_fix3b():
    # FIX 3b: dividing two unitful UnitArrs yields a UnitArr with combined units.
    r = UnitArr([1., 2.], 'm') / UnitArr([2., 4.], 's')
    assert type(r).__name__ == 'UnitArr'
    assert str(r.units) == '[m s**-1]'


def check_fix3c():
    # FIX 3c: dividing two unitless UnitArrs keeps the UnitArr subclass with
    # units=None.
    r2 = UnitArr([1., 2.]) / UnitArr([2., 4.])
    assert type(r2).__name__ == 'UnitArr'
    assert r2.units is None


def check_fix4():
    # FIX 4: IonisationTable.interp_parts with self-shielding (or a flux
    # factor) no longer divides the passed-in nH array in place, which
    # corrupted a snapshot's cached 'nH' block on every ion lookup.
    import numpy as np

    from pygad.cloudy import config_ion_table
    tbl = config_ion_table(2.0)
    nH = UnitArr(np.logspace(-6, 0, 7), 'cm**-3')
    T = UnitArr(np.full(7, 1e4), 'K')
    nH0 = np.array(nH).copy()
    f1 = tbl.interp_parts('C IV', nH, T, selfshield=True, z=2.0, fbaryon=0.16)
    assert np.array_equal(np.array(nH), nH0), "nH modified in place"
    f2 = tbl.interp_parts('C IV', nH, T, selfshield=True, z=2.0, fbaryon=0.16)
    assert np.array_equal(f1, f2), "repeated lookups differ"


def check_fix5():
    # mock_absorption_spectrum_of decided between one and several LOS with
    # len(los) == 2, so exactly two LOS were treated as a single (x,y) position.
    import numpy as np

    from pygad.analysis import mock_absorption_spectrum_of
    from pygad.units import UnitArr as UA
    s = Snapshot(os.path.join(pygad.environment.module_dir,
                              'snaps', 'snap_M1196_4x_320'), physical=False)
    los = UA([[34700., 35600.], [34800., 35700.]], 'ckpc h_0**-1')
    vel_extent = UA([2000., 3500.], 'km/s')
    taus = mock_absorption_spectrum_of(s.gas, los, 'H1215', vel_extent,
                                       Nbins=100)[0]
    assert np.shape(taus) == (2, 100), "two LOS gave shape %s" % (np.shape(taus),)
    tau0 = mock_absorption_spectrum_of(s.gas, los[0], 'H1215', vel_extent,
                                       Nbins=100)[0]
    assert np.shape(tau0) == (100,)
    assert np.allclose(taus[0], tau0), "multi-LOS spectrum differs from single"


def check_fix6():
    # Spectrum.plot_fit called plt.show()/plt.close() even when drawing into
    # the caller's axes, closing the caller's figure.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    from pygad.analysis.vpfit import Spectrum
    vel = np.linspace(0., 500., 201)
    spec = Spectrum('H1215', 2.0, None, np.ones_like(vel),
                    np.full_like(vel, 0.02), vel)
    spec.line_list = {'N': np.array([13.5]), 'b': np.array([25.]),
                      'l': np.array([spec.wavelengths[100]])}
    fig, ax = plt.subplots()
    assert spec.plot_fit(ax=ax) is ax
    assert plt.fignum_exists(fig.number), "plot_fit closed the caller's figure"
    plt.close(fig)


def check_fix7():
    # fit_profiles_sat (least-squares core): a NaN flux or zero-noise pixel
    # must not abort the fit; b stays within the tau lookup table's range;
    # unconstrained parameters get infinite, not zero, errors.
    import numpy as np

    from pygad.analysis import vpfit
    from pygad.analysis.tau_lookup import get_tau_lookup
    lc = 1548.195 * 3.0
    v = np.arange(-300., 300., 2.5)
    l = lc * (1. + v / 299792.458)
    rng = np.random.default_rng(1)
    flux = np.exp(-vpfit.model_tau('CIV1548', [13.5, 8., lc], l))
    flux += rng.normal(0., 0.02, len(v))
    noise = np.full(len(v), 0.02)
    rl, ri = vpfit.find_regions(l, flux, noise, N_sigma=10.0)
    flux[118], noise[122] = np.nan, 0.     # within the absorption
    ll = vpfit.fit_profiles_sat('CIV1548', l, flux, noise, rl, ri,
                                logN_bounds=[10., 17.], b_bounds=[0.5, 150.])
    i = np.argmax(ll['N'])
    assert abs(ll['N'][i] - 13.5) < 0.1 and abs(ll['b'][i] - 8.) < 1.5, ll
    assert np.min(ll['b']) >= get_tau_lookup('CIV1548', 'Voigt').b_min
    J = np.array([[1., 0.], [2., 0.], [3., 0.]])     # 2nd parameter has no effect
    err = vpfit._lsq_errors(J)
    assert np.isfinite(err[0]) and np.isinf(err[1]), err


def check_sanity():
    # Sanity: a unitful multiplication still yields a UnitArr with units.
    r = UnitArr([2., 4.], 'm') * UnitArr([2., 4.], 's')
    assert type(r).__name__ == 'UnitArr'
    assert str(r.units) == '[m s]'


if __name__ == '__main__':
    check_fix1()
    check_fix2()
    check_fix3a()
    check_fix3b()
    check_fix3c()
    check_fix4()
    check_fix5()
    check_fix6()
    check_fix7()
    check_sanity()
    print("all regression checks passed")

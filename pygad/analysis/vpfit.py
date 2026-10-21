"""
Routines to fit Voigt profiles to absorption line spectra.

This module implements the full Voigt-profile fitting pipeline used in pygad.
The top-level entry point is fit_profiles(), which reads a spectrum (from a
pygad spectrum file or from arrays supplied directly), identifies significant
absorption regions, fits multi-component Voigt/Gaussian/Lorentzian profiles
to each region, and optionally writes the results back to the HDF5 file.

Usage
-----
    # From a pygad spectrum file:
    pygad.analysis.fit_profiles(ion_name, spectrum_file=spectrum_file)

    # From arrays:
    pygad.analysis.fit_profiles(ion_name, l=wavelengths, flux=flux, noise=noise)

Pipeline
--------
    fit_profiles()
        └─ Spectrum.fit_profiles()
               ├─ prepare_spectrum()
               ├─ find_regions()
               └─ fit_profiles_sat()

Last edited: Romeel Dave, July 2026
Contributors: Romeel Dave, Clara Lilje

Doctests:
    >>> environment.verbose = environment.VERBOSE_QUIET
    >>> c_kms = float(c.in_units_of('km/s'))
    >>> l_rest = 1215.6701

    The optical depth of a redshifted line equals that of the same line at
    rest on the corresponding (velocity-matched) grid:
    >>> v = np.linspace(-100., 100., 201)
    >>> tau_0 = model_tau('H1215', [13., 20., l_rest], l_rest * (1. + v / c_kms))
    >>> tau_2 = model_tau('H1215', [13., 20., 3. * l_rest], 3. * l_rest * (1. + v / c_kms))
    >>> print('%.4f %.4f' % (tau_0.max(), tau_2.max()))
    0.3789 0.3789
    >>> bool(np.allclose(tau_0, tau_2, rtol=1e-6))
    True

    Spectrum derives velocities from wavelengths:
    >>> spec = Spectrum('H1215', 2.0, 3. * l_rest * (1. + np.array([0., 100., 200.]) / c_kms),
    ...                 np.ones(3), np.full(3, 0.01), None, gal_velocity_pos=None,
    ...                 logN_bounds=[12, 19], b_bounds=[5, 100])
    >>> print(np.round(np.asarray(spec.velocities, dtype=float), 6))
    [  0. 100. 200.]

    write_spectrum converts velocities (km/s, or any velocity units) to
    wavelengths, and honours overwrite=False:
    >>> import os, shutil, tempfile
    >>> tmpdir = tempfile.mkdtemp()
    >>> spec_name = os.path.join(tmpdir, 'spec')
    >>> def write(vels, overwrite=True, name=spec_name):
    ...     n = len(vels)
    ...     write_spectrum(name, 'H1215', [1., 2.], l_rest, 2.0, vels, np.ones(n),
    ...                    np.zeros(n), np.full(n, 0.01), np.zeros(n), np.zeros(n),
    ...                    np.zeros(n), np.zeros(n), np.zeros(n), overwrite=overwrite)
    ...     with h5py.File(spec_name + '.h5', 'r') as hf:
    ...         return np.array(hf['wavelength'])
    >>> print(np.round(write(np.array([0., 1000.])), 3))
    [3647.01  3659.175]
    >>> print(np.round(write(UnitArr([0., 1e6], 'm/s')), 3))
    [3647.01  3659.175]
    >>> print(np.round(write(np.array([0., 2000.]), overwrite=False), 3))
    [3647.01  3659.175]

    '.h5' is only appended when missing, and write_line_list accepts either
    form of the name:
    >>> print(np.round(write(np.array([0., 1000.]), name=spec_name + '.h5'), 3))
    [3647.01  3659.175]
    >>> sorted(os.listdir(tmpdir))
    ['spec.h5']
    >>> ll = {'region': [0], 'N': [13.], 'dN': [0.1], 'b': [20.], 'db': [1.],
    ...       'l': [3. * l_rest], 'dl': [0.01], 'EW': [0.1], 'Chisq': [1.]}
    >>> write_line_list(spec_name, ll, [[3646., 3648.]], [[0, 1]], mode='Gaussian')
    >>> with h5py.File(spec_name + '.h5', 'r') as hf:
    ...     print(hf['line_list'].attrs['mode'], len(hf['line_list/logN']))
    Gaussian 1
    >>> shutil.rmtree(tmpdir)

    Fitting a full-box (periodic) spectrum returns line centres on the input
    grid, also for a line crossing the box edge (at v=995 km/s, with the box
    spanning 0-1000 km/s), and the (periodic) model fits the input:
    >>> vel = (np.arange(400) + 0.5) * 2.5
    >>> l = 3. * l_rest * (1. + vel / c_kms)
    >>> true = [(13.5, 25., 150.), (13., 20., 995.)]
    >>> tau = np.zeros(len(vel))
    >>> for logN, b, v0 in true:
    ...     for shift in (-1000., 0., 1000.):
    ...         tau += model_tau('H1215', [logN, b, 3. * l_rest * (1. + (v0 + shift) / c_kms)], l)
    >>> flux, noise = np.exp(-tau), np.full(len(vel), 0.01)
    >>> line_list = fit_profiles('H1215', l=l, vel=vel, flux=flux, noise=noise,
    ...                          logN_bounds=[12, 15], b_bounds=[5, 100])
    Fitting profiles for H1215 spectrum at z=2 with 400 pixels
    >>> v_fit = wave_to_vel(np.asarray(line_list['l']), l_rest, 2.0)
    >>> for (logN, b, v0), i in zip(true, np.argsort(v_fit)):
    ...     print(abs(line_list['N'][i] - logN) < 0.05, abs(v_fit[i] - v0) < 5.)
    True True
    True True
    >>> spec = Spectrum('H1215', 2.0, l, flux, noise, vel, gal_velocity_pos=None,
    ...                 logN_bounds=[12, 15], b_bounds=[5, 100])
    >>> spec.fit_profiles(vel_range=0)
    >>> spec.wrap_pixel > 0, bool(np.array_equal(spec.fluxes, flux))
    (True, True)
    >>> spec.get_fluxes_model()
    >>> bool(np.mean(((spec.fluxes_model - flux) / noise)**2) < 2.)    # periodic model
    True

    The profile shape (`mode`) is used for fitting and for the model:
    >>> from pygad.analysis.tau_lookup import _tau_lookup_cache
    >>> _ = fit_profiles('H1215', l=l, vel=vel, flux=flux, noise=noise,
    ...                  logN_bounds=[12, 15], b_bounds=[5, 100], mode='Gaussian')
    Fitting profiles for H1215 spectrum at z=2 with 400 pixels
    >>> ('H1215', 'Gaussian') in _tau_lookup_cache
    True
    >>> voigt_model = spec.fluxes_model
    >>> spec.mode = 'Lorentzian'
    >>> spec.get_fluxes_model()
    >>> bool(np.max(np.abs(spec.fluxes_model - voigt_model)) > 0.1)
    True
    >>> environment.verbose = environment.VERBOSE_NORMAL
"""

__all__ = [
    "Spectrum",
    "fit_profiles",
    "model_tau",
    "fit_profiles_sat",
    "find_regions",
    "EquivalentWidth",
    "write_spectrum",
    "write_line_list",
]

import numpy as np
import pylab as plt

from .. import environment
from ..physics import c
from ..units import Unit, UnitArr, UnitQty, UnitScalar
from .absorption_spectra import line_profile, lines, thermal_b_param
import os
import h5py
from .physics import wave_to_vel, vel_to_wave, tau_to_flux
from .utils import read_h5_into_dict
from scipy import signal
import scipy
from scipy.optimize import least_squares
plt.rcParams['text.usetex'] = False


def find_peaks(flux_data, wavelength_data, min_height, distance):
    """
    Identify absorption troughs in a normalised flux spectrum.

    Inverts the flux (1 − flux) and applies scipy.signal.find_peaks to
    locate significant absorption features, together with their widths.

    Args:
        flux_data (numpy array):        Normalised flux array.
        wavelength_data (numpy array):  Wavelengths corresponding to flux_data (Å).
        min_height (float):             Minimum height in the inverted flux,
                                        i.e. minimum absorption depth (0–1).
        distance (int):                 Minimum separation between peaks (pixels).

    Returns:
        index (numpy array):   Pixel indices of detected absorption peaks.
        height (dict):         Peak properties from scipy.signal.find_peaks,
                               containing at least 'peak_heights'.
        widths (numpy array):  Width of each peak at half-prominence (pixels).
    """
    inverse_flux = 1 - flux_data
    index, height = scipy.signal.find_peaks(
        inverse_flux, height=min_height, distance=distance
    )
    widths = scipy.signal.peak_widths(inverse_flux, index)[0]
    wavelength_subset = wavelength_data[index]
    return index, height, widths


class Spectrum(object):
    """
    Container for a single absorption-line spectrum and its Voigt-profile fit.

    Stores the spectrum arrays (wavelengths, fluxes, noise, velocities) together
    with fitting configuration, and provides methods to extract a fitting window,
    run the profile fitter, and visualise the results.

    Typical usage::

        spec = Spectrum('HI1215', redshift, wavelengths, flux, noise, vel,
                        gal_velocity_pos=500., logN_bounds=[12, 19],
                        b_bounds=[5, 200])
        spec.fit_profiles(vel_range=600.)
        spec.plot_fit()
    """

    def __init__(self, ion_name, redshift, l, flux, noise, vel, **kwargs):
        """
        Initialise a Spectrum object for Voigt-profile fitting.

        Either l (wavelengths) or vel (velocities) must be supplied; the other
        is computed internally from the rest wavelength and redshift.  Any
        additional keyword arguments are stored directly as instance attributes,
        making it straightforward to pass fitting parameters at construction time.

        Args:
            ion_name (str):       Line identifier, e.g. 'HI1215', as listed in
                                  absorption_spectra.lines.
            redshift (float):     Redshift of the absorbing system.
            l (array-like):       Wavelength array (Å).  May be None when vel
                                  is provided.
            flux (array-like):    Normalised flux array.
            noise (array-like):   1-sigma noise array.
            vel (array-like):     Velocity array (km/s).  May be None when l
                                  is provided.
            **kwargs:             Optional keyword arguments stored as instance
                                  attributes.  Common examples:
                                    logN_bounds (list):    [min, max] log₁₀ column
                                                           density range.
                                    b_bounds (list):       [min, max] Doppler
                                                           parameter (km/s).
                                    snr (float):           Signal-to-noise ratio.
                                    gal_velocity_pos (float): Galaxy LOS velocity
                                                           (km/s).
                                    mode (str):            Profile shape used for
                                                           fitting and models:
                                                           'Voigt' (default),
                                                           'Gaussian' or
                                                           'Lorentzian'.
        """
        # assign optional keyword arguments
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.mode = getattr(self, 'mode', 'Voigt')
        # look up line to fit
        if isinstance(ion_name, str):
            line = lines[ion_name]
            lambda_rest = UnitScalar(line["l"], 'Angstrom')
        else:
            raise ValueError('Could not find line %s in database.' % ion_name)
        # compute wavelengths or velocities from the other
        if vel is not None:
            wave = lambda_rest * (redshift + 1.0) * (1.0 + vel / float(c.in_units_of('km/s')))
        elif l is not None:
            vel = (l / (lambda_rest * (redshift + 1.0)) - 1.0) * float(c.in_units_of('km/s'))
            wave = l
        else:
            raise ValueError('One of l or vel must be provided.')
        # load spectrum
        self.ion_name = ion_name
        self.lambda_rest = lambda_rest
        self.wavelengths = wave
        self.fluxes = flux
        self.noise = noise
        self.redshift = redshift
        self.velocities = vel
        self.continuum = np.ones(len(self.wavelengths))
        self.verbose = (environment.verbose > environment.VERBOSE_QUIET)
        if self.verbose:
            print('N,b bounds:', getattr(self, 'logN_bounds', None),
                  getattr(self, 'b_bounds', None))

    def get_initial_window(self, vel_range, v_central=None):
        """
        Compute the pixel window covering ±vel_range around a central velocity.

        If self.gal_velocity_pos is set it is used as v_central unless an
        explicit value is provided.  Handles periodic box geometry when
        v_start wraps below zero.

        Args:
            vel_range (float):   Half-width of the velocity window (km/s).
            v_central (float):   Centre of the window (km/s).  Defaults to
                                 self.gal_velocity_pos when that attribute exists.

        Returns:
            i_start (int):  Index of the first pixel in the window.
            i_end (int):    Index one past the last pixel in the window.
            N (int):        Total number of pixels in the window.
        """
        def _find_nearest(array, value):
            return np.abs(array - value).argmin()

        if self.gal_velocity_pos is not None:
            v_central = self.gal_velocity_pos

        dv = self.velocities[1] - self.velocities[0]
        v_start = v_central - vel_range
        v_end = v_central + vel_range
        N = int((v_end - v_start) / dv)

        # velocities is assumed to span the entire simulation box
        v_boxsize = self.velocities[-1] - self.velocities[0] + 0.5 * dv

        if v_start < 0.:
            v_start += v_boxsize
        i_start = _find_nearest(self.velocities, v_start)
        i_end = i_start + N

        return i_start, i_end, N

    def extend_to_continuum(self, i_start, i_end, N, contin_level=None):
        """
        Expand a pixel window outward until both edges reach the continuum level.

        Walks i_start downward and i_end upward one pixel at a time until the
        flux at each boundary is within 2 % of contin_level.  Uses periodic
        (wrap-around) indexing so the scan is safe near the spectrum edges.

        Args:
            i_start (int):        Current start pixel index.
            i_end (int):          Current end pixel index.
            N (int):              Current window width in pixels.
            contin_level (float): Target continuum flux value.  Defaults to
                                  self.continuum[0] when not provided.

        Returns:
            i_start (int):  Updated start pixel index.
            i_end (int):    Updated end pixel index.
            N (int):        Updated window width in pixels.
        """
        if contin_level is None:
            contin_level = self.continuum[0]

        continuum = False
        while not continuum:
            _flux = self.fluxes.take(i_start, mode='wrap')
            if np.abs(_flux - contin_level) / contin_level > 0.02:
                i_start -= 1
                N += 1
            else:
                continuum = True

        continuum = False
        while not continuum:
            _flux = self.fluxes.take(i_end, mode='wrap')
            if np.abs(_flux - contin_level) / contin_level > 0.02:
                i_end += 1
                N += 1
            else:
                continuum = True

        return i_start, i_end, N

    def buffer_with_continuum(self, waves, flux, noise, nbuffer=50):
        """
        Pad wavelength/flux/noise with continuum-level pixels at each end.

        Extends the wavelength axis by extrapolating with the pixel spacing at
        each edge and sets the padded flux values to 1 (continuum).  The buffer
        helps the Voigt fitter identify the baseline and avoid boundary effects.

        Args:
            waves (numpy array):   Wavelength array (Å).
            flux (numpy array):    Normalised flux array.
            nbuffer (int):         Number of continuum pixels to add at each end.

        Returns:
            waves (numpy array):  Extended wavelength array of length
                                  len(waves) + 2*nbuffer.
            flux (numpy array):   Extended flux array of length
                                  len(flux) + 2*nbuffer, with flux = 1 at
                                  both padded ends.
        """
        dl = waves[1] - waves[0]
        l_start = np.arange(waves[0] - dl*nbuffer, waves[0], dl)
        l_end = np.arange(waves[-1]+dl, waves[-1] + dl*(nbuffer+1), dl)

        waves = np.concatenate((l_start, waves, l_end))
        flux = np.concatenate((
            np.ones(nbuffer),
            flux,
            np.ones(nbuffer)
        ))
        mean_noise = np.sum(noise)/len(noise)
        new_noise = mean_noise * np.ones(2*nbuffer)
        noise = np.concatenate((
            new_noise[:nbuffer],
            noise,
            new_noise[nbuffer:]
        ))

        return waves, flux, noise

    def periodic_wrap(self):
        """
        Roll the spectrum so that the highest-flux (lowest-absorption) pixel is first.

        To avoid fitting an absorption feature that straddles the edge of the
        simulation box, this routine reorders the flux and noise arrays so that
        the pixel with the highest flux is at index 0.  Only flux and noise are
        re-ordered; the wavelength array is unchanged.  Call
        periodic_unwrap_wavelength() after fitting to restore correct wavelengths.

        Should only be used for spectra that span the entire periodic simulation
        volume.

        Returns:
            flux (numpy array):      Periodically wrapped flux array.
            noise (numpy array):     Periodically wrapped 1-sigma noise array.
            starting_pixel (int):    Original index of the highest-flux pixel;
                                     required by periodic_unwrap_wavelength().
        """
        l = self.wavelengths
        flux = self.fluxes
        noise = self.noise
        starting_pixel = np.argmax(flux)
        flux  = np.concatenate((flux[starting_pixel:],  flux[:starting_pixel]))
        noise = np.concatenate((noise[starting_pixel:], noise[:starting_pixel]))
        if environment.verbose >= environment.VERBOSE_TACITURN:
            print("Periodically wrapping spectrum, starting_pixel= %d" % starting_pixel)
        return flux, noise, starting_pixel

    def periodic_unwrap_wavelength(self, l=None):
        """
        Map wavelengths measured on the periodically wrapped spectrum back to
        the original (unwrapped) wavelength grid.

        After periodic_wrap() the rolled flux/noise arrays are fitted against
        the unchanged self.wavelengths, so a feature at wavelength l in the
        fit lies wrap_pixel pixels further along in the original spectrum.
        This routine applies that shift and wraps values that overflow the
        right-hand edge of the (periodic) box back to the left.  The
        wavelength grid is assumed to be uniform, as it is for spectra that are
        uniform in velocity.

        Args:
            l (array-like):   Wavelengths (Å) on the wrapped spectrum, e.g.
                              fitted line centres.  Defaults to
                              self.wavelengths.

        Returns:
            l (numpy array):  Corresponding wavelengths (Å) on the original
                              grid, within [l_0, l_0 + box width).
        """
        waves = np.asarray(self.wavelengths, dtype=float)
        if l is None:
            l = waves
        l = np.asarray(l, dtype=float)
        l_width = len(waves) * (waves[1] - waves[0])
        shift = waves[self.wrap_pixel] - waves[0]
        return waves[0] + np.mod(l - waves[0] + shift, l_width)

    def prepare_spectrum(self, vel_range, do_continuum_buffer=True, nbuffer=50):
        """
        Extract and prepare the portion of the spectrum to be fitted.

        If self.gal_velocity_pos is set, a velocity window of ±vel_range around
        the galaxy is extracted and extended outward to the continuum level at
        both edges.  Otherwise the full spectrum is periodically wrapped so that
        the highest-flux pixel is at the start.  Optionally pads each end with
        continuum pixels and builds a flat noise array from the SNR.

        Sets the following instance attributes after returning:
            waves_fit (numpy array):   Wavelengths of the region to fit (Å).
            fluxes_fit (numpy array):  Fluxes of the region to fit.
            noise_fit (numpy array):   1-sigma noise array for the region.
            wrap_pixel (int):          Starting pixel used by periodic_wrap()
                                       (0 when gal_velocity_pos is set).

        Args:
            vel_range (float):           Half-width of the velocity window (km/s).
                                         Ignored when gal_velocity_pos is None.
            do_continuum_buffer (bool):  Pad each end with nbuffer continuum pixels
                                         before fitting.
            nbuffer (int):               Number of buffer pixels per end when
                                         do_continuum_buffer is True.
        """
        if self.gal_velocity_pos is not None:
            i_start, i_end, N = self.get_initial_window(vel_range)
            i_start, i_end, N = self.extend_to_continuum(i_start, i_end, N)
            self.wrap_pixel = 0
            if i_start < 0:
                i_start += len(self.wavelengths)
                i_end += len(self.wavelengths)
            fluxes, noise = self.fluxes, self.noise
        else:
            # keep self.fluxes/self.noise on the original grid; only the
            # arrays handed to the fitter are wrapped
            fluxes, noise, self.wrap_pixel = self.periodic_wrap()
            i_start = 0
            i_end = len(self.wavelengths)
            N = i_end - i_start

        self.waves_fit  = self.wavelengths.take(range(i_start, i_end), mode='wrap')
        self.fluxes_fit = fluxes.take(range(i_start, i_end), mode='wrap')
        self.noise_fit = noise.take(range(i_start, i_end), mode='wrap')

        i_wrap = len(self.wavelengths) - i_start
        wave_boxsize = self.wavelengths[-1] - self.wavelengths[0]
        dl = self.wavelengths[1] - self.wavelengths[0]
        if i_wrap < N:
            self.waves_fit[i_wrap:] += wave_boxsize + dl

        if do_continuum_buffer is True:
            self.waves_fit, self.fluxes_fit, self.noise_fit = self.buffer_with_continuum(
                self.waves_fit, self.fluxes_fit, self.noise_fit, nbuffer=nbuffer
            )


    def fit_periodic_spectrum(self, vel_range, **kwargs):
        """
        Fit Voigt profiles to a spectrum that spans the entire simulation volume.

        Fits the whole periodically wrapped spectrum (see fit_profiles() with
        gal_velocity_pos unset), adds the line velocities as
        self.line_list['v'], and, if self.gal_velocity_pos is set, keeps only
        the lines within ±vel_range of it (measured periodically).

        Args:
            vel_range (float):  Half-width (km/s) of the velocity window around
                                self.gal_velocity_pos whose lines are kept.
            **kwargs:           Passed on to fit_profiles().

        Sets:
            self.line_list (dict): Best-fit line parameters with keys
                                   ['N', 'dN', 'b', 'db', 'l', 'dl', 'EW',
                                   'Chisq', 'region', 'v'].
        """
        gal_velocity_pos = getattr(self, 'gal_velocity_pos', None)
        self.gal_velocity_pos = None
        try:
            self.fit_profiles(vel_range, **kwargs)
        finally:
            self.gal_velocity_pos = gal_velocity_pos
        line_list = {k: np.asarray(v) for k, v in self.line_list.items()}
        line_list['v'] = np.asarray(wave_to_vel(
            line_list['l'], float(self.lambda_rest), self.redshift
        ), dtype=float)
        if gal_velocity_pos is not None:
            vel = np.asarray(self.velocities, dtype=float)
            v_box = len(vel) * (vel[1] - vel[0])
            dv = np.mod(line_list['v'] - gal_velocity_pos + 0.5 * v_box, v_box) - 0.5 * v_box
            keep = np.abs(dv) < vel_range
            line_list = {k: v[keep] for k, v in line_list.items()}
        self.line_list = line_list

    def get_tau_model(self):
        """
        Compute the total model optical depth from all fitted absorption lines.

        Sums contributions from every line in self.line_list over the full
        wavelength array self.wavelengths using model_tau().  Without
        gal_velocity_pos the spectrum is a periodic full box, so lines near
        one edge also absorb at the other.

        Sets:
            self.tau_model (numpy array): Optical depth at each wavelength pixel.
        """
        self.tau_model = np.zeros(len(self.wavelengths))
        for i in range(len(self.line_list["N"])):
            p = np.array([self.line_list["N"][i], self.line_list["b"][i], self.line_list["l"][i]])
            self.tau_model += _model_tau_periodic(self.ion_name, p, self.wavelengths,
                                                  self.mode, periodic=self._periodic)

    @property
    def _periodic(self):
        """Whether the spectrum is treated as a periodic full box."""
        return getattr(self, 'gal_velocity_pos', None) is None

    def get_fluxes_model(self):
        """
        Compute the total model flux from all fitted absorption lines.

        Calls get_tau_model() to compute self.tau_model, then converts to
        normalised flux using tau_to_flux().

        Sets:
            self.fluxes_model (numpy array): Normalised model flux at each pixel.
        """
        self.get_tau_model()
        self.fluxes_model = tau_to_flux(self.tau_model)

    def fit_profiles(self, vel_range, do_continuum_buffer=True, nbuffer=50,
                     chisq_lim=2.0, 
                     chisq_factor=0.95, N_sigma_constr=3.0, max_lines=12,
                     mode=None):
        """
        Prepare the spectrum and fit Voigt profiles to all absorption regions.

        Orchestrates the full fitting pipeline:
            1. prepare_spectrum()  — extract/wrap the spectral window,
                                     optionally pad with continuum, build noise.
            2. find_regions()      — identify pixels with significant absorption.
            3. fit_profiles_sat()  — fit multi-component profiles per region.
            4. periodic_unwrap_wavelength() — restore wavelengths if wrapped.

        Sets the following instance attributes after returning:
            self.line_list (dict): Best-fit line parameters.  Line centres 'l'
                                   are on the original wavelength grid.
            self.regions_l (numpy array): Wavelength boundaries of each region.
            self.regions_i (numpy array): Pixel boundaries of each region.
            For a periodically wrapped spectrum both region arrays refer to the
            original grid, and a region crossing the box edge has start > end.

        Args:
            vel_range (float):           Half-width of the velocity window (km/s).
            do_continuum_buffer (bool):  Pad each end with nbuffer continuum pixels.
            nbuffer (int):               Number of continuum buffer pixels per end.
            chisq_lim (float):           Not used any more; see fit_profiles_sat().
            chisq_factor (float):        A new line is accepted only when it reduces
                                         χ² by at least this factor (must be ≤ 1).
            N_sigma_constr (float):      Not used any more; see fit_profiles_sat().
            max_lines (int):             Maximum Voigt components allowed per region.
            mode (str):                  Profile shape ('Voigt', 'Gaussian' or
                                         'Lorentzian'); sets self.mode.  Defaults
                                         to self.mode.
        """
        if mode is not None:
            self.mode = mode
        self.prepare_spectrum(vel_range, do_continuum_buffer=do_continuum_buffer,
                              nbuffer=nbuffer)
        self.regions_l, self.regions_i = find_regions(
            self.waves_fit, self.fluxes_fit, self.noise_fit, verbose=self.verbose
        )
        self.line_list = fit_profiles_sat(
            self.ion_name, self.waves_fit, self.fluxes_fit, self.noise_fit,
            self.regions_l, self.regions_i,
            chisq_lim=chisq_lim, chisq_factor=chisq_factor,
            N_sigma_constr=N_sigma_constr,
            max_lines=max_lines,
            logN_bounds=self.logN_bounds,
            b_bounds=self.b_bounds, mode=self.mode, verbose=self.verbose
        )
        if self.wrap_pixel > 0:
            # the fit was done on the rolled spectrum: move everything back
            self.line_list["l"] = self.periodic_unwrap_wavelength(self.line_list["l"])
            if len(self.regions_l) > 0:
                self.regions_l = self.periodic_unwrap_wavelength(self.regions_l)
                i_offset = self.wrap_pixel - (nbuffer if do_continuum_buffer else 0)
                self.regions_i = np.mod(np.asarray(self.regions_i) + i_offset,
                                        len(self.wavelengths))

    def plot_fit(self, ax=None):
        """
        Plot the data spectrum, individual fitted profiles, and the combined model.

        Draws on the supplied axes (or a new figure):
            • Data flux as a solid grey line.
            • Each individual fitted profile as a semi-transparent dashed line.
            • A short red tick at the top of the plot at each line's centre.
            • The total combined model flux as a dashed pink line.

        Args:
            ax (matplotlib.axes.Axes | None): Axes to draw into.  When None
                                              (default) a new figure is created,
                                              shown and closed; given axes are
                                              left to the caller.

        Returns:
            ax (matplotlib.axes.Axes):        The axes drawn into.
        """
        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=(12,3))

        x_val = self.wavelengths
        ax.plot(x_val, self.fluxes, label='data', c='tab:grey', lw=2, ls='-')

        self.get_fluxes_model()
        for i in range(len(self.line_list['N'])):
            p = np.array([self.line_list['N'][i], self.line_list['b'][i], self.line_list['l'][i]])
            _tau_model = _model_tau_periodic(self.ion_name, p, self.wavelengths,
                                             self.mode, periodic=self._periodic)
            line_obj, = ax.plot(x_val, tau_to_flux(_tau_model), alpha=0.5, lw=1, ls='--')
            l_cent = self.line_list['l'][i]
            ax.axvline(l_cent, ymin=0.95, ymax=0.98, color=line_obj.get_color(), linestyle='-', linewidth=1)

        ax.plot(x_val, self.fluxes_model, label='model', c='tab:pink', ls='--', lw=2)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(x_val[0], x_val[-1])
        ax.legend(loc='best', fontsize=8)
        if own_fig:
            plt.show()
            plt.close(fig)
        return ax


def fit_profiles_sat(
    ion_name,
    l,
    flux,
    noise,
    regions_l,
    regions_i,
    chisq_lim=2,
    chisq_factor=0.95,
    N_sigma_constr=3.0,
    max_lines=12,
    mode="Voigt",
    logN_bounds=[8, 20],
    b_bounds=[1, 300],
    verbose=False
):
    """
    Fit Voigt/other profiles to a set of pre-identified absorption regions.

    Each region starts with a single component; components are added one at
    a time (at the deepest residual, or by splitting an existing component)
    and all components of the region are refitted by least squares after
    every change.  A component is only kept when it lowers the total χ² by at
    least _DCHISQ_PER_LINE (the Akaike information criterion for its three
    parameters); afterwards insignificant components are removed or merged.
    Finally all components of the spectrum are refitted jointly, so the wings
    of neighbouring regions are accounted for.

    Line centres are fitted as velocity offsets, which keeps the problem well
    conditioned.  For a single saturated line in the flat part of the curve of
    growth (HI with 10^14.5 ≲ N ≲ 10^18 cm⁻²) N and b are strongly correlated
    and only weakly constrained by one transition; blends of such lines can be
    indistinguishable from one broader line.  Use the errors dN, db.

    Args:
        ion_name (str):             The line to fit as listed in
                                    analysis.absorption_spectra.lines,
                                    e.g. 'HI1215'.
        l (array-like):             Wavelengths of the full input spectrum (Å).
        flux (array-like):          Normalised flux at each wavelength.
        noise (array-like):         1-sigma noise at each wavelength.  Must be > 0.
        regions_l (numpy array):    Shape (n_regions, 2); start and end wavelengths
                                    of each absorption region as returned by
                                    find_regions().
        regions_i (numpy array):    Shape (n_regions, 2); start and end pixel
                                    indices of each absorption region.
        chisq_lim (float):          Not used any more (components are added
                                    while they are significant); kept for
                                    backwards compatibility.
        chisq_factor (float):       A new component must also reduce the total
                                    χ² of its region by at least this factor (≤ 1).
        N_sigma_constr (float):     Not used any more (the fit is unconstrained
                                    apart from the parameter bounds); kept for
                                    backwards compatibility.
        max_lines (int):            Maximum components per absorption region.
        mode (str):                 Profile shape: 'Voigt', 'Gaussian', or
                                    'Lorentzian'.  See absorption_spectra.line_profile().
        logN_bounds (list):         [min, max] allowed log₁₀ column density (cm⁻²).
        b_bounds (list):            [min, max] allowed Doppler parameter (km/s).
        verbose (bool):             Print progress information to stdout.

    Returns:
        line_list (dict):  Best-fit parameters for all detected lines, with keys:
                             'region' (int)   — absorption region index.
                             'N'      (float) — log₁₀ column density (cm⁻²).
                             'dN'     (float) — 1-sigma uncertainty on N
                                                (from the final joint fit).
                             'b'      (float) — Doppler parameter (km/s).
                             'db'     (float) — 1-sigma uncertainty on b.
                             'l'      (float) — central wavelength (Å).
                             'dl'     (float) — 1-sigma uncertainty on l.
                             'EW'     (float) — equivalent width (Å).
                             'Chisq'  (float) — reduced χ² of the final model
                                                over the line's region.
    """
    from .tau_lookup import get_tau_lookup, model_tau_fast
    _lookup = get_tau_lookup(ion_name, mode)

    line = lines[ion_name]
    l0 = float(line["l"].split()[0])
    if isinstance(l, np.ndarray) or l.units in [1, None]:
        l = UnitArr(l, "Angstrom")
    l     = np.asarray(l,     dtype=float)
    flux  = np.asarray(flux,  dtype=float).copy()
    noise = np.asarray(noise, dtype=float).copy()
    # pixels without a usable flux or noise do not constrain the fit
    bad = ~(np.isfinite(flux) & np.isfinite(noise) & (noise > 0))
    flux[bad], noise[bad] = 1.0, 1e30
    # the lookup table clamps b to its range, where the model is flat in b
    b_bounds = [max(b_bounds[0], _lookup.b_min), min(b_bounds[1], _lookup.b_max)]

    # ── Inner helper functions ────────────────────────────────────────────

    def _tau_to_flux(tau):
        """Convert optical depth to flux, clipping tau to avoid over/underflow."""
        return np.exp(-np.clip(tau, -50, 50))

    def _chisq(p, l, flux, noise, ion_name, mode):
        """
        Compute the reduced chi-squared between data and model flux.

        Args:
            p (numpy array):    Flat parameter array [logN, b, λ₀, …].
            l (numpy array):    Wavelength grid (Å).
            flux (numpy array): Observed normalised flux.
            noise (numpy array): 1-sigma noise.
            ion_name (str):     Line identifier.
            mode (str):         Profile type.

        Returns:
            chisq (float): Reduced χ² (sum of squared residuals / non-zero pixels).
        """
        model_flux = _tau_to_flux(model_tau_fast(ion_name, p, l, mode, lookup=_lookup))
        dx_array = (flux - model_flux) / noise
        return np.sum(dx_array * dx_array) / max(np.count_nonzero(dx_array), 1)

    def _add_line(p, l, flux, noise):
        """
        Append a new component to the flat parameter array.

        The initial (logN, b) come from _grow_line() and the line is centred
        at the deepest point of the current residual flux.

        Args:
            p (numpy array):     Current flat parameter array [logN, b, λ₀, …].
            l (numpy array):     Wavelength array of the fitting region (Å).
            flux (numpy array):  Observed flux in the fitting region.
            noise (numpy array): 1-sigma noise in the fitting region.

        Returns:
            p (numpy array): Updated parameter array with the new line appended.
        """
        if len(p) == 0:
            resid = flux
        else:
            resid = 1.0 + flux - _tau_to_flux(model_tau_fast(ion_name, p, l, mode, lookup=_lookup))
        n_guess, b_guess, l_guess = _grow_line(ion_name, l, flux, noise, resid, l0, mode)
        if len(p) > 0 and np.any(np.isclose(np.asarray(p)[2::3], l_guess, rtol=0., atol=1e-9)):
            # never start two components at exactly the same position
            l_guess += 0.5 * (l[1] - l[0])
        return np.append(p, [n_guess, b_guess, l_guess])

    def _grow_line(ion_name, l, flux, noise, resid, l0, mode,
                   i_line=None, floor_sigma=1.5, smooth_sigma=1., unsat_sigma=3.):
        """
        Find the largest (logN, b) component that does not violate a flux floor.

        Searches a 100 × 40 grid of (logN, b) values.  For each b the unit
        optical-depth profile is computed once via the lookup table and then
        linearly scaled for each logN, giving 40× fewer model evaluations than
        calling model_tau_fast for every (logN, b) pair.  The combination with
        the lowest reduced χ² near the line core is returned, provided the
        model flux everywhere exceeds smoothed_residual − floor_sigma × noise.
        This is only the starting point; the least-squares fit refines it.

        Args:
            ion_name (str):       Line identifier.
            l (numpy array):      Wavelength array of the fitting region (Å).
            flux (numpy array):   Observed flux in the fitting region.
            noise (numpy array):  1-sigma noise in the fitting region.
            resid (numpy array):  Current residual flux (1 + flux − model_flux).
            l0 (float):           Rest wavelength of the line (Å).
            mode (str):           Profile type.
            i_line (int | None):  Pixel index at which to centre the new line.
                                  Uses the deepest smoothed-residual pixel when None.
            floor_sigma (float):  The model must exceed
                                  smoothed_resid − floor_sigma × noise everywhere.
            smooth_sigma (float): Gaussian smoothing width (pixels) applied to the
                                  residual before locating the line centre.
            unsat_sigma (float):  Pixels with flux below unsat_sigma × noise are
                                  treated as saturated when finding the line centre.

        Returns:
            best_N (float):  Best-fit log₁₀ column density (cm⁻²).
            best_b (float):  Best-fit Doppler parameter (km/s).
            l_line (float):  Central wavelength of the new line (Å).
        """
        smoothed = scipy.ndimage.gaussian_filter1d(resid, smooth_sigma) if smooth_sigma > 0. else resid
        if i_line is None:
            i_line = np.argmin(smoothed)
        l_line = l[i_line]

        smoothed  = np.minimum(smoothed, 1.0)
        floor     = smoothed - floor_sigma * noise

        N_range = np.linspace(logN_bounds[0], logN_bounds[1], 100)
        b_range = np.logspace(np.log10(b_bounds[0]), np.log10(b_bounds[1]), 40)

        best_chisq = 1.e20
        best_N, best_b = logN_bounds[0], b_bounds[0]

        for bpar in b_range:
            # Compute the profile shape once per b value via the lookup table;
            # subsequent N values need only a scalar multiply (no model call).
            tau_unit   = model_tau_fast(ion_name, [0.0, bpar, l_line], l, mode, lookup=_lookup)

            for Ncol in N_range:
                tau_trial  = (10.0 ** Ncol) * tau_unit
                model_flux = np.exp(-np.clip(tau_trial, -50, 50))
                diff = model_flux - floor
                if np.any(diff < 0):
                    continue

                # Evaluate chi-sq only near the line core (within half-depth)
                i_lo = i_line
                while i_lo > 0           and (1.-model_flux[i_lo])  > 0.5*(1.-model_flux[i_line]): i_lo -= 1
                i_hi = i_line
                while i_hi < len(l) - 2  and (1.-model_flux[i_hi])  > 0.5*(1.-model_flux[i_line]): i_hi += 1
                sl = slice(i_lo, i_hi + 1)
                dx = (resid[sl] - model_flux[sl]) / noise[sl]
                nz = np.count_nonzero(dx)
                if nz == 0:
                    continue
                chi2 = np.sum(dx * dx) / nz
                if chi2 < best_chisq:
                    best_chisq, best_N, best_b = chi2, Ncol, bpar

        return best_N, best_b, l_line

    def _split_line(p, i):
        """Replace component i by two narrower, weaker ones at λ ± b/2."""
        logN, b, l_c = p[3*i:3*i+3]
        dl = 0.5 * b / _lookup.c_kms * l_c
        new = [logN - 0.3, 0.7 * b, l_c - dl, logN - 0.3, 0.7 * b, l_c + dl]
        return np.append(np.delete(p, [3*i, 3*i+1, 3*i+2]), new)

    def _fit(p, l, flux, noise, l_bounds=None):
        """Least-squares fit of all components in p; see _lsq_fit_lines()."""
        return _lsq_fit_lines(_lookup, p, l, flux, noise, logN_bounds, b_bounds,
                              l_bounds=l_bounds)

    def _chisq_tot(p, l, flux, noise):
        """Total (not reduced) χ² of the model p."""
        model_flux = _tau_to_flux(model_tau_fast(ion_name, p, l, mode, lookup=_lookup))
        return float(np.sum(((flux - model_flux) / noise) ** 2))

    def _prune(p, l, flux, noise, label):
        """
        Remove or merge components that are not significant (AIC).

        Each component is tentatively removed (also the last one), and each
        pair of neighbouring components tentatively merged into one; the rest
        is refitted, and the simpler model is kept when it raises the total χ²
        by less than _DCHISQ_PER_LINE.  Repeats until nothing changes.

        Returns:
            p (numpy array): The pruned, refitted parameter array.
        """
        chi2 = _chisq_tot(p, l, flux, noise)
        changed = True
        while changed and len(p) > 0:
            changed = False
            n_lines = len(p) // 3
            trials = []
            for i in np.argsort(p[0::3]):              # weakest first
                trials.append(("removed line %d (N=%.3f)" % (i, p[3*i]),
                               np.delete(p, [3*i, 3*i+1, 3*i+2])))
            order = np.argsort(p[2::3])
            for j in range(n_lines - 1):
                i, k = order[j], order[j+1]
                N_i, N_k = 10**p[3*i], 10**p[3*k]
                merged = [np.log10(N_i + N_k),
                          (N_i * p[3*i+1] + N_k * p[3*k+1]) / (N_i + N_k),
                          (N_i * p[3*i+2] + N_k * p[3*k+2]) / (N_i + N_k)]
                trial = np.delete(p, [3*i, 3*i+1, 3*i+2, 3*k, 3*k+1, 3*k+2])
                trials.append(("merged lines %d and %d (N=%.3f, %.3f)" % (i, k, p[3*i], p[3*k]),
                               np.append(trial, merged)))
            for what, trial in trials:
                trial, _ = _fit(trial, l, flux, noise)
                chi2_trial = _chisq_tot(trial, l, flux, noise)
                if chi2_trial - chi2 < _DCHISQ_PER_LINE:
                    if verbose:
                        print("%s: %s, chisq_tot %.1f -> %.1f" % (label, what, chi2, chi2_trial))
                    p, chi2, changed = trial, chi2_trial, True
                    break
        return p

    # ── Fit each region ───────────────────────────────────────────────────

    line_list = {
        "region": np.array([], dtype=int),
        "l": np.array([]),
        "dl": np.array([]),
        "b": np.array([]),
        "db": np.array([]),
        "N": np.array([]),
        "dN": np.array([]),
        "EW": np.array([]),
        "Chisq": np.array([])
    }
    params_all = np.array([])
    region_of  = np.array([], dtype=int)

    for ireg in range(len(regions_l)):
        l_reg = l[regions_i[ireg, 0] : regions_i[ireg, 1]]
        f_reg = flux[regions_i[ireg, 0] : regions_i[ireg, 1]]
        n_reg = noise[regions_i[ireg, 0] : regions_i[ireg, 1]]
        if len(l_reg) < 4:
            continue

        # Add components one at a time and refit all of them, until a new
        # component is no longer significant.  A new component either goes at
        # the deepest residual or comes from splitting an existing one (for
        # blends whose residual has no clear minimum); the best fit is kept.
        params, _ = _fit(_add_line(np.array([]), l_reg, f_reg, n_reg), l_reg, f_reg, n_reg)
        chi2_tot = _chisq_tot(params, l_reg, f_reg, n_reg)
        while len(params) // 3 < max_lines:
            candidates = [_add_line(params, l_reg, f_reg, n_reg)]
            candidates += [_split_line(params, i) for i in range(len(params) // 3)]
            trial, chi2_trial = None, np.inf
            for cand in candidates:
                cand, _ = _fit(cand, l_reg, f_reg, n_reg)
                chi2_cand = _chisq_tot(cand, l_reg, f_reg, n_reg)
                if chi2_cand < chi2_trial:
                    trial, chi2_trial = cand, chi2_cand
            if trial is None:
                break
            if verbose:
                print("Region %d: %d lines, chisq_tot %.1f -> %.1f"
                      % (ireg, len(trial) // 3, chi2_tot, chi2_trial))
            if (chi2_tot - chi2_trial < _DCHISQ_PER_LINE
                    or chi2_trial > chisq_factor * chi2_tot):
                break
            params, chi2_tot = trial, chi2_trial

        params = _prune(params, l_reg, f_reg, n_reg, "Region %d" % ireg)
        if verbose:
            print("Region %d: FINAL FIT %d lines, N=%s, chisq=%.3f"
                  % (ireg, len(params) // 3, np.round(params[0::3], 3),
                     _chisq(params, l_reg, f_reg, n_reg, ion_name, mode)))
        params_all = np.append(params_all, params)
        region_of  = np.append(region_of, np.full(len(params) // 3, ireg))

    # ── Full spectrum: joint fit of all components (overlapping wings of
    # neighbouring regions), dropping any that became insignificant ─────────

    def _region_bounds(region_of):
        # each centre stays within its region, so region, EW and Chisq of the
        # line list refer to where the line is
        return np.array([regions_l[ireg] for ireg in region_of], dtype=float)

    perr_all = np.zeros_like(params_all)
    if len(params_all) > 0:
        params_all, perr_all = _fit(params_all, l, flux, noise, _region_bounds(region_of))
        chi2 = _chisq_tot(params_all, l, flux, noise)
        keep = np.ones(len(params_all) // 3, dtype=bool)
        for i in np.argsort(params_all[0::3]):
            trial = params_all.reshape(-1, 3)[keep & (np.arange(len(keep)) != i)].ravel()
            if _chisq_tot(trial, l, flux, noise) - chi2 < _DCHISQ_PER_LINE:
                keep[i] = False
        if not keep.all():
            if verbose:
                print("Full spectrum: removed %d insignificant line(s)" % np.sum(~keep))
            params_all = params_all.reshape(-1, 3)[keep].ravel()
            region_of  = region_of[keep]
            params_all, perr_all = _fit(params_all, l, flux, noise, _region_bounds(region_of))

    model_flux = _tau_to_flux(model_tau_fast(ion_name, params_all, l, mode, lookup=_lookup))
    for ip in range(len(params_all) // 3):
        ireg = region_of[ip]
        sl = slice(regions_i[ireg, 0], regions_i[ireg, 1])
        dx = (flux[sl] - model_flux[sl]) / noise[sl]
        tau_line = model_tau_fast(ion_name, params_all[3*ip:3*ip+3], l[sl], mode, lookup=_lookup)
        line_list["region"] = np.append(line_list["region"], ireg)
        line_list["N"]      = np.append(line_list["N"],  params_all[3*ip])
        line_list["b"]      = np.append(line_list["b"],  params_all[3*ip+1])
        line_list["l"]      = np.append(line_list["l"],  params_all[3*ip+2])
        line_list["dN"]     = np.append(line_list["dN"], perr_all[3*ip])
        line_list["db"]     = np.append(line_list["db"], perr_all[3*ip+1])
        line_list["dl"]     = np.append(line_list["dl"], perr_all[3*ip+2])
        line_list["EW"]     = np.append(line_list["EW"], EquivalentWidth(_tau_to_flux(tau_line), l[sl]))
        line_list["Chisq"]  = np.append(line_list["Chisq"], np.sum(dx * dx) / max(np.count_nonzero(dx), 1))

    if verbose:
        print("Full spectrum: FINAL FIT %d lines in %d regions, chisq=%.3f. Line list [i,N,b,l]:"
              % (len(line_list["N"]), len(regions_l), _chisq(params_all, l, flux, noise, ion_name, mode)))
        for ip in range(len(line_list["N"])):
            print(ip, line_list["N"][ip], line_list["b"][ip], line_list["l"][ip])

    return line_list


# Minimum decrease of the total χ² for an additional component (3 parameters)
# to be kept: the Akaike information criterion, 2 per free parameter.
_DCHISQ_PER_LINE = 6.0


def _lsq_fit_lines(lookup, p, l, flux, noise, logN_bounds, b_bounds, l_bounds=None):
    """
    Least-squares fit of all components in p to the flux.

    The line centres are fitted as velocity offsets (km/s) from the middle of
    the wavelength grid, so all parameters are of similar scale, and the
    Jacobian is computed per component from the lookup table (analytic in
    logN, central differences in b and v).

    Args:
        lookup (TauLookupTable): Table for the line and profile shape.
        p (array-like):          Flat start parameters [logN, b, λ₀, …].
        l (numpy array):         Wavelengths (Å).
        flux (numpy array):      Normalised flux.
        noise (numpy array):     1-sigma noise (> 0).
        logN_bounds (list):      [min, max] log₁₀ column density.
        b_bounds (list):         [min, max] Doppler parameter (km/s).
        l_bounds (array-like):   Optional (n_lines, 2) [min, max] wavelengths
                                 of each centre (Å); default: the range of l.

    Returns:
        p (numpy array):    Best-fit [logN, b, λ₀, …].
        perr (numpy array): 1-sigma errors from the Jacobian at the solution;
                            inf for parameters the data do not constrain.
    """
    c_kms = lookup.c_kms
    p = np.asarray(p, dtype=float).reshape(-1, 3)
    n = len(p)
    if n == 0:
        return np.array([]), np.array([])
    l_ref = 0.5 * (l[0] + l[-1])
    if l_bounds is None:
        l_bounds = np.tile([l[0], l[-1]], (n, 1))
    v_bounds = (np.asarray(l_bounds, dtype=float) / l_ref - 1.0) * c_kms
    lo = np.column_stack([np.full(n, logN_bounds[0]), np.full(n, b_bounds[0]), v_bounds[:, 0]]).ravel()
    hi = np.column_stack([np.full(n, logN_bounds[1]), np.full(n, b_bounds[1]), v_bounds[:, 1]]).ravel()
    x0 = np.column_stack([p[:, 0], p[:, 1], (p[:, 2] / l_ref - 1.0) * c_kms]).ravel()
    pad = 1e-6 * (hi - lo)
    x0 = np.clip(x0, lo + pad, hi - pad)
    h_v = lookup.dv

    def _tau_one(logN, b, v):
        return lookup.compute_tau(logN, b, l_ref * (1.0 + v / c_kms), l)

    def _taus(x):
        return np.array([_tau_one(*x[3*i:3*i+3]) for i in range(n)])

    def _resid(x):
        tau = np.sum(_taus(x), axis=0)
        return (flux - np.exp(-np.clip(tau, -50, 50))) / noise

    def _jac(x):
        taus = _taus(x)
        F_sig = np.exp(-np.clip(np.sum(taus, axis=0), -50, 50)) / noise
        J = np.empty((len(l), 3 * n))
        for i in range(n):
            logN, b, v = x[3*i:3*i+3]
            # central difference in b, kept inside the table's b range
            b_lo = max(0.99 * b, lookup.b_min)
            b_hi = min(1.01 * b, lookup.b_max)
            J[:, 3*i]   = F_sig * np.log(10.0) * taus[i]
            J[:, 3*i+1] = F_sig * (_tau_one(logN, b_hi, v) - _tau_one(logN, b_lo, v)) / (b_hi - b_lo)
            J[:, 3*i+2] = F_sig * (_tau_one(logN, b, v + h_v) - _tau_one(logN, b, v - h_v)) / (2 * h_v)
        return J

    soln = least_squares(_resid, x0, jac=_jac, bounds=(lo, hi), method="trf",
                         x_scale="jac")
    x = soln.x
    perr = _lsq_errors(soln.jac)
    p_fit = x.reshape(-1, 3).copy()
    p_fit[:, 2] = l_ref * (1.0 + p_fit[:, 2] / c_kms)
    perr = perr.reshape(-1, 3)
    perr[:, 2] *= l_ref / c_kms
    return p_fit.ravel(), perr.ravel()


def _lsq_errors(J):
    """
    1-sigma parameter errors sqrt(diag((JᵀJ)⁻¹)) from the Jacobian of the
    noise-weighted residuals, via its SVD.  Directions with (numerically)
    vanishing singular values are unconstrained, and every parameter with a
    share in them gets an infinite error.
    """
    try:
        _, s, Vt = np.linalg.svd(J, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.full(J.shape[1], np.nan)
    if len(s) == 0 or s[0] == 0:
        return np.full(J.shape[1], np.inf)
    ok = s > np.finfo(float).eps * max(J.shape) * s[0]
    var = np.sum((Vt[ok] / s[ok, None]) ** 2, axis=0)
    if not ok.all():
        var[np.any(np.abs(Vt[~ok]) > 1e-6, axis=0)] = np.inf
    return np.sqrt(var)


def find_regions(
    wavelengths,
    fluxes,
    noise,
    min_region_width=2,
    N_sigma=10.0,
    extend=False,
    buffer=2,
    det_flag=False,
    verbose=False,
):
    """
    Identify wavelength regions containing significant absorption features.

    Computes equivalent-width-weighted detection statistics by convolving the
    flux decrement and noise with Gaussians of varying width (std = 2–10 pixels),
    then selects the peak detection ratio at each pixel.  Contiguous pixels with
    a detection ratio above zero are grouped into candidate regions, which are
    kept only if their combined significance exceeds N_sigma.  Nearby regions
    separated by fewer than 5 pixels are merged, and each accepted region is
    padded by buffer pixels on each side.

    Args:
        wavelengths (numpy array):    Wavelength array (Å).
        fluxes (numpy array):         Normalised flux array.
        noise (numpy array):          1-sigma noise at each pixel.
        min_region_width (int):       Minimum width of a candidate region (pixels).
                                      Regions narrower than this are discarded.
        N_sigma (float):              Significance threshold in standard deviations.
                                      Only regions with combined detection significance
                                      above this value are retained.
        extend (bool):                If True, expand each region boundary outward
                                      until the flux returns to the continuum level.
                                      Default is False.
        buffer (int):                 Number of pixels to add to each edge of every
                                      accepted region.
        det_flag (bool):              If True, return immediately with empty lists
                                      (used to suppress detection during testing).
        verbose (bool):               Print a summary of found regions to stdout.

    Returns:
        regions_l (numpy array): Shape (n_regions, 2); start and end wavelengths
                                 of each accepted absorption region.
        regions_i (numpy array): Shape (n_regions, 2); start and end pixel indices
                                 of each accepted absorption region.
    """
    num_pixels = len(wavelengths)
    min_pix = 1
    max_pix = num_pixels - 1

    flux_ews  = [0.0] * num_pixels
    noise_ews = [0.0] * num_pixels
    det_ratio = [-float("inf")] * num_pixels

    for i in range(min_pix, max_pix):
        flux_dec = 1.0 - fluxes[i]
        if flux_dec < noise[i]:
            flux_dec = 0.0
        flux_ews[i]  = 0.5 * abs(wavelengths[i - 1] - wavelengths[i + 1]) * flux_dec
        noise_ews[i] = 0.5 * abs(wavelengths[i - 1] - wavelengths[i + 1]) * noise[i]

    flux_ews[0]  = 0.0
    noise_ews[0] = 0.0

    std_min = 2
    std_max = 11
    xarr = np.array([p - (num_pixels - 1) / 2.0 for p in range(num_pixels)])

    for std in range(std_min, std_max):
        gaussian   = np.exp(-0.5 * (xarr / std) ** 2)
        flux_func  = np.convolve(flux_ews,  gaussian,              "same")
        noise_func = np.convolve(np.square(noise_ews), np.square(gaussian), "same")

        for i in range(min_pix, max_pix):
            noise_func[i] = 1.0 / np.sqrt(noise_func[i])
            if flux_func[i] * noise_func[i] > det_ratio[i]:
                det_ratio[i] = flux_func[i] * noise_func[i]

    if det_flag:
        return [], []

    start = 0
    region_endpoints = []
    for i in range(num_pixels):
        if start == 0 and det_ratio[i] > 0 and fluxes[i] < 1.0:
            start = i
        elif start != 0 and (det_ratio[i] < 0 or fluxes[i] > 1.0):
            end = i
            region_endpoints.append([start, end])
            start = 0

    significant_region_endpoints = []
    for reg in region_endpoints:
        det_ratio    = np.array(det_ratio)
        significance = np.sqrt(np.sum(det_ratio[reg[0] : reg[1]] ** 2))
        if significance == np.inf:
            significance = 0
        if significance > N_sigma:
            significant_region_endpoints.append(reg)

    if extend:
        regions_expanded = []
        for reg in significant_region_endpoints:
            start = reg[0]
            i = start
            while i > 0 and fluxes[i] < 1.0:
                i -= 1
            start_new = i
            end = reg[1]
            j = end
            while j < (len(fluxes) - 1) and fluxes[j] < 1.0:
                j += 1
            end_new = j
            regions_expanded.append([start_new, end_new])
    else:
        regions_expanded = significant_region_endpoints

    regions_l = []
    regions_i = []
    for i in range(len(regions_expanded) - 1):
        if len(regions_expanded) == i:
            break
        start = regions_expanded[i][0]
        end   = regions_expanded[i][1]
        if len(regions_expanded) == i + 1:
            break
        if (regions_expanded[i + 1][0] - end) < 5:
            regions_expanded[i][1] = regions_expanded[i + 1][1]
            regions_expanded = np.delete(regions_expanded, (i + 1), axis=0)

    for i in range(len(regions_expanded)):
        start = regions_expanded[i][0]
        end   = regions_expanded[i][1]
        if i > 0:
            start_min = regions_expanded[i-1][1] - buffer
        for j in range(start, end):
            flux_dec = 1.0 - fluxes[j]
            if start >= buffer:
                start -= buffer
            if i > 0 and start < start_min:
                start = start_min
            if end < len(wavelengths) - buffer:
                end += buffer
            regions_expanded[i][0] = start
            regions_expanded[i][1] = end
            regions_l.append([wavelengths[start], wavelengths[end]])
            regions_i.append([start, end])
            break

    while len(regions_l) > 100000:
        for i in range(len(regions_l)-1):
            if regions_i[i][1] - buffer >= regions_i[i+1][0]:
                print('removing overlapping reg: %d %d' % (regions_i[i][1]-buffer, regions_i[i+1][0]))
                regions_i[i][1] = regions_i[i+1][1]
                regions_l[i][1] = regions_i[i+1][1]
                regions_l.pop(i+1)
                regions_i.pop(i+1)
                break

    if verbose:
        print('Found %d detection regions:' % len(regions_l))
        for i in range(len(regions_l)):
            print(i, regions_l[i][0],'-',regions_l[i][1],'  pixels:', regions_i[i][0],'-',regions_i[i][1])
    return np.array(regions_l), np.array(regions_i)


def find_saturated_regions(
    wavelengths, fluxes, noise,
    min_region_width=2, N_sigma=10.0, extend=False, verbose=False
):
    """
    Identify pixels that are close to the peak detection ratio (i.e. saturated).

    Unlike find_regions(), which selects pixels above a significance floor,
    this routine selects pixels whose detection ratio is within N_sigma of the
    global maximum.  It is used to locate deeply saturated absorption cores
    so that their bounding (unsaturated) wings can be used to constrain the fit.

    For each detected saturated sub-region the function also returns two small
    windows flanking that region on the left and right (the 'bounding regions'),
    which contain the unsaturated pixels used by fit_profiles_sat() to anchor
    the column density and Doppler parameter of the saturated component.

    Args:
        wavelengths (numpy array):    Wavelength array (Å).
        fluxes (numpy array):         Normalised flux array.
        noise (numpy array):          1-sigma noise array.
        min_region_width (int):       Minimum width of a saturated sub-region in
                                      pixels; narrower candidates are discarded.
        N_sigma (float):              A pixel is classed as saturated when its
                                      detection ratio is within N_sigma of the
                                      global maximum detection ratio.
        extend (bool):                If True, extend each detected region until
                                      the flux returns to the continuum level.
        verbose (bool):               Print a summary of found regions to stdout.

    Returns:
        regions_l (numpy array):         Shape (n_sat, 2); start and end wavelengths
                                         of each saturated sub-region.
        regions_i (numpy array):         Shape (n_sat, 2); start and end pixel indices
                                         of each saturated sub-region.
        bounding_regions_i (numpy array): Shape (n_sat, 2, 2); for each saturated
                                          region, the pixel ranges of the left and
                                          right bounding (unsaturated) windows,
                                          as [[left_start, left_end],
                                              [right_start, right_end]].
    """
    num_pixels = len(wavelengths)
    min_pix = 1
    max_pix = num_pixels - 1

    flux_ews  = [0.0] * num_pixels
    noise_ews = [0.0] * num_pixels
    det_ratio = [-float("inf")] * num_pixels

    for i in range(min_pix, max_pix):
        flux_dec = 1.0 - fluxes[i]
        if flux_dec < noise[i]:
            flux_dec = 0.0
        flux_ews[i]  = 0.5 * abs(wavelengths[i - 1] - wavelengths[i + 1]) * flux_dec
        noise_ews[i] = 0.5 * abs(wavelengths[i - 1] - wavelengths[i + 1]) * noise[i]

    flux_ews[0]  = 0.0
    noise_ews[0] = 0.0

    std_min = 2
    std_max = 11
    xarr    = np.array([p - (num_pixels - 1) / 2.0 for p in range(num_pixels)])

    for std in range(std_min, std_max):
        gaussian   = np.exp(-0.5 * (xarr / std) ** 2)
        flux_func  = np.convolve(flux_ews,  gaussian,              "same")
        noise_func = np.convolve(np.square(noise_ews), np.square(gaussian), "same")

        for i in range(min_pix, max_pix):
            noise_func[i] = 1.0 / np.sqrt(noise_func[i])
            if flux_func[i] * noise_func[i] > det_ratio[i]:
                det_ratio[i] = flux_func[i] * noise_func[i]

    start = 0
    region_endpoints = []
    for i in range(num_pixels):
        if start == 0 and np.abs(det_ratio[i]-np.max(det_ratio)) < N_sigma:
            start = i
        elif start != 0 and np.abs(det_ratio[i]-np.max(det_ratio)) > N_sigma:
            if (i - start) > min_region_width:
                end = i
                region_endpoints.append([start, end])
            start = 0

    if extend:
        regions_expanded = []
        for reg in region_endpoints:
            start = reg[0]
            i = start
            while i > 0 and fluxes[i] < 1.0:
                i -= 1
            start_new = i
            end = reg[1]
            j = end
            while j < (len(fluxes) - 1) and fluxes[j] < 1.0:
                j += 1
            end_new = j
            regions_expanded.append([start_new, end_new])
    else:
        regions_expanded = region_endpoints

    regions_l         = []
    regions_i         = []
    bounding_regions_i = []
    buffer = 3
    for i in range(len(regions_expanded)):
        start    = regions_expanded[i][0]
        end      = regions_expanded[i][1]
        end_init = end
        if i < (len(regions_expanded) - 1) and np.abs(end - regions_expanded[i + 1][0]) < 2:
            end = regions_expanded[i + 1][1]
        for j in range(start, end):
            flux_dec = 1.0 - fluxes[j]
            if flux_dec > abs(noise[j]) * N_sigma:
                if start >= buffer:
                    start -= buffer
                if end < len(wavelengths) - buffer:
                    end += buffer
                regions_l.append([wavelengths[start], wavelengths[end]])
                regions_i.append([start, end])

                start1 = 0          if (start-18) < 0          else (start-18)
                start2 = start      if (start-8)  < 0          else (start-8)
                end1   = end        if (end+8)  > len(fluxes)  else (end+8)
                end2   = int(len(fluxes)) if (end+18) > len(fluxes) else (end+18)

                bounding_regions_i.append([[start1, start2], [end1, end2]])
                break

    return np.array(regions_l), np.array(regions_i), np.array(bounding_regions_i)


def model_tau(ion_name, p, l, mode="Voigt"):
    """
    Compute the total optical depth spectrum for a set of absorption lines.

    Iterates over all lines encoded in p and sums their individual optical
    depth profiles using line_profile().  This is the exact (Faddeeva-function)
    implementation; use model_tau_fast() from tau_lookup for speed-critical
    contexts such as the fitting inner loop.

    Args:
        ion_name (str):      Line identifier, e.g. 'HI1215', as listed in
                             absorption_spectra.lines.
        p (array-like):      Flat parameter array [logN₁, b₁, λ₁, logN₂, b₂, λ₂, …]
                             where logN is log₁₀ column density (cm⁻²), b is the
                             Doppler parameter (km/s), and λ is the central
                             (observed) wavelength (Å).
        l (array-like):      Wavelength grid over which to compute the spectrum (Å).
        mode (str):          Profile shape: 'Voigt' (default), 'Gaussian', or
                             'Lorentzian'.

    Returns:
        total_tau (numpy array): Summed optical depth at each wavelength in l.
                                 Shape (len(l),), all values ≥ 0.
    """
    p = np.array(p)
    total_tau = np.zeros(len(l), dtype=float)
    line = lines[ion_name]
    if len(p) == 0:
        return total_tau
    l = np.asarray(l, dtype=float)
    l_rest = float(UnitScalar(line["l"], "Angstrom"))
    for ip in range(int(len(p) / 3)):
        # Evaluate in the rest frame: the cross-section scales with the rest
        # wavelength, not the observed λ, while l * l_rest / λ preserves the
        # velocity offsets (as in tau_lookup).
        _, tau = line_profile(
            line, 10 ** p[ip * 3], b=p[ip * 3 + 1], l0=l_rest,
            l=l * (l_rest / p[ip * 3 + 2]), mode=mode
        )
        total_tau += tau
    return total_tau


def _model_tau_periodic(ion_name, p, l, mode="Voigt", periodic=False):
    """
    model_tau(), plus (if periodic) the images of every line shifted by one box
    width either way, for a spectrum whose uniform wavelength grid l spans a
    whole periodic box.
    """
    tau = model_tau(ion_name, p, l, mode)
    p = np.asarray(p, dtype=float).reshape(-1, 3)
    if periodic and len(p) > 0:
        l = np.asarray(l, dtype=float)
        l_width = len(l) * (l[1] - l[0])
        for shift in (-l_width, l_width):
            images = p.copy()
            images[:, 2] += shift
            tau += model_tau(ion_name, images.ravel(), l, mode)
    return tau


def EquivalentWidth(fluxes, waves):
    """
    Compute the equivalent width of an absorption feature.

    Integrates (1 − flux) over the wavelength array using the trapezoidal
    rule with exact edge handling for the first and last pixels.

    Args:
        fluxes (numpy array): Normalised flux array for the line/region.
        waves (numpy array):  Wavelength array (Å), same length as fluxes.

    Returns:
        EW (float): Equivalent width in the same units as waves (Å).
    """
    fluxes = np.asarray(fluxes, dtype=float)
    waves  = np.asarray(waves,  dtype=float)
    dwave          = np.empty_like(waves)
    dwave[1:-1]    = 0.5 * np.abs(waves[2:] - waves[:-2])
    dwave[0]       = np.abs(waves[1]  - waves[0])
    dwave[-1]      = np.abs(waves[-1] - waves[-2])
    return float(np.sum((1.0 - fluxes) * dwave))


def _h5_filename(spec_name):
    """Spectrum file name: '.h5' is appended unless already present."""
    spec_name = str(spec_name)
    return spec_name if spec_name.endswith('.h5') else spec_name + '.h5'


def write_spectrum(
    spec_name, line, LOS_pos, lambda_rest, redshift,
    vels, fluxes, taus, noise, col_dens, phys_dens, temps, mets, vpec,
    overwrite=True,
):
    """
    Write a pygad absorption-line spectrum to an HDF5 file.

    Creates a new HDF5 file (or overwrites an existing one if overwrite=True)
    and stores the spectrum arrays together with physical metadata about the
    line of sight.  The file format is compatible with write_line_list() and
    fit_profiles().

    Args:
        spec_name (str):         Output filename; '.h5' is appended unless it
                                 already ends in '.h5'.
        line (str):              Ion name, e.g. 'HI1215', stored as an attribute
                                 of the lambda_rest dataset.
        LOS_pos (list/array):    (x, y, z) position of the line of sight in
                                 simulation units.  If only 2 values are given
                                 they are assumed to be (x, y) and z is set to -1.
        lambda_rest (float):     Rest wavelength of the ion (Å).
        redshift (float):        Redshift of the simulation snapshot.
        vels (array-like):       LOS velocities of each pixel (km/s if given
                                 without units).
        fluxes (array-like):     Normalised fluxes of each pixel, including any
                                 noise and instrumental smoothing applied.
        taus (array-like):       Optical depths of each pixel.
        noise (array-like):      1-sigma noise at each pixel.
        col_dens (array-like):   Ion column density at each pixel (cm⁻²).
        phys_dens (array-like):  Optical-depth-weighted physical gas density at
                                 each pixel (cm⁻³).
        temps (array-like):      Optical-depth-weighted gas temperature at each
                                 pixel (K).
        mets (array-like):       Optical-depth-weighted metallicity (mass fraction)
                                 at each pixel; stored as log₁₀(Z).
        vpec (array-like):       LOS peculiar velocity at each pixel (km/s).
        overwrite (bool):        If False and the file already exists, the write
                                 is skipped and a warning is printed.  Default True.
    """
    import os
    import h5py

    filename = _h5_filename(spec_name)
    if os.path.isfile(filename) and not overwrite:
        if environment.verbose >= environment.VERBOSE_TACITURN:
            print(
                "WARNING: write_spectrum() failed: File %s exists, and overwrite set to False"
                % filename
            )
        return

    vels = np.asarray(UnitQty(vels, "km/s").in_units_of("km/s"), dtype=float)
    lambda_rest = float(UnitScalar(lambda_rest, "Angstrom"))
    waves = lambda_rest * (redshift + 1.0) * (1.0 + vels / float(c.in_units_of("km/s")))
    mets  = np.log10(np.where(mets < 1.0e-10, 1.0e-10, mets))
    if len(LOS_pos) == 2:
        LOS_pos = np.append(np.array(LOS_pos), -1.0)

    with h5py.File(filename, "w") as hf:
        lam0 = hf.create_dataset("lambda_rest", data=lambda_rest)
        lam0.attrs["ion_name"] = line
        hf.create_dataset("LOS_pos",     data=np.array(LOS_pos))
        hf.create_dataset("redshift",    data=redshift)
        hf.create_dataset("velocity",    data=np.array(vels))
        hf.create_dataset("wavelength",  data=np.array(waves))
        hf.create_dataset("flux",        data=np.array(fluxes))
        hf.create_dataset("tau",         data=np.array(taus))
        hf.create_dataset("noise",       data=np.array(noise))
        hf.create_dataset("col_density", data=np.array(col_dens))
        hf.create_dataset("phys_density",data=np.array(phys_dens))
        hf.create_dataset("temperature", data=np.array(temps))
        hf.create_dataset("metallicity", data=np.array(mets))
        hf.create_dataset("vpec",        data=np.array(vpec))


def write_line_list(spec_name, line_list, regions_l, regions_i, mode="Voigt",
                    periodic=False):
    """
    Append Voigt-profile fit results to an existing pygad spectrum HDF5 file.

    Reads the ion name and wavelength grid from the file, reconstructs the
    combined model flux from line_list, then writes all fit parameters into a
    new 'line_list' group.  Any pre-existing 'line_list' or 'lines' group is
    replaced.

    Args:
        spec_name (str):       Path to the HDF5 spectrum file produced by
                               write_spectrum(), with or without the '.h5'.  Must contain 'lambda_rest'
                               (with 'ion_name' attribute) and 'wavelength'.
        line_list (dict):      Best-fit line parameters as returned by
                               fit_profiles_sat(), with keys:
                                 'region', 'N', 'dN', 'b', 'db',
                                 'l', 'dl', 'EW', 'Chisq'.
        regions_l (array-like): Shape (n_regions, 2); start and end wavelengths
                                of each absorption region (Å).
        regions_i (array-like): Shape (n_regions, 2); start and end pixel indices
                                of each absorption region.
        mode (str):            Profile shape the lines were fitted with, used
                               for the model flux; stored as an attribute.
        periodic (bool):       The spectrum spans a periodic box: lines near
                               one edge also absorb at the other.
    """
    import h5py

    spec_name = _h5_filename(spec_name)
    with h5py.File(spec_name, "r") as hf:
        line  = hf["lambda_rest"].attrs["ion_name"]
        waves = np.array(hf["wavelength"])

    tau_model = np.zeros(len(waves))
    for i in range(len(line_list["N"])):
        p = np.array([line_list["N"][i], line_list["b"][i], line_list["l"][i]])
        tau_model += _model_tau_periodic(line, p, waves, mode, periodic=periodic)
    model_flux = np.exp(-np.clip(tau_model, -30, 30))

    regions_l0 = [x[0] for x in regions_l]
    regions_l1 = [x[1] for x in regions_l]
    regions_i0 = [x[0] for x in regions_i]
    regions_i1 = [x[1] for x in regions_i]
    N     = line_list["N"]
    dN    = line_list["dN"]
    b     = line_list["b"]
    db    = line_list["db"]
    l     = line_list["l"]
    dl    = line_list["dl"]
    EW    = line_list["EW"]
    chisq = line_list["Chisq"]

    with h5py.File(spec_name, "a") as hf:
        if "line_list" in hf.keys():
            if environment.verbose >= environment.VERBOSE_TACITURN:
                print("Deleting and replacing line_list in %s" % spec_name)
            del hf["line_list"]
        elif "lines" in hf.keys():
            del hf["lines"]
        grp = hf.create_group("line_list")
        grp.attrs["mode"] = mode
        grp.create_dataset("region",           data=np.array(line_list["region"], dtype=int))
        grp.create_dataset("logN",             data=np.array(N))
        grp.create_dataset("dlogN",            data=np.array(dN))
        grp.create_dataset("b",                data=np.array(b))
        grp.create_dataset("db",               data=np.array(db))
        grp.create_dataset("l",                data=np.array(l))
        grp.create_dataset("dl",               data=np.array(dl))
        grp.create_dataset("EW",               data=np.array(EW))
        grp.create_dataset("chisq",            data=np.array(chisq))
        grp.create_dataset("model_flux",       data=np.array(model_flux))
        grp.create_dataset("region_lam_start", data=np.array(regions_l0))
        grp.create_dataset("region_lam_end",   data=np.array(regions_l1))
        grp.create_dataset("region_pix_start", data=np.array(regions_i0, dtype=int))
        grp.create_dataset("region_pix_end",   data=np.array(regions_i1, dtype=int))


def fit_profiles(
    line,
    spectrum_file=None,
    l=None,
    vel=None,
    flux=None,
    noise=None,
    gal_v_pos=None,
    vel_range=0,
    chisq_lim=2.0,
    chisq_factor=0.95,
    max_lines=15,
    N_sigma_constr=3.0,
    mode="Voigt",
    logN_bounds=[12, 19],
    b_bounds=[0, 100],
    write_lines=False,
    plot_fit=False,
):
    """
    Top-level entry point: fit Voigt profiles to an absorption-line spectrum.

    Reads the spectrum either from a pygad HDF5 file or from arrays supplied
    directly, constructs a Spectrum object, runs the full fitting pipeline via
    Spectrum.fit_profiles(), writes the results back to the HDF5 file, and
    displays a diagnostic plot.

    Args:
        line (str):               Line identifier as listed in
                                  analysis.absorption_spectra.lines, e.g. 'HI1215'.
        spectrum_file (str):      Path to a pygad HDF5 spectrum file (with or
                                  without the '.h5' extension).  When
                                  provided, wavelengths, flux, noise, velocity, and
                                  redshift are read from the file and any arrays
                                  supplied via l/flux/noise are ignored.
        l (array-like):           Wavelength array (Å).  Required when
                                  spectrum_file is None.  Without vel, the
                                  redshift is taken from l[0], i.e. the first
                                  pixel is assumed to be at zero velocity.
        vel (array-like):         Velocity array (km/s).  When given with l, the
                                  redshift is derived from l[0] and vel[0].
        flux (array-like):        Normalised flux array.  Required when
                                  spectrum_file is None.
        noise (array-like):       1-sigma noise array.  Required when
                                  spectrum_file is None.
        gal_v_pos (float):        LOS velocity of the target galaxy (km/s).
                                  When provided together with vel_range, only the
                                  surrounding window is fitted.
        vel_range (float):        Half-width of the velocity window around the
                                  galaxy to fit (km/s).  Set to 0 (default) to
                                  fit the entire spectrum.
        chisq_lim (float):        Not used any more; see fit_profiles_sat().
        chisq_factor (float):     A new line is accepted only when it reduces χ²
                                  by at least this factor (must be ≤ 1).
        max_lines (int):          Maximum Voigt components per absorption region.
        N_sigma_constr (float):   Not used any more; see fit_profiles_sat().
        mode (str):               Profile shape: 'Voigt' (default), 'Gaussian',
                                  or 'Lorentzian'.
        logN_bounds (list):       [min, max] log₁₀ column density (cm⁻²) allowed
                                  during fitting.
        b_bounds (list):          [min, max] Doppler parameter (km/s).  When
                                  b_bounds[0] == 0, the lower bound is set
                                  automatically to 25 % of the thermal width at
                                  T = 10⁴ K for the given ion.
        write_lines (bool):       Write lines to spectrum_file (which must exist).
        plot_fit (bool):          Plot spectrum with fitted lines.

    Returns:
        line_list (dict):  Best-fit line parameters (see fit_profiles_sat()),
                           with line centres on the input wavelength grid.
                           Optionally also written to spectrum_file via
                           write_line_list() and displayed via
                           Spectrum.plot_fit().
    """
    if spectrum_file is not None:
        spectrum_file = _h5_filename(spectrum_file)
        f = h5py.File(spectrum_file, 'r')
        if 'galaxy_properties/vlos' in f:
            gal_velocity_pos = float(np.array(f['galaxy_properties/vlos']))
        else:
            gal_velocity_pos = gal_v_pos
        if gal_velocity_pos is None or vel_range <= 0:
            print('Fitting full-box periodic spectrum from file', spectrum_file)
        else:
            print('Fitting spectrum from %s around galaxy at v=%g +/- %g km/s'
                  % (spectrum_file, gal_velocity_pos, vel_range))
        l        = np.array(f['wavelength'])
        flux     = np.array(f['flux'])
        noise    = np.array(f['noise'])
        #noise = np.where(noise < 0.03333, 0.03333, noise)
        vel      = np.array(f['velocity'])
        redshift = float(np.array(f['redshift']))
        f.close()
    elif line is not None and l is not None:
        if isinstance(line, str):
            line_struct = lines[line]
            lambda_rest = float(UnitScalar(line_struct["l"], 'Angstrom'))
            l = np.asarray(l, dtype=float)
            if vel is not None:
                # consistent with the wavelengths Spectrum rebuilds from vel
                vel = np.asarray(vel, dtype=float)
                c_kms = float(c.in_units_of('km/s'))
                redshift = float(l[0] / (lambda_rest * (1.0 + vel[0] / c_kms)) - 1.0)
            else:
                # assumes the first pixel is at zero velocity
                redshift = float(l[0] / lambda_rest - 1.0)
        else:
            raise ValueError('Could not find line %s in database.' % line)
        gal_velocity_pos = gal_v_pos
        print('Fitting profiles for %s spectrum at z=%g with %d pixels' % (line, redshift, len(l)))
    else:
        raise ValueError('Must provide either spectrum_file or [wave, flux, noise, vel]')

    if vel_range <= 0:
        gal_velocity_pos = None

    if b_bounds[0] == 0.:
        T       = UnitScalar(1.e4, "K")
        b_therm = thermal_b_param(line, T, "km/s")
        b_bounds = [0.25*float(b_therm), b_bounds[1]]

    spec = Spectrum(
        line, redshift, l, flux, noise, vel,
        gal_velocity_pos=gal_velocity_pos,
        logN_bounds=logN_bounds, b_bounds=b_bounds, mode=mode
    )
    spec.fit_profiles(
        vel_range=vel_range, chisq_lim=chisq_lim,
        chisq_factor=chisq_factor,
        max_lines=max_lines, N_sigma_constr=N_sigma_constr
    )
    if write_lines:
        write_line_list(spectrum_file, spec.line_list, spec.regions_l, spec.regions_i,
                        mode=mode, periodic=spec._periodic)
    if plot_fit:
        spec.plot_fit()

    return spec.line_list


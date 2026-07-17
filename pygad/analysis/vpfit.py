"""
Routines to fit Voigt profiles to absorption line spectra.

This module implements the full Voigt-profile fitting pipeline used in pygad.
The top-level entry point is fit_profiles(), which reads a spectrum (from a
pygad HDF5 file or from arrays supplied directly), identifies significant
absorption regions, fits multi-component Voigt/Gaussian/Lorentzian profiles
to each region, and writes the results back to the HDF5 file.

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

Last edited: Romeel Dave 14 July 2026
Contributors: Romeel Dave, Clara Lilje
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
from physics import wave_to_vel, vel_to_wave, tau_to_flux
from utils import read_h5_into_dict
from scipy import signal
import scipy
from scipy.optimize import minimize, NonlinearConstraint
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
        """
        # assign optional keyword arguments
        for key in kwargs:
            setattr(self, key, kwargs[key])
        # look up line to fit
        if isinstance(ion_name, str):
            line = lines[ion_name]
            lambda_rest = UnitScalar(line["l"], 'Angstrom')
        else:
            print('Could not find line %s in database.' % ion_name)
            exit
        # compute wavelengths or velocities from the other
        if vel is not None:
            wave = lambda_rest * (redshift + 1.0) * (1.0 + vel / float(c.in_units_of('km/s')))
        elif l is not None:
            vel = l / (lambda_rest * (redshift + 1.0)) * float(c.in_units_of('km/s')) - 1.0
            wave = l
        else:
            print('One of l or vel must be provided.')
            exit
        # load spectrum
        self.ion_name = ion_name
        self.lambda_rest = lambda_rest
        self.wavelengths = wave
        self.fluxes = flux
        self.noise = noise
        self.redshift = redshift
        self.velocities = vel
        self.continuum = np.ones(len(l))
        self.verbose = (environment.verbose >= environment.VERBOSE_QUIET)
        if self.verbose:
            print('N,b bounds:', self.logN_bounds, self.b_bounds)

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

    def buffer_with_continuum(self, waves, flux, nbuffer=50, snr_default=30.):
        """
        Pad a wavelength/flux pair with continuum-level pixels at each end.

        Extends the wavelength axis by extrapolating with the pixel spacing at
        each edge and sets the padded flux values to 1 (continuum).  The buffer
        helps the Voigt fitter identify the baseline and avoid boundary effects.

        Args:
            waves (numpy array):   Wavelength array (Å).
            flux (numpy array):    Normalised flux array.
            nbuffer (int):         Number of continuum pixels to add at each end.
            snr_default (float):   Signal-to-noise ratio used when self.snr is not
                                   set.  Currently reserved for future noise-padded
                                   buffer implementations.

        Returns:
            waves (numpy array):  Extended wavelength array of length
                                  len(waves) + 2*nbuffer.
            flux (numpy array):   Extended flux array of length
                                  len(flux) + 2*nbuffer, with flux = 1 at
                                  both padded ends.
        """
        if hasattr(self, 'snr'):
            snr = self.snr
        else:
            snr = snr_default
        dl = waves[1] - waves[0]
        l_start = np.arange(waves[0] - dl*nbuffer, waves[0], dl)
        l_end = np.arange(waves[-1]+dl, waves[-1] + dl*(nbuffer+1), dl)

        waves = np.concatenate((l_start, waves, l_end))
        new_noise = np.zeros(2*nbuffer)
        flux = np.concatenate((
            tau_to_flux(np.zeros(nbuffer)) + new_noise[:nbuffer],
            flux,
            tau_to_flux(np.zeros(nbuffer)) + new_noise[nbuffer:]
        ))

        return waves, flux

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

    def periodic_unwrap_wavelength(self):
        """
        Restore wavelengths to their original values after periodic_wrap().

        After periodic_wrap() the wavelength array is no longer aligned with
        the rolled flux/noise arrays.  This routine shifts the wavelengths so
        that index 0 corresponds to wrap_pixel, and wraps any values that
        overflow the right-hand edge of the simulation box back to the left.

        Returns:
            l (numpy array):  Corrected wavelength array in the same units as
                              self.wavelengths.
        """
        l = self.wavelengths
        l_box = l[-1] + (l[-1]-l[-2])
        l = l - l[0] + l[self.wrap_pixel]
        l = np.where(l > l_box, l - l_box, l)
        return l

    def prepare_spectrum(self, vel_range, do_continuum_buffer=False, nbuffer=10, snr_default=30):
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
            snr_default (float):         Signal-to-noise ratio used to construct
                                         noise_fit when self.snr is not set.
        """
        if self.gal_velocity_pos is not None:
            i_start, i_end, N = self.get_initial_window(vel_range)
            i_start, i_end, N = self.extend_to_continuum(i_start, i_end, N)
            if i_start < 0:
                i_start += len(self.wavelengths)
                i_end += len(self.wavelengths)
        else:
            self.orig_fluxes = self.fluxes
            self.orig_noise = self.noise
            self.fluxes, self.noise, self.wrap_pixel = self.periodic_wrap()
            i_start = 0
            i_end = len(self.wavelengths)
            N = i_end - i_start

        self.waves_fit  = self.wavelengths.take(range(i_start, i_end), mode='wrap')
        self.fluxes_fit = self.fluxes.take(range(i_start, i_end), mode='wrap')

        i_wrap = len(self.wavelengths) - i_start
        wave_boxsize = self.wavelengths[-1] - self.wavelengths[0]
        dl = self.wavelengths[1] - self.wavelengths[0]
        if i_wrap < N:
            self.waves_fit[i_wrap:] += wave_boxsize + dl

        if do_continuum_buffer is True:
            self.waves_fit, self.fluxes_fit = self.buffer_with_continuum(
                self.waves_fit, self.fluxes_fit, nbuffer=nbuffer
            )

        if hasattr(self, 'snr'):
            snr = self.snr
        else:
            snr = snr_default
        self.noise_fit = np.asarray([1./snr] * len(self.fluxes_fit))

    def fit_periodic_spectrum(self):
        """
        Fit Voigt profiles to a spectrum that spans the entire simulation volume.

        Periodically wraps the spectrum so that the highest-flux pixel is at
        index 0, delegates to pg.analysis.fit_profiles(), then unwraps the
        fitted line wavelengths back to their original positions.  Any lines
        outside the velocity range ±vel_range around self.gal_velocity_pos are
        removed from the final line list.

        Sets:
            self.line_list (dict): Best-fit line parameters with keys
                                   ['N', 'dN', 'b', 'db', 'l', 'dl', 'EW',
                                   'Chisq', 'region'].
        """
        wrap_flux, wrap_noise, wrap_start = self.periodic_wrap()
        self.line_list = pg.analysis.fit_profiles(
            self.ion_name, self.wavelengths, wrap_flux, wrap_noise,
            chisq_lim=2.0, max_lines=10,
            logN_bounds=self.logN_bounds, b_bounds=self.b_bounds, mode='Voigt'
        )
        self.line_list['l'] = pg.analysis.periodic_unwrap_wavelength(
            self.line_list['l'], self.wavelengths, wrap_start
        )
        self.line_list['v'] = wave_to_vel(
            self.line_list['l'], self.lambda_rest, self.redshift
        )
        outwith_vel_mask = ~(
            (self.line_list['v'] > self.gal_velocity_pos - vel_range) &
            (self.line_list['v'] < self.gal_velocity_pos + vel_range)
        )
        for k in self.line_list.keys():
            self.line_list[k] = np.delete(self.line_list[k], outwith_vel_mask)

    def get_tau_model(self):
        """
        Compute the total model optical depth from all fitted absorption lines.

        Sums contributions from every line in self.line_list over the full
        wavelength array self.wavelengths using model_tau().

        Sets:
            self.tau_model (numpy array): Optical depth at each wavelength pixel.
        """
        self.tau_model = np.zeros(len(self.wavelengths))
        for i in range(len(self.line_list["N"])):
            p = np.array([self.line_list["N"][i], self.line_list["b"][i], self.line_list["l"][i]])
            self.tau_model += model_tau(self.ion_name, p, self.wavelengths, 'Voigt')

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
                     snr_default=30., chisq_lim=2.0, chisq_unacceptable=25,
                     chisq_factor=0.95, N_sigma_constr=3.0, max_lines=12):
        """
        Prepare the spectrum and fit Voigt profiles to all absorption regions.

        Orchestrates the full fitting pipeline:
            1. prepare_spectrum()  — extract/wrap the spectral window,
                                     optionally pad with continuum, build noise.
            2. find_regions()      — identify pixels with significant absorption.
            3. fit_profiles_sat()  — fit multi-component profiles per region.
            4. periodic_unwrap_wavelength() — restore wavelengths if wrapped.

        Sets the following instance attributes after returning:
            self.line_list (dict): Best-fit line parameters.
            self.regions_l (numpy array): Wavelength boundaries of each region.
            self.regions_i (numpy array): Pixel boundaries of each region.

        Args:
            vel_range (float):           Half-width of the velocity window (km/s).
            do_continuum_buffer (bool):  Pad each end with nbuffer continuum pixels.
            nbuffer (int):               Number of continuum buffer pixels per end.
            snr_default (float):         Fallback SNR used to build the noise array.
            chisq_lim (float):           Reduced χ² acceptance threshold; no more
                                         lines are added once χ² falls below this.
            chisq_unacceptable (float):  Reduced χ² above which the fit for a region
                                         is flagged as unreliable.
            chisq_factor (float):        A new line is accepted only when it reduces
                                         χ² by at least this factor (must be ≤ 1).
            N_sigma_constr (float):      The model flux is constrained to exceed
                                         flux − N_sigma_constr × noise at every pixel.
            max_lines (int):             Maximum Voigt components allowed per region.
        """
        self.prepare_spectrum(vel_range, do_continuum_buffer=True, nbuffer=50, snr_default=30.)
        self.regions_l, self.regions_i = find_regions(
            self.waves_fit, self.fluxes_fit, self.noise_fit, verbose=self.verbose
        )
        self.line_list = fit_profiles_sat(
            self.ion_name, self.waves_fit, self.fluxes_fit, self.noise_fit,
            self.regions_l, self.regions_i,
            chisq_lim=chisq_lim, chisq_factor=chisq_factor,
            chisq_unacceptable=chisq_unacceptable, N_sigma_constr=N_sigma_constr,
            max_lines=max_lines,
            logN_bounds=self.logN_bounds,
            b_bounds=self.b_bounds, mode='Voigt', verbose=self.verbose
        )
        if self.wrap_pixel > 0:
            l = self.periodic_unwrap_wavelength()

    def plot_fit(self, ax=None):
        """
        Plot the data spectrum, individual fitted profiles, and the combined model.

        Draws on the supplied axes (or a new figure):
            • Data flux as a solid grey line.
            • Each individual fitted profile as a semi-transparent dashed line.
            • A short red tick at the top of the plot at each line's centre.
            • The total combined model flux as a dashed pink line.

        Args:
            ax (matplotlib.axes.Axes | None): Axes to draw into.  A new figure
                                              is created when None (default).
        """
        if ax is None:
            fig, ax = plt.subplots()

        x_val = self.wavelengths
        ax.plot(x_val, self.fluxes, label='data', c='tab:grey', lw=2, ls='-')

        self.get_fluxes_model()
        for i in range(len(self.line_list['N'])):
            p = np.array([self.line_list['N'][i], self.line_list['b'][i], self.line_list['l'][i]])
            _tau_model = model_tau(self.ion_name, p, self.wavelengths)
            ax.plot(x_val, tau_to_flux(_tau_model), alpha=0.5, lw=1, ls='--')
            l_cent = self.line_list['l'][i]
            ax.axvline(l_cent, ymin=0.95, ymax=0.98, color='r', linestyle='-', linewidth=1)

        ax.plot(x_val, self.fluxes_model, label='model', c='tab:pink', ls='--', lw=2)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(x_val[0], x_val[-1])
        ax.legend(loc='best', fontsize=8)
        plt.show()
        plt.close()


def fit_profiles_sat(
    ion_name,
    l,
    flux,
    noise,
    regions_l,
    regions_i,
    chisq_lim=2,
    chisq_factor=0.95,
    chisq_unacceptable=50.,
    N_sigma_constr=3.0,
    max_lines=12,
    mode="Voigt",
    logN_bounds=[8, 20],
    b_bounds=[1, 300],
    verbose=False
):
    """
    Fit Voigt/other profiles to a set of pre-identified absorption regions.

    Begins with a single component per region, adding lines iteratively until
    the reduced chi-squared falls below chisq_lim or max_lines is reached.
    After the per-region fit, the full-spectrum line list is refined by
    attempting to remove or combine redundant lines and by slightly reducing
    all column densities if that improves the global chi-squared.

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
        chisq_lim (float):          Reduced χ² threshold for a satisfactory fit.
                                    No more lines are added once χ² < chisq_lim.
        chisq_factor (float):       A newly added line is accepted only when it
                                    reduces χ² by at least this factor (≤ 1).
        chisq_unacceptable (float): Reduced χ² above which the fit is considered
                                    to have failed; triggers a reset in saturated
                                    region handling.
        N_sigma_constr (float):     The constrained minimiser requires the model
                                    flux to exceed flux − N_sigma_constr × noise
                                    at every pixel.
        max_lines (int):            Maximum Voigt components per absorption region.
                                    If reached the fit is declared done regardless
                                    of χ², which may yield a poor result.
        mode (str):                 Profile shape: 'Voigt', 'Gaussian', or
                                    'Lorentzian'.  See absorption_spectra.line_profile().
        logN_bounds (list):         [min, max] allowed log₁₀ column density (cm⁻²).
        b_bounds (list):            [min, max] allowed Doppler parameter (km/s).
        verbose (bool):             Print progress information to stdout.

    Returns:
        line_list (dict):  Best-fit parameters for all detected lines, with keys:
                             'region' (int)   — absorption region index.
                             'N'      (float) — log₁₀ column density (cm⁻²).
                             'dN'     (float) — 1-sigma uncertainty on N.
                             'b'      (float) — Doppler parameter (km/s).
                             'db'     (float) — 1-sigma uncertainty on b.
                             'l'      (float) — central wavelength (Å).
                             'dl'     (float) — 1-sigma uncertainty on l.
                             'EW'     (float) — equivalent width (Å).
                             'Chisq'  (float) — reduced χ² of the region fit.
    """
    from .tau_lookup import get_tau_lookup, model_tau_fast
    _lookup = get_tau_lookup(ion_name, mode)

    np.set_printoptions(formatter={'float': '{:.4f}'.format})

    if isinstance(ion_name, str):
        line = lines[ion_name]
    l0 = line["l"]
    if isinstance(l, np.ndarray) or l.units in [1, None]:
        l = UnitArr(l, "Angstrom")

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
        model_flux = _tau_to_flux(model_tau_fast(ion_name, p, l, mode))
        dx_array = (flux - model_flux) / noise
        return np.sum(dx_array * dx_array) / np.count_nonzero(dx_array)

    def _add_line(ion_name, p, bnd, l, flux, noise, l0, mode, i_line=None, grow_line=True):
        """
        Append a new Voigt component to the parameter and bounds arrays.

        Calls _grow_line() to estimate the best initial (logN, b) for the new
        component, or falls back to a simple width-based heuristic when
        grow_line=False.  The line is centred at the deepest point of the
        current residual flux unless i_line is explicitly given.

        Args:
            ion_name (str):      Line identifier.
            p (numpy array):     Current flat parameter array [logN, b, λ₀, …].
            bnd (numpy array):   Current bounds array, shape (3*n_lines, 2).
            l (numpy array):     Wavelength array of the fitting region (Å).
            flux (numpy array):  Observed flux in the fitting region.
            noise (numpy array): 1-sigma noise in the fitting region.
            l0 (float):          Rest wavelength of the line (Å).
            mode (str):          Profile type.
            i_line (int | None): Pixel index for the new line centre.  Uses the
                                 deepest residual pixel when None.
            grow_line (bool):    Call _grow_line() to find the best (logN, b)
                                 rather than using the simple heuristic.

        Returns:
            p (numpy array):   Updated parameter array with new line appended.
            bnd (numpy array): Updated bounds array with new line's bounds appended.
        """
        if len(p) == 0:
            resid = flux
        else:
            resid = 1.0 + flux - _tau_to_flux(model_tau(ion_name, p, l, mode))
        l_bounds = [l[1], l[-2]]

        if grow_line:
            n_guess, b_guess, l_guess = _grow_line(
                ion_name, l, flux, noise, resid, l0, mode, i_line=i_line
            )
        else:
            b_guess = (l_bounds[1] - l_bounds[0]) / float(l0) * 3.0e5 / 5.0
            b_guess = max(2 * b_bounds[0], 0.5 * min(b_bounds[1], b_guess))
            n_guess = 14.0 - resid[np.argmin(resid)]
            l_guess = l[np.argmin(resid)]

        p = np.append(p, n_guess)
        p = np.append(p, b_guess)
        p = np.append(p, l_guess)

        n_bounds = [n_guess-0.5, n_guess+0.5]
        b_bounds_new = [b_guess*0.5, b_guess*2]
        if len(bnd) == 0:
            bnd = np.array([n_bounds])
        else:
            bnd = np.append(bnd, np.array([n_bounds]), axis=0)
        bnd = np.append(bnd, np.array([b_bounds_new]), axis=0)
        bnd = np.append(bnd, np.array([l_bounds]), axis=0)
        return p, bnd

    def _grow_line(ion_name, l, flux, noise, resid, l0, mode,
                   i_line=None, floor_sigma=1.5, smooth_sigma=1., unsat_sigma=3.):
        """
        Find the largest (logN, b) component that does not violate a flux floor.

        Searches a 40 × 40 grid of (logN, b) values.  For each b the unit
        optical-depth profile is computed once via the lookup table and then
        linearly scaled for each logN, giving 40× fewer model evaluations than
        calling model_tau_fast for every (logN, b) pair.  The combination with
        the lowest reduced χ² near the line core is returned, provided the
        model flux everywhere exceeds smoothed_residual − floor_sigma × noise.

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

        N_range = np.linspace(logN_bounds[0], logN_bounds[1], 40)
        b_range = np.logspace(np.log10(b_bounds[0]), np.log10(b_bounds[1]), 40)

        best_chisq = 1.e20
        best_N, best_b = logN_bounds[0], b_bounds[0]

        for bpar in b_range:
            # Compute the profile shape once per b value via the lookup table;
            # subsequent N values need only a scalar multiply (no model call).
            tau_unit   = model_tau_fast(ion_name, [0.0, bpar, l_line], l, mode)

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

    def _grow_line_old(ion_name, l, flux, noise, resid, l0, mode,
                       i_line=None, floor_sigma=1.5, smooth_sigma=1., unsat_sigma=3.):
        """
        Legacy implementation of _grow_line (retained for reference).

        This version calls model_tau_fast for every (logN, b) pair in the
        search grid (up to 1 600 calls per invocation) and accumulates all
        allowed parameter sets before selecting the best χ².  It has been
        superseded by _grow_line(), which pre-computes the unit profile once
        per b value and achieves equivalent results with 40× fewer model calls.

        Args:  (same as _grow_line)
        Returns:  (same as _grow_line)
        """
        if smooth_sigma > 0.:
            smoothed = scipy.ndimage.gaussian_filter1d(resid, smooth_sigma)
        else:
            smoothed = resid
        if i_line is None:
            i_line = np.argmin(smoothed)

        l_line = l[i_line]
        if resid[i_line] < np.min(abs(noise)):
            i_lo = i_line
            while resid[i_lo] < unsat_sigma * noise[i_lo] and i_lo > 0: i_lo -= 1
            i_hi = i_line
            while resid[i_hi] < unsat_sigma * noise[i_hi] and i_hi < len(l)-1: i_hi += 1
            i_line = int(0.5 * (i_lo+i_hi))
            l_line = l[i_line]
            N_lim  = logN_bounds[1]
            b_lim = 2.*min(abs(l_line-l[i_lo]),abs(l[i_hi]-l_line)) * float(c.in_units_of('km/s')) / float(l0)
        else:
            N_lim = 15.0
            fdec_bottom = 1.-resid[i_line]
            i_lo = i_line
            while 1.-resid[i_lo] < 0.5 * fdec_bottom and i_lo > 0: i_lo -= 1
            i_hi = i_line
            while 1.-resid[i_hi] < 0.5 * fdec_bottom and i_hi < len(l)-1: i_hi += 1
            b_lim = 4.*min(abs(l_line-l[i_lo]),abs(l[i_hi]-l_line)) * float(c.in_units_of('km/s')) / float(l0)
        b_lim = min(max(b_lim, max(b_bounds[0],20)), b_bounds[1])

        smoothed = np.where(smoothed > 1., 1., smoothed)
        floor = smoothed - floor_sigma * noise

        N_range = np.linspace(start=logN_bounds[0], stop=N_lim, num=40)
        b_range = np.linspace(start=np.log10(b_bounds[0]), stop=np.log10(b_lim), num=40)
        b_range = 10**b_range
        N_min = logN_bounds[0]
        p_allowed = np.array([logN_bounds[0], b_bounds[0], l_line])
        chisq = [1.e20]
        for bpar in b_range:
            for Ncol in N_range:
                if Ncol < N_min:
                    continue
                p_trial = np.array([Ncol, bpar, l_line])
                model = _tau_to_flux(model_tau_fast(ion_name, p_trial, l, mode))
                diff = model-floor
                if np.any(diff<0):
                    continue
                else:
                    p_allowed = np.append(p_allowed, p_trial)
                    i_lo = np.argmin(model)
                    while (1.-model[i_lo]) > 0.5 * (1.-model[i_line]) and i_lo > 0: i_lo -= 1
                    i_hi = np.argmin(model)
                    while (1.-model[i_hi]) > 0.5 * (1.-model[i_line]) and i_hi < len(model)-2: i_hi += 1
                    dx_array = (resid[i_lo:i_hi+1] - model[i_lo:i_hi+1]) / noise[i_lo:i_hi+1]
                    i_min = np.argmin(diff)
                    chi2 = np.sum(dx_array * dx_array) / np.count_nonzero(dx_array)
                    chisq.append(chi2)
        i_p = np.argmin(np.array(chisq))
        return p_allowed[3*i_p], p_allowed[3*i_p+1], l_line

    def _maxiter(n, nmax):
        """
        Return the maximum minimiser iterations for a fit with n components.

        Grants more iterations for small line counts where accuracy matters
        most, tapering toward nmax to bound runtime for complex fits.

        Args:
            n (int):    Current number of Voigt components.
            nmax (int): Maximum components allowed in this region.

        Returns:
            maxiter (int): Maximum number of minimiser iterations to allow.
        """
        if n <= 5: return 100
        else: return max(50, 50+(nmax-n)*10)

    def _model_flux(p):
        """
        Return the model flux for the current region given parameter array p.

        Thin wrapper around model_tau_fast used as the callable for
        NonlinearConstraint in the constrained minimisation step.

        Args:
            p (numpy array): Flat parameter array [logN, b, λ₀, …].

        Returns:
            model_flux (numpy array): Normalised model flux at each pixel of l_reg.
        """
        return _tau_to_flux(model_tau_fast(ion_name, p, l_reg, mode, lookup=_lookup))

    def _constraint_jac(p):
        """
        Jacobian of _model_flux w.r.t. the parameter array p.

        Computed via forward finite differences with a step proportional to
        √(machine epsilon).  Supplying this to NonlinearConstraint avoids
        scipy's own internal finite-difference calls and reduces the total
        number of model evaluations per minimiser iteration.

        Args:
            p (numpy array): Flat parameter array [logN, b, λ₀, …], length 3*n.

        Returns:
            jac (numpy array): Jacobian of shape (n_pixels, 3*n).
        """
        eps   = np.sqrt(np.finfo(float).eps)
        n_p   = len(p)
        n_pix = len(l_reg)
        jac   = np.empty((n_pix, n_p))
        f0    = _tau_to_flux(model_tau_fast(ion_name, p, l_reg, mode, lookup=_lookup))
        for j in range(n_p):
            dp        = np.zeros(n_p)
            dp[j]     = eps * max(abs(p[j]), 1.0)
            f_plus    = _tau_to_flux(model_tau_fast(ion_name, p + dp, l_reg, mode, lookup=_lookup))
            jac[:, j] = (f_plus - f0) / dp[j]
        return jac

    # ── Main fitting loop (code unchanged below) ──────────────────────────

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

    sat_regions = False

    for ireg in range(len(regions_l)):

        params = []
        bounds = []
        n_lines = 0
        best_nlines = 1
        chisq_old = 1.0e20
        chisq_accept = abs(chisq_lim)
        l_reg = l[regions_i[ireg, 0] : regions_i[ireg, 1]]
        f_reg = flux[regions_i[ireg, 0] : regions_i[ireg, 1]]
        n_reg = noise[regions_i[ireg, 0] : regions_i[ireg, 1]]

        regions_l_sat, regions_i_sat, bounding_i = find_saturated_regions(
            l_reg, f_reg, n_reg, min_region_width=15, verbose=verbose
        )
        bounding_i = []

        params_reg = []
        bounds_reg = []
        best_nlines = 0
        if len(bounding_i) != 0:
            sat_regions = True
            if verbose:
                print('Region %d has %d saturated area(s) at pixels:'%(ireg, len(regions_i_sat)), regions_i_sat)
            for ireg_sat in range(len(regions_l_sat)):
                l_reg_sat = l_reg[regions_i_sat[ireg_sat, 0] : regions_i_sat[ireg_sat, 1]]
                f_reg_sat = f_reg[regions_i_sat[ireg_sat, 0] : regions_i_sat[ireg_sat, 1]]
                n_reg_sat = n_reg[regions_i_sat[ireg_sat, 0] : regions_i_sat[ireg_sat, 1]]

                l_reg_bound_left  = l_reg[bounding_i[ireg_sat, 0][0] : bounding_i[ireg_sat, 0][1]]
                l_reg_bound_right = l_reg[bounding_i[ireg_sat, 1][0] : bounding_i[ireg_sat, 1][1]]
                f_reg_bound_left  = f_reg[bounding_i[ireg_sat, 0][0] : bounding_i[ireg_sat, 0][1]]
                f_reg_bound_right = f_reg[bounding_i[ireg_sat, 1][0] : bounding_i[ireg_sat, 1][1]]
                n_reg_bound_left  = n_reg[bounding_i[ireg_sat, 0][0] : bounding_i[ireg_sat, 0][1]]
                n_reg_bound_right = n_reg[bounding_i[ireg_sat, 1][0] : bounding_i[ireg_sat, 1][1]]

                width        = (l_reg_bound_right[0]-l_reg_bound_left[-1])
                l_bounds_sat = [l_reg_bound_left[0], l_reg_bound_right[-1]]
                l_reg_bounds = np.concatenate((l_reg_bound_left, l_reg_bound_right))
                f_reg_bounds = np.concatenate((f_reg_bound_left, f_reg_bound_right))
                n_reg_bounds = np.concatenate((n_reg_bound_left, n_reg_bound_right))

                N_range = np.linspace(start=logN_bounds[0], stop=logN_bounds[1], num=20)
                b_range = np.linspace(start=np.log10(b_bounds[0]), stop=np.log10(b_bounds[1]), num=20)
                b_range = 10**b_range
                i_middle    = int((regions_i_sat[ireg_sat, 0] + regions_i_sat[ireg_sat, 1]) / 2)
                middle_guess = l_reg[i_middle]
                bounds = np.array(np.array([logN_bounds]))
                bounds = np.append(bounds, np.array([b_bounds]), axis=0)
                bounds = np.append(bounds, np.array([l_bounds_sat]), axis=0)

                chisq_best = 1.e20
                chisq_soln = 1.e20
                for Ncol in N_range:
                    for bpar in b_range:
                        params     = np.array([Ncol, bpar, middle_guess])
                        chisq_soln = _chisq(params, l_reg_bounds, f_reg_bounds, n_reg_bounds, ion_name, mode)
                        if chisq_soln < chisq_best:
                            chisq_best = chisq_soln
                            Nbest = Ncol
                            bbest = bpar
                params = np.array([Nbest, bbest, middle_guess])
                if verbose:
                    print("Found best-fit sat line (chisq=%g) with params"%(chisq_best), params)

                if len(params_reg) == 0:
                    params_reg = np.array(params)
                else:
                    params_reg = np.append(params_reg, np.array(params))

                if len(bounds_reg) == 0:
                    bounds_reg = np.array(bounds)
                else:
                    bounds_reg = np.append(bounds_reg, np.array(bounds))

            if verbose:
                print(
                    "Saturated line gives full region %d (%g-%g): chisq= %g with %d lines"
                    % (ireg, regions_l[ireg, 0], regions_l[ireg, 1],
                       chisq_soln, int(len(params_reg) / 3))
                )

            params = np.reshape(params_reg, (int(len(params_reg) / 3), 3))
            bounds = np.reshape(bounds_reg, (int(len(params_reg)), 2))
            if chisq_soln > 10000:
                params = []
                bounds = []
                sat_regions = False
                print('ChiSquare is too big, probably no saturated region.')

        else:
            params      = []
            bounds      = []
            n_lines     = 0
            best_nlines = 1
            chisq_old   = 1.0e20
            chisq_soln  = chisq_old
            chisq_accept = abs(chisq_lim)
            l_reg = l[regions_i[ireg, 0] : regions_i[ireg, 1]]
            f_reg = flux[regions_i[ireg, 0] : regions_i[ireg, 1]]
            n_reg = noise[regions_i[ireg, 0] : regions_i[ireg, 1]]

        if len(params) != 0:
            resid = (1.0 + f_reg - _tau_to_flux(model_tau(ion_name, params.flatten(), l_reg, mode)))
        else:
            distance = int(len(l_reg)/20)
            if distance < 1:
                distance = 1

        n_lines    = 0
        chisq_best = 1.e20
        chisq_old  = 1.e20
        delta_l    = l_reg[1]-l_reg[0]
        while n_lines < max_lines-1:
            params, bounds = _add_line(ion_name, params, bounds, l_reg, f_reg, n_reg, float(l0.split()[0]), mode)
            if params[-1] in params[2::3]:
                params[-1] = params[-1] + delta_l * (0.5*np.random.rand() - 1)
            n_lines    = int(len(params) / 3)
            resid      = 1.0 + f_reg - _tau_to_flux(model_tau(ion_name, params.flatten(), l_reg, mode))
            chisq_soln = _chisq(params, l_reg, f_reg, n_reg, ion_name, mode)
            if chisq_soln < chisq_best:
                best_nlines  = n_lines
                best_params  = params
                best_bounds  = bounds
                chisq_best   = chisq_soln
            if chisq_soln < chisq_accept:
                break
            if chisq_soln > chisq_factor * chisq_old:
                break
            if params[-3] <= logN_bounds[0] and params[-2] <= b_bounds[0] and n_lines > 4:
                params     = np.delete(params, [-3, -2, -1], axis=0)
                bounds     = np.delete(bounds, [-3, -2, -1], axis=0)
                n_lines    = int(len(params) / 3)
                chisq_soln = _chisq(params, l_reg, f_reg, n_reg, ion_name, mode)
                break
            chisq_old = chisq_soln
            if verbose:
                print(f'Region {ireg}: Added line {n_lines-1} with N=%.4f, b=%.4f, l=%.4f, chisq=%.3f' % (params[-3],params[-2],params[-1],chisq_soln))
        if verbose:
            print(f'Region {ireg}: Found {n_lines} lines in first guess, chisq=%.3f'%chisq_soln)

        best_nlines = n_lines
        best_params = params
        best_bounds = bounds
        chisq_best  = chisq_soln
        first_time  = True
        while n_lines < max_lines and chisq_soln > chisq_accept:
            if not first_time:
                params, bounds = _add_line(ion_name, params, bounds, l_reg, f_reg, n_reg, float(l0.split()[0]), mode)
                n_lines = int(len(params) / 3)
            chisq_fcn  = lambda *args: _chisq(*args)
            constraint = NonlinearConstraint(
                fun=_model_flux,
                lb=f_reg - N_sigma_constr * n_reg,
                ub=np.inf,
                jac=_constraint_jac,
            )
            soln = minimize(
                chisq_fcn,
                params,
                bounds=bounds,
                args=(l_reg, f_reg, n_reg, ion_name, mode),
                method="trust-constr",
                constraints=constraint,
                options={"maxiter": _maxiter(n_lines, max_lines), "gtol": 1e-8},
            )
            params     = soln.x
            chisq_soln = _chisq(params, l_reg, f_reg, n_reg, ion_name, mode)
            if verbose and not first_time:
                print("Region %d: Added new line %d (N=%g), after %d iters, chisq=%.3f"
                      % (ireg, n_lines, params[-3], soln.nit, chisq_soln))
            first_time = False
            if chisq_soln < chisq_factor * chisq_best:
                best_nlines = n_lines
                best_params = params
                best_bounds = bounds
                chisq_best  = chisq_soln
            elif n_lines <= 2 and chisq_soln < chisq_best:
                best_nlines = n_lines
                best_params = params
                best_bounds = bounds
                chisq_best  = chisq_soln
                continue
            else:
                params     = best_params
                bounds     = best_bounds
                n_lines    = int(len(params) / 3)
                chisq_soln = chisq_best
                break

        compute_errors = True
        delta_params   = [0.02, 0.05, 0.0001] * n_lines
        if compute_errors:
            params_jiggled = params + delta_params * (2 * np.random.rand(len(params)) - 1)
            chisq_fcn      = lambda *args: _chisq(*args)
            soln = minimize(
                chisq_fcn,
                params_jiggled,
                args=(l_reg, f_reg, n_reg, ion_name, mode),
                method="BFGS",
                options={"maxiter": 100},
            )
            cov = soln.hess_inv

        while n_lines > 1:
            for i in range(n_lines):
                trial_params = params.copy()
                i_del        = 3*i
                trial_params = np.delete(trial_params, [i_del, i_del+1, i_del+2], axis=0)
                chisq_trial  = _chisq(trial_params, l_reg, f_reg, n_reg, ion_name, mode)
                delta_chisq  = abs(chisq_trial-chisq_best)/chisq_trial
                if delta_chisq < 0.01 or chisq_trial < chisq_accept:
                    if verbose:
                        print("Region %d: Removed line %d (N=%g): chisq=%g, chisq_old=%g"
                              %(ireg, i_del, params[3*i], chisq_trial, chisq_best))
                    params     = trial_params.copy()
                    bounds     = np.delete(bounds, [i_del, i_del+1, i_del+2], axis=0)
                    chisq_best = chisq_trial
                    n_lines    = int(len(params)/3)
                    break
                else:
                    continue
            if i >= n_lines-2:
                break

        while n_lines > 1:
            for i in range(n_lines-1):
                trial_params = params.copy()
                ip  = 3*i
                ip1 = 3*(i+1)
                N_i  = 10**params[ip]
                N_i1 = 10**params[ip1]
                trial_params[ip]   = np.log10(N_i + N_i1)
                trial_params[ip+1] = (N_i * params[ip+1] + N_i1 * params[ip+4]) / (N_i + N_i1)
                trial_params[ip+2] = (N_i * params[ip+2] + N_i1 * params[ip+5]) / (N_i + N_i1)
                trial_params       = np.delete(trial_params, [ip+3, ip+4, ip+5], axis=0)
                chisq_trial  = _chisq(trial_params, l_reg, f_reg, n_reg, ion_name, mode)
                delta_chisq  = abs(chisq_trial-chisq_best)/chisq_trial
                if delta_chisq < 0.01 or chisq_trial < chisq_accept:
                    if verbose:
                        print("Region %d: Combining lines %d and %d (N=%g and %g): chisq=%g, chisq_old=%g"
                              %(ireg, i, i+1, N_i, N_i1, chisq_trial, chisq_best))
                    params     = trial_params.copy()
                    bounds     = np.delete(bounds, [ip+3, ip+4, ip+5], axis=0)
                    chisq_best = chisq_trial
                    n_lines    = int(len(params)/3)
                    break
                else:
                    continue
            if i >= n_lines-2:
                break

        chisq_soln = _chisq(params, l_reg, f_reg, n_reg, ion_name, mode)
        for ip in np.arange(n_lines):
            line_list["region"] = np.append(line_list["region"], ireg)
            line_list["N"]      = np.append(line_list["N"],  params[ip * 3])
            line_list["b"]      = np.append(line_list["b"],  params[ip * 3 + 1])
            line_list["l"]      = np.append(line_list["l"],  params[ip * 3 + 2])
            line_list["dN"]     = np.append(line_list["dN"], np.sqrt(cov[ip * 3, ip * 3]))
            line_list["db"]     = np.append(line_list["db"], np.sqrt(cov[ip * 3 + 1, ip * 3 + 1]))
            line_list["dl"]     = np.append(line_list["dl"], np.sqrt(cov[ip * 3 + 2, ip * 3 + 2]))
            tau_line = model_tau(
                ion_name,
                [params[ip * 3], params[ip * 3 + 1], params[ip * 3 + 2]],
                l_reg, mode,
            )
            line_list["EW"]    = np.append(line_list["EW"],    EquivalentWidth(_tau_to_flux(tau_line), l_reg))
            line_list["Chisq"] = np.append(line_list["Chisq"], chisq_soln)

        if verbose:
            print(f"Region {ireg}: FINAL FIT {n_lines} lines, N={params[0::3]}, chisq=%.3f"%(chisq_soln))
            if chisq_soln > chisq_accept:
                print("Region %d: WARNING large chisq=%.3f > %.3f; check fit"
                      % (ireg, chisq_soln, chisq_accept))

    # ── Full-spectrum refinement: remove, combine, and reduce lines ───────

    n_lines = len(line_list["N"])
    while n_lines > 1:
        params = []
        for ip in range(n_lines):
            params.append(line_list["N"][ip])
            params.append(line_list["b"][ip])
            params.append(line_list["l"][ip])
        params     = np.array(params)
        chisq_soln = _chisq(params, l, flux, noise, ion_name, mode)

        for i in range(n_lines):
            trial_params = params.copy()
            i_del        = 3*i
            trial_params = np.delete(trial_params, [i_del, i_del+1, i_del+2], axis=0)
            chisq_trial  = _chisq(trial_params, l, flux, noise, ion_name, mode)
            if chisq_trial < chisq_soln:
                line_list  = {k: np.delete(v, i) for k, v in line_list.items()}
                chisq_soln = chisq_trial
                n_lines    = len(line_list["N"])
                if verbose:
                    print("Full spectrum: Removed line %d (N=%g), %d left: chisq=%g, chisq_old=%g"
                          %(i_del, params[3*i], n_lines, chisq_trial, chisq_soln))
                break
            else:
                continue
        if i >= n_lines-2:
            break

    n_lines = len(line_list["N"])
    while n_lines > 1:
        params = []
        for ip in range(n_lines):
            params.append(line_list["N"][ip])
            params.append(line_list["b"][ip])
            params.append(line_list["l"][ip])
        params     = np.array(params)
        chisq_soln = _chisq(params, l, flux, noise, ion_name, mode)

        for i in range(n_lines-1):
            trial_params = params.copy()
            ip  = 3*i
            ip1 = 3*(i+1)
            N_i  = 10**params[ip]
            N_i1 = 10**params[ip1]
            trial_params[ip]   = np.log10(N_i + N_i1)
            trial_params[ip+1] = (N_i * params[ip+1] + N_i1 * params[ip+4]) / (N_i + N_i1)
            trial_params[ip+2] = (N_i * params[ip+2] + N_i1 * params[ip+5]) / (N_i + N_i1)
            trial_params       = np.delete(trial_params, [ip+3, ip+4, ip+5], axis=0)
            chisq_trial  = _chisq(trial_params, l, flux, noise, ion_name, mode)
            if chisq_trial < chisq_soln:
                if verbose:
                    print("Full spectrum: Combined lines %d and %d (N=%g and %g): chisq=%g, chisq_old=%g"
                          %(i, i+1, N_i, N_i1, chisq_trial, chisq_soln))
                line_list["N"][i] = trial_params[ip]
                line_list["b"][i] = trial_params[ip+1]
                line_list["l"][i] = trial_params[ip+2]
                line_list  = {k: np.delete(v, i+1) for k, v in line_list.items()}
                n_lines    = len(line_list["N"])
                chisq_soln = chisq_trial
                break
            else:
                continue
        if i >= n_lines-2:
            break

    chisq_trial = 0.
    f_reduce    = 0.999
    while chisq_trial < chisq_soln:
        chisq_trial  = chisq_soln
        trial_params = params.copy()
        trial_params[::3] *= f_reduce
        chisq_trial  = _chisq(trial_params, l, flux, noise, ion_name, mode)
        if chisq_trial < chisq_soln:
            if verbose:
                print(f"Multiplying all column densities by %d improves overall fit from chisq=%g to %g"
                      % (f_reduce, chisq_soln, chisq_trial))
            params     = trial_params.copy()
            chisq_soln = chisq_trial

    params  = []
    n_lines = len(line_list["N"])
    for ip in range(n_lines):
        params.append(line_list["N"][ip])
        params.append(line_list["b"][ip])
        params.append(line_list["l"][ip])
    params     = np.array(params)
    chisq_soln = _chisq(params, l, flux, noise, ion_name, mode)

    if verbose:
        print(f"Full spectrum: FINAL FIT {n_lines} lines in %d regions, chisq=%.3f. Line list [i,N,b,l]:"
              %(len(regions_l), chisq_soln))
        for ip in range(n_lines):
            print(ip, line_list["N"][ip], line_list["b"][ip], line_list["l"][ip])

    return line_list


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
    for ip in range(int(len(p) / 3)):
        _, tau = line_profile(
            line, 10 ** p[ip * 3], b=p[ip * 3 + 1], l0=p[ip * 3 + 2], l=l, mode=mode
        )
        total_tau += tau
    return total_tau


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
        spec_name (str):         Output filename (without extension; '.h5' is
                                 appended automatically).
        line (str):              Ion name, e.g. 'HI1215', stored as an attribute
                                 of the lambda_rest dataset.
        LOS_pos (list/array):    (x, y, z) position of the line of sight in
                                 simulation units.  If only 2 values are given
                                 they are assumed to be (x, y) and z is set to -1.
        lambda_rest (float):     Rest wavelength of the ion (Å).
        redshift (float):        Redshift of the simulation snapshot.
        vels (array-like):       LOS velocities of each pixel (km/s).
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

    if os.path.isfile(spec_name) and not overwrite:
        if environment.verbose >= environment.VERBOSE_TACITURN:
            print(
                "WARNING: write_spectrum() failed: File %s exists, and overwrite set to False"
                % spec_name
            )
        return

    waves = lambda_rest * (redshift + 1.0) * (1.0 + vels / c)
    mets  = np.log10(np.where(mets < 1.0e-10, 1.0e-10, mets))
    if len(LOS_pos) == 2:
        LOS_pos = np.append(np.array(LOS_pos), -1.0)

    with h5py.File("%s.h5" % spec_name, "w") as hf:
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


def write_line_list(spec_name, line_list, regions_l, regions_i):
    """
    Append Voigt-profile fit results to an existing pygad spectrum HDF5 file.

    Reads the ion name and wavelength grid from the file, reconstructs the
    combined model flux from line_list, then writes all fit parameters into a
    new 'line_list' group.  Any pre-existing 'line_list' or 'lines' group is
    replaced.

    Args:
        spec_name (str):       Path to the HDF5 spectrum file produced by
                               write_spectrum().  Must contain 'lambda_rest'
                               (with 'ion_name' attribute) and 'wavelength'.
        line_list (dict):      Best-fit line parameters as returned by
                               fit_profiles_sat(), with keys:
                                 'region', 'N', 'dN', 'b', 'db',
                                 'l', 'dl', 'EW', 'Chisq'.
        regions_l (array-like): Shape (n_regions, 2); start and end wavelengths
                                of each absorption region (Å).
        regions_i (array-like): Shape (n_regions, 2); start and end pixel indices
                                of each absorption region.
    """
    import h5py

    with h5py.File(spec_name, "r") as hf:
        line  = hf["lambda_rest"].attrs["ion_name"]
        waves = np.array(hf["wavelength"])

    tau_model = np.zeros(len(waves))
    for i in range(len(line_list["N"])):
        p = np.array([line_list["N"][i], line_list["b"][i], line_list["l"][i]])
        tau_model += model_tau(line, p, waves)
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
    chisq_unacceptable=50.0,
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
        spectrum_file (str):      Path to a pygad HDF5 spectrum file.  When
                                  provided, wavelengths, flux, noise, velocity, and
                                  redshift are read from the file and any arrays
                                  supplied via l/flux/noise are ignored.
        l (array-like):           Wavelength array (Å).  Required when
                                  spectrum_file is None.
        vel (array-like):         Velocity array (km/s).  Used to derive l when
                                  both l and vel are provided.
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
        chisq_lim (float):        Reduced χ² acceptance threshold; fitting stops
                                  adding lines once χ² < chisq_lim.
        chisq_unacceptable (float): Reduced χ² above which the regional fit is
                                  considered to have failed.
        chisq_factor (float):     A new line is accepted only when it reduces χ²
                                  by at least this factor (must be ≤ 1).
        max_lines (int):          Maximum Voigt components per absorption region.
        N_sigma_constr (float):   The constrained minimiser requires the model
                                  flux to exceed flux − N_sigma_constr × noise
                                  at every pixel.
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
        None.  Results are written to spectrum_file via write_line_list() and
        displayed via Spectrum.plot_fit().
    """
    if spectrum_file is not None:
        f = h5py.File(spectrum_file, 'r')
        try:
            gal_velocity_pos = np.array(f['galaxy_properties/vlos'])
        except:
            gal_velocity_pos = gal_v_pos
        if gal_velocity_pos is None or vel_range <= 0:
            print('Fitting full-box periodic spectrum from file', spectrum_file)
        else:
            print('Fitting spectrum from %s around galaxy at v=%g +/- %g km/s'
                  % (spectrum_file, gal_velocity_pos, vel_range))
        l        = np.array(f['wavelength'])
        flux     = np.array(f['flux'])
        noise    = np.array(f['noise'])
        vel      = np.array(f['velocity'])
        redshift = np.array(f['redshift'])
        f.close()
    elif line is not None and l is not None:
        print('Fitting profiles for %s spectrum with %d pixels' % (line, len(l)))
    else:
        print('Must provide either spectrum_file or [wave, flux, noise, vel]')
        exit

    if b_bounds[0] == 0.:
        T       = UnitScalar(1.e4, "K")
        b_therm = thermal_b_param(line, T, "km/s")
        b_bounds = [0.25*float(b_therm), b_bounds[1]]

    gal_velocity_pos = None

    spec = Spectrum(
        line, redshift, l, flux, noise, vel,
        gal_velocity_pos=gal_velocity_pos,
        logN_bounds=logN_bounds, b_bounds=b_bounds
    )
    spec.fit_profiles(
        vel_range=vel_range, chisq_lim=chisq_lim,
        chisq_unacceptable=chisq_unacceptable, chisq_factor=chisq_factor,
        max_lines=max_lines, N_sigma_constr=N_sigma_constr
    )
    if write_lines:
        write_line_list(spectrum_file, spec.line_list, spec.regions_l, spec.regions_i)
    if plot_fit:
        spec.plot_fit()


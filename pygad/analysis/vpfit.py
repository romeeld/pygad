""" 
Routines to fit Voigt profiles to absorption line spectra.

usage:
    % pygad.analysis.fit_profiles(ion_name, spectrum_file)

    OR

    % pygad.analysis.fit_profiles(ion_name, wavelengths, fluxes, noise)

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

# from .. import utils
from .. import environment
from ..physics import c
from ..units import Unit, UnitArr, UnitQty, UnitScalar
from .absorption_spectra import line_profile, lines, thermal_b_param
import numpy as np
import os
import h5py
from physics import wave_to_vel, vel_to_wave, tau_to_flux
from utils import read_h5_into_dict
from scipy import signal
import scipy
from scipy.optimize import minimize, NonlinearConstraint
plt.rcParams['text.usetex'] = False


def find_peaks(flux_data,wavelength_data,min_height,distance):

    inverse_flux = 1-flux_data
    min_height = min_height
    distance = distance
    index,height = scipy.signal.find_peaks(inverse_flux,height=min_height,distance=distance)

    widths = scipy.signal.peak_widths(inverse_flux, index)[0]
    wavelength_subset = wavelength_data[index]
    return(index,height,widths)

class Spectrum(object):


    def __init__(self, ion_name, redshift, l, flux, noise, vel, **kwargs):

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
        elif wave is not None:
            vel = l / (lambda_rest * (redshift + 1.0)) * float(c.in_units_of('km/s')) - 1.0
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

        # get the portion of the CGM spectrum that we want to fit

        def _find_nearest(array, value):
            return np.abs(array - value).argmin()

        if self.gal_velocity_pos is not None:
            v_central = self.gal_velocity_pos

        # get the velocity start and end positions
        dv = self.velocities[1] - self.velocities[0]
        v_start = v_central - vel_range
        v_end = v_central + vel_range
        N = int((v_end - v_start) / dv)

        # velocities is assumed to span the entire simulation box
        v_boxsize = self.velocities[-1] - self.velocities[0] + 0.5 * dv

        # get the start and end indices.
        if v_start < 0.:
            v_start += v_boxsize
        i_start = _find_nearest(self.velocities, v_start)
        i_end = i_start + N

        return i_start, i_end, N


    def extend_to_continuum(self, i_start, i_end, N, contin_level=None):

        # from the initial velocity window, extend the start and end back to the level of the continuum of the input spectrum/

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

        # add a buffer to either end of the velocity window at the continuum level to aid the voigt fitting.

        if hasattr(self, 'snr'):
            snr = self.snr
        else:
            snr = snr_default
        dl = waves[1] - waves[0]
        l_start = np.arange(waves[0] - dl*nbuffer, waves[0], dl)
        l_end = np.arange(waves[-1]+dl, waves[-1] + dl*(nbuffer+1), dl)
        
        waves = np.concatenate((l_start, waves, l_end))

        #sigma_noise = 1./snr
        #new_noise = np.random.normal(0.0, sigma_noise, 2*nbuffer)
        new_noise = np.zeros(2*nbuffer)
        flux = np.concatenate((tau_to_flux(np.zeros(nbuffer)) + new_noise[:nbuffer], flux, tau_to_flux(np.zeros(nbuffer)) + new_noise[nbuffer:]))
        
        return waves, flux

    def periodic_wrap(self):
        """
        To avoid situations where the end of a spectrum is in the middle of
        an absorption feature, this routine periodically wraps the spectrum
        such that the endpoint has the highest flux value.  The wavelengths
        are not changed, only the flux and noise vectors are wrapped.
        Should only be used with periodic simulations. Assumes the spectrum
        spans the entire simulation volume.
    
        Args:
            l (numpy array):     list of wavelength for region.
            flux (numpy array):  fluxes of wavelength for region.
            noise (numpy array): noise array (1sigma)
    
        Returns:
            flux:  periodically wrapped fluxes
            noise: periodically wrapped noise array
            starting_pixel: the pixel number where the highest flux occurs
        """
   
        l = self.wavelengths
        flux = self.fluxes
        noise = self.noise
        starting_pixel = np.argmax(flux)
        flux  = np.concatenate((flux[starting_pixel:],  flux[:starting_pixel]))
        noise = np.concatenate((noise[starting_pixel:], noise[:starting_pixel]))
        #flux = np.concatenate((flux[starting_pixel:-1], flux[0 : starting_pixel + 1]))
        #noise = np.concatenate((noise[starting_pixel:-1], noise[0 : starting_pixel + 1]))
        if environment.verbose >= environment.VERBOSE_TACITURN:
            print("Periodically wrapping spectrum, starting_pixel= %d" % starting_pixel)
    
        return flux, noise, starting_pixel

    def periodic_unwrap_wavelength(self):
        """
        After periodic_wrap(), the lines will have the wrong
        wavelength.  This routine 'unwraps' the wavelengths to
        place the line back where it should be.

        Returns:
            waves:  Values of wavelengths after unwrapping.
        """
   
        l = self.wavelengths
        l_box = l[-1] + (l[-1]-l[-2])
        l = l - l[0] + l[self.wrap_pixel]
        l = np.where(l > l_box, l - l_box, l)  # wrap wavelengths

        return l


    def prepare_spectrum(self, vel_range, do_continuum_buffer=False, nbuffer=10, snr_default=30):

        if self.gal_velocity_pos is not None:
            # cut out the portion of the spectrum that we want within some velocity range, making sure the section we cut out 
            # goes back up to the conintuum level (no dicontinuities)
            i_start, i_end, N = self.get_initial_window(vel_range)
            i_start, i_end, N = self.extend_to_continuum(i_start, i_end, N)
            # cope with spectra that go beyond the left hand edge of the box (periodic wrapping)
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

        # extract the wavelengths and fluxes for fitting
        self.waves_fit = self.wavelengths.take(range(i_start, i_end), mode='wrap')
        self.fluxes_fit = self.fluxes.take(range(i_start, i_end), mode='wrap')

        # check if the start and end wavelengths go over the limits of the box
        i_wrap = len(self.wavelengths) - i_start
        wave_boxsize = self.wavelengths[-1] - self.wavelengths[0]
        dl = self.wavelengths[1] - self.wavelengths[0]
        if i_wrap < N:
            # spectrum wraps, i_wrap is the first index of the wavelengths that have been moved to the left side of the box
            self.waves_fit[i_wrap:] += wave_boxsize + dl
            # then for any fitted lines with position outwith the right-most box limits: subtract dl + wave_boxsize

        # add a buffer of continuum to either side to help the voigt fitter identify where to fit
        if do_continuum_buffer is True:
            self.waves_fit, self.fluxes_fit = self.buffer_with_continuum(self.waves_fit, self.fluxes_fit, nbuffer=nbuffer)

        # get the noise level
        if hasattr(self, 'snr'):
            snr = self.snr
        else:
            snr = snr_default
        self.noise_fit = np.asarray([1./snr] * len(self.fluxes_fit))


    def fit_periodic_spectrum(self):

        # the fitting approach for periodic spectra, i.e. those which span the length of the Simba volume

        wrap_flux, wrap_noise, wrap_start = self.periodic_wrap()
        self.line_list = pg.analysis.fit_profiles(self.ion_name, self.wavelengths, wrap_flux, wrap_noise,
                                         chisq_lim=2.0, max_lines=10, logN_bounds=self.logN_bounds, b_bounds=self.b_bounds, mode='Voigt')
        self.line_list['l'] = pg.analysis.periodic_unwrap_wavelength(self.line_list['l'], self.wavelengths, wrap_start)
        self.line_list['v'] = wave_to_vel(self.line_list['l'], self.lambda_rest, self.redshift)

        outwith_vel_mask = ~((self.line_list['v'] > self.gal_velocity_pos - vel_range) & (self.line_list['v'] < self.gal_velocity_pos + vel_range))

        for k in self.line_list.keys():
            self.line_list[k] = np.delete(self.line_list[k], outwith_vel_mask)


    def get_tau_model(self):

        # compute the total optical depth of the model from the individual lines

        self.tau_model = np.zeros(len(self.wavelengths))
        for i in range(len(self.line_list["N"])):
            p = np.array([self.line_list["N"][i], self.line_list["b"][i], self.line_list["l"][i]])
            self.tau_model += model_tau(self.ion_name, p, self.wavelengths, 'Voigt')

    
    def get_fluxes_model(self):

        # compute the total flux from the individual lines

        self.get_tau_model()
        self.fluxes_model = tau_to_flux(self.tau_model)



    def fit_profiles(self, vel_range, do_continuum_buffer=True, nbuffer=50, 
                     snr_default=30., chisq_lim=2.0, chisq_unacceptable=25, 
                     chisq_factor=0.95, N_sigma_constr=3.0, max_lines=12):
 
        # prepare the portion of the spectrum to fit
        # extract from full spectrum, wrap periodically, buffer with a continuum, set the noise level for fitting
        self.prepare_spectrum(vel_range, do_continuum_buffer=True, nbuffer=50, snr_default=30.,)

        # identify regions with significant absorption
        self.regions_l, self.regions_i = find_regions(self.waves_fit, self.fluxes_fit, self.noise_fit, verbose=self.verbose)

        # fit profiles within each region (main routine)
        self.line_list = fit_profiles_sat(self.ion_name, self.waves_fit, self.fluxes_fit, self.noise_fit,
                                          self.regions_l, self.regions_i,
                                          chisq_lim=chisq_lim, chisq_factor=chisq_factor,
                                          chisq_unacceptable=chisq_unacceptable, N_sigma_constr=N_sigma_constr, 
                                          max_lines=max_lines, 
                                          logN_bounds=self.logN_bounds, 
                                          b_bounds=self.b_bounds, mode='Voigt', verbose=self.verbose)
            
        # If necessary, unwrap wavelengths to return to original l values
        if self.wrap_pixel > 0:
            l = self.periodic_unwrap_wavelength()

        # adjust the output lines to cope with wrapping
        #wave_boxsize = self.wavelengths[-1] - self.wavelengths[0]
        #dl = self.wavelengths[1] - self.wavelengths[0]
            
        #for i in range(len(self.line_list['l'])):
        #    if self.line_list['l'][i] > self.wavelengths[-1]:
        #        self.line_list['l'][i]  -= (wave_boxsize + dl)
        #    elif self.line_list['l'][i] < self.wavelengths[0]:
        #        self.line_list['l'][i] += (wave_boxsize + dl)


    def plot_fit(self, ax=None):
    
        # plot the results :)
    
        if ax is None:
            fig, ax = plt.subplots()
    
        x_val = self.wavelengths
    
        ax.plot(x_val, self.fluxes, label='data', c='tab:grey', lw=2, ls='-')
    
        self.get_fluxes_model()
        for i in range(len(self.line_list['N'])):
            p = np.array([self.line_list['N'][i], self.line_list['b'][i], self.line_list['l'][i]])
            _tau_model = model_tau(self.ion_name, p, self.wavelengths)
            #ax.plot(x_val, tau_to_flux(_tau_model), alpha=0.5, lw=1, ls='--', label='%g %g'%(self.line_list['N'][i],self.line_list['b'][i]))
            ax.plot(x_val, tau_to_flux(_tau_model), alpha=0.5, lw=1, ls='--')
            l_cent = self.line_list['l'][i]
            ax.axvline(l_cent, ymin=0.95, ymax=0.98, color='r', linestyle='-', linewidth=1)
    
        ax.plot(x_val, self.fluxes_model, label='model', c='tab:pink', ls='--', lw=2)
    
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(x_val[0], x_val[-1])
        #ax.set_xlim(0, self.gal_velocity_pos +vel_range)
        #ax.set_xlim(max(self.gal_velocity_pos-2*vel_range, 0), min(self.gal_velocity_pos+2*vel_range, self.velocities[-1]))
        ax.legend(loc='best',fontsize=8)
        
        #chisq = np.around(np.unique(self.line_list['Chisq']), 2)
        #chisq = [str(i) for i in chisq]
        #plt.title(r'$\chi^2_r = {x}$'.format(x = ', '.join(chisq) ))
        
        #if filename == None:
        #    filename = self.spectrum_file.split('/')[-1].replace('.h5', '.png')
        #plt.savefig(f'../figures/spec_{i}.png')
        plt.show()
        #plt.savefig('../figures/spec_gal_1.png')
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
    logN_bounds=[8,20],
    b_bounds=[1, 300],
    verbose=False
):
    """
    Fit Voigt/other profiles to the given spectrum.  Begins with one
    line, then adds lines until desired chi-sq is achieved.

    Args:
        ion_name (str):         The line to fit as listed in
                            `analysis.absorption_spectra.lines`, e.g. 'HI1215'.
        l (array-like):     The wavelengths of the input spectrum to fit.
        flux (array-like):  The normalized flux at the given wavelengths,
                            i.e. the spectrum to fit.
        noise (array-like): The normalized 1-sigma noise vector at the given
                            wavelengths.  Must always be >0.
        chisq_lim (float):  Number of sigma below which chi-sq is considered
                            to be a "good fit" and no more lines are added.
                            If <0, then the value used is abs(chisq_lim)+0.1*n_lines,
                            where n_lines is the number of lines for that trial.
        chisq_factor (float):  Factor (<=1) by which chisq must improve to accept added line.
        max_lines (int):    Maximum number of lines allowed in a given detection
                            region, after which the fit declared done regardless of chisq.
                            If limit is hit, this may result in a poor fit.
        mode (str):         Type of line profile: Gaussian/Lorentzian/Voigt;
                            see absorption_spectra.line_profile().
        logN_bounds (list): Initial log(column density) is restricted to this range
        b_bounds (list):    Initial line width is restricted to this range (km/s)
        verbose (boolean):  Verbosity of output to screen


    Returns:
        profiles:    Dictionary of [N, dN, b, db, l, dl, EW] of best-fit
                     profiles.
        tau_model:   Optical depths of best-fit model.

    """

    from .tau_lookup import get_tau_lookup, model_tau_fast
    _lookup = get_tau_lookup(ion_name, mode)

    np.set_printoptions(formatter={'float': '{:.4f}'.format})

    #plt.plot(l,flux)
    if isinstance(ion_name, str):
        line = lines[ion_name]
    l0 = line["l"]
    if isinstance(l, np.ndarray) or l.units in [
        1,
        None,
    ]:  # set units of l to Angstrom if none supplied
        l = UnitArr(l, "Angstrom")

    def _tau_to_flux(tau):  # return flux from tau, avoiding over/underflow
        return np.exp(-np.clip(tau, -50, 50))

    def _chisq(p, l, flux, noise, ion_name, mode):  # reduced chisq
        model_flux = _tau_to_flux(model_tau_fast(ion_name, p, l, mode))
        dx_array = (flux - model_flux) / noise
        #dx_array = np.where(flux < abs(noise), 0., dx_array)
        return np.sum(dx_array * dx_array) / np.count_nonzero(dx_array)

    def _add_line(ion_name, p, bnd, l, flux, noise, l0, mode, i_line=None, grow_line=True):  # adds N, b, l for a new line
        if len(p) == 0:
            resid = flux
        else:
            resid = 1.0 + flux - _tau_to_flux(model_tau(ion_name, p, l, mode))  # residual spectrum
        l_bounds = [l[1], l[-2]]
        #print(l[resid<1],resid[resid<1])

        if grow_line:
            # Grow line to max (N,b) allowed given the residual
            n_guess, b_guess, l_guess = _grow_line(ion_name, l, flux, noise, resid, l0, mode, i_line=i_line)
        else:
            # Make an educated guess at the new line parameters
            b_guess = (
                (l_bounds[1] - l_bounds[0]) / float(l0) * 3.0e5 / 5.0
            )  # first guess at b
            b_guess = max(2 * b_bounds[0], 0.5 * min(b_bounds[1], b_guess))
            n_guess = 14.0 - resid[np.argmin(resid)]  # first guess at logN
            l_guess = l[np.argmin(resid)]

        # append line
        p = np.append(p, n_guess)  # rough guess of logN
        p = np.append(p, b_guess)  # first guess of b
        p = np.append(p, l_guess)  # add line @min of residual flux

        # append bounds
        n_bounds = [n_guess-0.5, n_guess+0.5]
        b_bounds = [b_guess*0.5, b_guess*2]
        if len(bnd) == 0:
            bnd = np.array([n_bounds])
        else:
            bnd = np.append(bnd, np.array([n_bounds]), axis=0)
        bnd = np.append(bnd, np.array([b_bounds]), axis=0)
        bnd = np.append(bnd, np.array([l_bounds]), axis=0)
        return p, bnd

    def _grow_line(ion_name, l, flux, noise, resid, l0, mode,
               i_line=None, floor_sigma=1.5, smooth_sigma=1.,
               unsat_sigma=3.):

        smoothed = scipy.ndimage.gaussian_filter1d(resid, smooth_sigma) if smooth_sigma > 0. else resid
        if i_line is None:
            i_line = np.argmin(smoothed)
        l_line = l[i_line]
    
        # (saturated / unsaturated branch — unchanged) ...
    
        smoothed  = np.minimum(smoothed, 1.0)
        floor     = smoothed - floor_sigma * noise
    
        N_range = np.linspace(logN_bounds[0], logN_bounds[1], 40)
        b_range = np.logspace(np.log10(b_bounds[0]), np.log10(b_bounds[1]), 40)
    
        best_chisq = 1.e20
        best_N, best_b = logN_bounds[0], b_bounds[0]
    
        for bpar in b_range:
            # ── KEY OPTIMISATION: compute profile shape once per b value ──
            tau_unit = model_tau_fast(ion_name, [0.0, bpar, l_line], l, mode)  # logN=0 → N=1
    
            for Ncol in N_range:
                tau_trial = (10.0 ** Ncol) * tau_unit          # linear scaling, no model call
                model_flux = np.exp(-np.clip(tau_trial, -50, 50))
                diff = model_flux - floor
                if np.any(diff < 0):
                    continue                                    # line too strong
    
                # chi-sq near line core only
                i_mid = i_line
                i_lo  = i_mid
                while i_lo > 0           and (1.-model_flux[i_lo])  > 0.5*(1.-model_flux[i_mid]): i_lo -= 1
                i_hi  = i_mid
                while i_hi < len(l) - 2  and (1.-model_flux[i_hi])  > 0.5*(1.-model_flux[i_mid]): i_hi += 1
                sl = slice(i_lo, i_hi + 1)
                dx = (resid[sl] - model_flux[sl]) / noise[sl]
                nz = np.count_nonzero(dx)
                if nz == 0:
                    continue
                chi2 = np.sum(dx * dx) / nz
                if chi2 < best_chisq:
                    best_chisq, best_N, best_b = chi2, Ncol, bpar

        return best_N, best_b, l_line

    def _grow_line_old(ion_name, l, flux, noise, resid, l0, mode, i_line=None, floor_sigma=1.5, smooth_sigma=1., unsat_sigma=3.):  # adds N, b, l for a new line at l by growing N, b
        # compute location of new line, if pixel value not specified in i_line
        if smooth_sigma > 0.:
            smoothed = scipy.ndimage.gaussian_filter1d(resid, smooth_sigma)
        else:
            smoothed = resid
        if i_line is None:
            i_line = np.argmin(smoothed)

        l_line = l[i_line]
        if resid[i_line] < np.min(abs(noise)):
            # if saturated use the center of the saturated region 
            i_lo = i_line
            while resid[i_lo] < unsat_sigma * noise[i_lo] and i_lo > 0: i_lo -= 1
            i_hi = i_line
            while resid[i_hi] < unsat_sigma * noise[i_hi] and i_hi < len(l)-1: i_hi += 1
            i_line = int(0.5 * (i_lo+i_hi))
            l_line = l[i_line]
            N_lim  = logN_bounds[1]
            b_lim = 2.*min(abs(l_line-l[i_lo]),abs(l[i_hi]-l_line)) * float(c.in_units_of('km/s')) / float(l0)
        else:
            # if unsaturated use min of smoothed residual
            N_lim = 15.0
            fdec_bottom = 1.-resid[i_line]
            i_lo = i_line
            while 1.-resid[i_lo] < 0.5 * fdec_bottom and i_lo > 0: i_lo -= 1
            i_hi = i_line
            while 1.-resid[i_hi] < 0.5 * fdec_bottom and i_hi < len(l)-1: i_hi += 1
            b_lim = 4.*min(abs(l_line-l[i_lo]),abs(l[i_hi]-l_line)) * float(c.in_units_of('km/s')) / float(l0)
        b_lim = min(max(b_lim, max(b_bounds[0],20)), b_bounds[1])

        # Set floor which model cannot go below
        smoothed = np.where(smoothed > 1., 1., smoothed)
        floor = smoothed - floor_sigma * noise

        # set up range in N,b
        N_range = np.linspace(start=logN_bounds[0], stop=N_lim, num=40)
        b_range = np.linspace(start=np.log10(b_bounds[0]), stop=np.log10(b_lim), num=40)
        b_range = 10**b_range
        # incremeent N,b until model goes below floor
        N_min = logN_bounds[0]
        p_allowed = np.array([logN_bounds[0], b_bounds[0], l_line])
        chisq = [1.e20]
        for bpar in b_range:
            for Ncol in N_range:
                if Ncol < N_min:
                    continue
                p_trial = np.array([Ncol, bpar, l_line])
                model = _tau_to_flux(model_tau_fast(ion_name, p_trial, l, mode))
                #print(bpar,Ncol,l[0],l[-1],model[0],model[-1],floor[0],floor[-1])
                diff = model-floor
                # Save the largest line that satisfies the condition model>floor everywhere 
                if np.any(diff<0):
                    #print('Cannot add:',bpar,Ncol,np.min(diff), np.argmin(diff), l[np.argmin(diff)], n_exceed)
                    #print(diff)
                    #print(model)
                    #print(floor)
                    #print(flux)
                    continue
                else:
                    #print('Can add:',bpar,Ncol, floor)
                    p_allowed = np.append(p_allowed, p_trial)
                    # compute chi-sq just near this line
                    i_lo = np.argmin(model)
                    while (1.-model[i_lo]) > 0.5 * (1.-model[i_line]) and i_lo > 0: i_lo -= 1
                    i_hi = np.argmin(model)
                    while (1.-model[i_hi]) > 0.5 * (1.-model[i_line]) and i_hi < len(model)-2: i_hi += 1
                    dx_array = (resid[i_lo:i_hi+1] - model[i_lo:i_hi+1]) / noise[i_lo:i_hi+1]
                    i_min = np.argmin(diff)
                    chi2 = np.sum(dx_array * dx_array) / np.count_nonzero(dx_array)
                    chisq.append(chi2)
                    #print('Could add:',ion_name,bpar,Ncol,l_line,np.min(model),np.min(resid),chisq[-1])
        i_p = np.argmin(np.array(chisq))  # from all allowed lines, choose one with lowest chisq
        return p_allowed[3*i_p], p_allowed[3*i_p+1], l_line

    def _maxiter(n, nmax):
        if n <= 5: return 100
        else: return max(50, 50+(nmax-n)*10)

    def _model_flux(p):
        return _tau_to_flux(model_tau_fast(ion_name, p, l_reg, mode, lookup=_lookup))

    def _constraint_jac(p):
        """
        Jacobian of model_flux w.r.t. params p, shape (n_pixels, len(p)).
        Uses central finite differences with step matched to float64 precision.
        """
        eps   = np.sqrt(np.finfo(float).eps)   # ~1.5e-8
        n_p   = len(p)
        n_pix = len(l_reg)
        jac   = np.empty((n_pix, n_p))
        f0    = _tau_to_flux(model_tau_fast(ion_name, p, l_reg, mode, lookup=_lookup))
        for j in range(n_p):
            dp       = np.zeros(n_p)
            dp[j]    = eps * max(abs(p[j]), 1.0)
            f_plus   = _tau_to_flux(model_tau_fast(ion_name, p + dp, l_reg, mode, lookup=_lookup))
            jac[:, j] = (f_plus - f0) / dp[j]
        return jac


################ MAIN CODE FOR FITTING LINES 

    # dicts to store results
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

    # loop over regions
    sat_regions = False
    
    for ireg in range(len(regions_l)):
    #for ireg in range(0,4):
    
        params = []
        bounds = []
        n_lines = 0
        best_nlines = 1
        chisq_old = 1.0e20
        chisq_accept = abs(chisq_lim)
        l_reg = l[regions_i[ireg, 0] : regions_i[ireg, 1]]
        f_reg = flux[regions_i[ireg, 0] : regions_i[ireg, 1]]
        n_reg = noise[regions_i[ireg, 0] : regions_i[ireg, 1]]
        #plt.plot(l_reg,f_reg)


        ###Saturated Region Detection
        regions_l_sat, regions_i_sat, bounding_i = find_saturated_regions(l_reg, f_reg, n_reg, min_region_width=15, verbose=verbose)
        bounding_i = [] # don't use sat region detection
        
        params_reg = []
        bounds_reg =  []
        best_nlines = 0
        if len(bounding_i) != 0:
            sat_regions = True
            if verbose:
                print('Region %d has %d saturated area(s) at pixels:'%(ireg, len(regions_i_sat)), regions_i_sat)
            for ireg_sat in range(len(regions_l_sat)):
                l_reg_sat = l_reg[regions_i_sat[ireg_sat, 0] : regions_i_sat[ireg_sat, 1]]
                f_reg_sat = f_reg[regions_i_sat[ireg_sat, 0] : regions_i_sat[ireg_sat, 1]]
                n_reg_sat = n_reg[regions_i_sat[ireg_sat, 0] : regions_i_sat[ireg_sat, 1]]
                
                #plt.plot(l_reg_sat,f_reg_sat)
                l_reg_bound_left =  l_reg[bounding_i[ireg_sat, 0][0] : bounding_i[ireg_sat, 0][1]]
                l_reg_bound_right =  l_reg[bounding_i[ireg_sat, 1][0] : bounding_i[ireg_sat, 1][1]]
                f_reg_bound_left =  f_reg[bounding_i[ireg_sat, 0][0] : bounding_i[ireg_sat, 0][1]]
                f_reg_bound_right =  f_reg[bounding_i[ireg_sat, 1][0] : bounding_i[ireg_sat, 1][1]]
                n_reg_bound_left = n_reg[bounding_i[ireg_sat, 0][0] : bounding_i[ireg_sat, 0][1]]
                n_reg_bound_right = n_reg[bounding_i[ireg_sat, 1][0] : bounding_i[ireg_sat, 1][1]]
                
                # set up sat region bounds for evaluating chisq
                width = (l_reg_bound_right[0]-l_reg_bound_left[-1])
                l_bounds_sat = [l_reg_bound_left[0], l_reg_bound_right[-1]]
                l_reg_bounds = np.concatenate((l_reg_bound_left,l_reg_bound_right))
                f_reg_bounds = np.concatenate((f_reg_bound_left,f_reg_bound_right))
                n_reg_bounds = np.concatenate((n_reg_bound_left,n_reg_bound_right))

                # set up grid search for best fit
                N_range = np.linspace(start=logN_bounds[0], stop=logN_bounds[1], num=20)
                b_range = np.linspace(start=np.log10(b_bounds[0]), stop=np.log10(b_bounds[1]), num=20)
                b_range = 10**b_range
                i_middle = int((regions_i_sat[ireg_sat, 0] + regions_i_sat[ireg_sat, 1]) / 2)
                middle_guess = l_reg[i_middle]
                bounds = np.array(np.array([logN_bounds]))
                bounds = np.append(bounds, np.array([b_bounds]), axis=0)
                bounds = np.append(bounds, np.array([l_bounds_sat]), axis=0)

                # find best fit
                chisq_best = 1.e20
                for Ncol in N_range:
                    for bpar in b_range:
                        params = np.array([Ncol, bpar, middle_guess])
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
                    params_reg = np.append(params_reg,np.array(params))
                
                if len(bounds_reg) == 0:
                    bounds_reg = np.array(bounds)
                else:
                    bounds_reg = np.append(bounds_reg, np.array(bounds))
            
            if verbose:
                print(
                    "Saturated line gives full region %d (%g-%g): chisq= %g with %d lines"
                    % (
                        ireg,
                        regions_l[ireg, 0],
                        regions_l[ireg, 1],
                        chisq_soln,
                        int(len(params_reg) / 3),
                    )
                )


            params = np.reshape(params_reg,(int(len(params_reg) / 3),3))
            #print(bounds_reg)
            bounds =  np.reshape(bounds_reg,(int(len(params_reg)),2))
            #####Recheck, arbitrary value
            if chisq_soln > 10000:
                params = []
                bounds = []
                sat_regions = False
                print('ChiSquare is too big, probably no saturated region.')

        else:  # fit unsaturated region
            params = []
            bounds = []
            n_lines = 0
            best_nlines = 1
            chisq_old = 1.0e20
            chisq_soln = chisq_old
            chisq_accept = abs(chisq_lim)
            l_reg = l[regions_i[ireg, 0] : regions_i[ireg, 1]]
            f_reg = flux[regions_i[ireg, 0] : regions_i[ireg, 1]]
            n_reg = noise[regions_i[ireg, 0] : regions_i[ireg, 1]]
            

        if len(params) != 0:
            resid = ( 1.0 + f_reg - _tau_to_flux(model_tau(self.ion_name, params.flatten(), l_reg, mode)) )  # residual spectrum
        else:
            distance = int(len(l_reg)/20)
            if distance < 1:
                distance = 1
            #index,height,widths = find_peaks(f_reg,l_reg,0.2,distance)
            #n_lines = len(index)

        # populate a first-guess set of lines in region 
        n_lines = 0
        chisq_best = 1.e20
        chisq_old = 1.e20
        delta_l = l_reg[1]-l_reg[0]
        while n_lines < max_lines-1:
            params, bounds = _add_line(ion_name, params, bounds, l_reg, f_reg, n_reg, float(l0.split()[0]), mode)
            if params[-1] in params[2::3]:
                #print(f'jiggling line {params[-1]} {params[2::3]}')
                params[-1] = params[-1] + delta_l * (0.5*np.random.rand() - 1)  
            n_lines = int(len(params) / 3)
            resid = 1.0 + f_reg - _tau_to_flux(model_tau(ion_name, params.flatten(), l_reg, mode))  # residual spectrum
            chisq_soln = _chisq(params, l_reg, f_reg, n_reg, ion_name, mode)
            if chisq_soln < chisq_best:
                best_nlines = n_lines
                best_params = params
                best_bounds = bounds
                chisq_best = chisq_soln
            if chisq_soln < chisq_accept:  # we're all good :)
                break
            if chisq_soln > chisq_factor * chisq_old:  # we're not improving fast enough :(
                break
            if params[-3] <= logN_bounds[0] and params[-2] <= b_bounds[0] and n_lines > 4:  # line added isn't significant
                params = np.delete(params, [-3, -2, -1], axis=0)
                bounds = np.delete(bounds, [-3, -2, -1], axis=0)
                n_lines = int(len(params) / 3)
                chisq_soln = _chisq(params, l_reg, f_reg, n_reg, ion_name, mode)
                break
            chisq_old = chisq_soln
            if verbose:
                print(f'Region {ireg}: Added line {n_lines-1} with N=%.4f, b=%.4f, l=%.4f, chisq=%.3f' % (params[-3],params[-2],params[-1],chisq_soln))
        if verbose:
            print(f'Region {ireg}: Found {n_lines} lines in first guess, chisq=%.3f'%chisq_soln)

        # loop to add lines until desired chisq achieved
        best_nlines = n_lines
        best_params = params
        best_bounds = bounds
        chisq_best = chisq_soln
        first_time = True
        while n_lines < max_lines and chisq_soln > chisq_accept:
            if not first_time:
                params, bounds = _add_line(ion_name, params, bounds, l_reg, f_reg, n_reg, float(l0.split()[0]), mode)
                n_lines = int(len(params) / 3)
            # Do a constrained minimization, trying to keep the model above flux - N_sigma_constr * noise
            chisq_fcn = lambda *args: _chisq(*args)
            constraint = NonlinearConstraint(
                fun = _model_flux,
                lb  = f_reg - N_sigma_constr * n_reg,  # lower bound per pixel
                ub  = np.inf,                           # no upper bound on model flux
                jac=_constraint_jac,
            )
            soln = minimize(
                chisq_fcn,
                params,
                bounds=bounds,
                args=(l_reg, f_reg, n_reg, ion_name, mode),
                method="trust-constr",
                constraints = constraint,
                options={"maxiter": _maxiter(n_lines, max_lines), "gtol": 1e-8},
            )
            params = soln.x  # set params to new best-fit values
            chisq_soln = _chisq(params, l_reg, f_reg, n_reg, ion_name, mode)
            if verbose and not first_time:
                if first_time:
                    print( "Region %d: With %d lines after %d iters, chisq=%.3f" % (ireg, n_lines, soln.nit, chisq_soln))
                else:
                    print( "Region %d: Added new line %d (N=%g), after %d iters, chisq=%.3f" % (ireg, n_lines, params[-3], soln.nit, chisq_soln))
            first_time = False
            # Require non-trivial improvement
            if chisq_soln < chisq_factor * chisq_best:
                best_nlines = n_lines
                best_params = params
                best_bounds = bounds
                chisq_best = chisq_soln
            # keep trying for small number of lines even if little improvement
            elif n_lines <= 2 and chisq_soln < chisq_best:
                best_nlines = n_lines
                best_params = params
                best_bounds = bounds
                chisq_best = chisq_soln
                continue
            # keep trying if chisq is very high
            elif n_lines <= 6 and chisq_soln > chisq_unacceptable: 
                if chisq_soln < chisq_best:
                    best_nlines = n_lines
                    best_params = params
                    best_bounds = bounds
                    chisq_best = chisq_soln
                continue
            # if it's not improving enough reset to previous best params and stop
            else:
                params = best_params
                bounds = best_bounds
                n_lines = int(len(params) / 3)
                chisq_soln = chisq_best
                break

        # jiggle params and refit to compute hessian
        compute_errors = True
        delta_params = [0.02, 0.05, 0.0001] * n_lines
        if compute_errors:
            params_jiggled = params + delta_params * ( 2 * np.random.rand(len(params)) - 1)  
            chisq_fcn = lambda *args: _chisq(*args)
            soln = minimize(
                chisq_fcn,
                params_jiggled,
                args=(l_reg, f_reg, n_reg, ion_name, mode),
                method="BFGS",
                options={"maxiter": 100},
            )
            cov = soln.hess_inv  # covariance matrix of final soluiton
            # if jiggled param is better (shouldn't usually happen), then use that
            #chisq_trial = _chisq(params_jiggled, l_reg, f_reg, n_reg, ion_name, mode)
            #if chisq_trial < chisq_best:
            #    if verbose:
            #        print(f'Region {ireg}: Fit improved with jiggled params chisq={chisq_trial}')
            #    params = soln.x
            #    n_lines = int(len(params) / 3)
            #    best_nlines = n_lines
            #    best_params = params_jiggled
            #    best_bounds = bounds
            #    chisq_best = chisq_trial

        # remove small lines as long as chisq doesn't go up by much
        while n_lines > 1:
            for i in range(n_lines):
                trial_params = params.copy()
                #i_del = no.argmin(np.array([params[ip*3] for ip in np.arange(n_lines)]))
                i_del = 3*i
                trial_params = np.delete(trial_params, [i_del, i_del+1, i_del+2], axis=0)
                chisq_trial = _chisq(trial_params, l_reg, f_reg, n_reg, ion_name, mode)
                delta_chisq = abs(chisq_trial-chisq_best)/chisq_trial
                if delta_chisq < 0.01 or chisq_trial < chisq_accept:
                    if verbose:
                        print("Region %d: Removed line %d (N=%g): chisq=%g, chisq_old=%g"%(ireg, i_del, params[3*i], chisq_trial, chisq_best))
                    params = trial_params.copy()
                    bounds = np.delete(bounds, [i_del, i_del+1, i_del+2], axis=0)
                    chisq_best = chisq_trial
                    n_lines = int(len(params)/3)
                    break
                else:
                    continue
            if i >= n_lines-2:
                break

        # Try combining adjacent lines to lower the number
        while n_lines > 1:
            for i in range(n_lines-1):
                trial_params = params.copy()
                # Combines lines i and i+1
                ip = 3*i
                ip1 = 3*(i+1)
                N_i = 10**params[ip]
                N_i1 = 10**params[ip1]
                trial_params[ip] = np.log10(N_i + N_i1)
                trial_params[ip+1] = (N_i * params[ip+1] + N_i1 * params[ip+4]) / (N_i + N_i1)
                trial_params[ip+2] = (N_i * params[ip+2] + N_i1 * params[ip+5]) / (N_i + N_i1)
                trial_params = np.delete(trial_params, [ip+3, ip+4, ip+5], axis=0)
                chisq_trial = _chisq(trial_params, l_reg, f_reg, n_reg, ion_name, mode)
                delta_chisq = abs(chisq_trial-chisq_best)/chisq_trial
                if delta_chisq < 0.01 or chisq_trial < chisq_accept:
                    if verbose:
                        print("Region %d: Combining lines %d and %d (N=%g and %g): chisq=%g, chisq_old=%g"%(ireg, i, i+1, N_i, N_i1, chisq_trial, chisq_best))
                    params = trial_params.copy()
                    bounds = np.delete(bounds, [ip+3, ip+4, ip+5], axis=0)
                    chisq_best = chisq_trial
                    n_lines = int(len(params)/3)
                    break
                else:
                    continue
            if i >= n_lines-2:
                break

        # load final line list
        chisq_soln = _chisq(params, l_reg, f_reg, n_reg, ion_name, mode)
        for ip in np.arange(n_lines):
            line_list["region"] = np.append(line_list["region"], ireg)
            line_list["N"] = np.append(line_list["N"], params[ip * 3])
            line_list["b"] = np.append(line_list["b"], params[ip * 3 + 1])
            line_list["l"] = np.append(line_list["l"], params[ip * 3 + 2])
            line_list["dN"] = np.append(line_list["dN"], np.sqrt(cov[ip * 3, ip * 3]))
            line_list["db"] = np.append(
                line_list["db"], np.sqrt(cov[ip * 3 + 1, ip * 3 + 1])
            )
            line_list["dl"] = np.append(
                line_list["dl"], np.sqrt(cov[ip * 3 + 2, ip * 3 + 2])
            )
            tau_line = model_tau(
                ion_name,
                [params[ip * 3], params[ip * 3 + 1], params[ip * 3 + 2]],
                l_reg,
                mode,
            )
            line_list["EW"] = np.append(
                line_list["EW"], EquivalentWidth(_tau_to_flux(tau_line), l_reg)
            )
            line_list["Chisq"] = np.append(line_list["Chisq"], chisq_soln)


        if verbose:
            print(f"Region {ireg}: FINAL FIT {n_lines} lines, N={params[0::3]}, chisq=%.3f"%(chisq_soln))
            if chisq_soln > chisq_accept:
                print( "Region %d: WARNING large chisq=%.3f > %.3f; check fit" % (ireg, chisq_soln, chisq_accept))

    # Now look at entire spectrum vs. full model to see if any lines can be removed/combined/reduced
    n_lines = len(line_list["N"])
    params = []
    for ip in range(n_lines):
        #print(ip, n_lines, line_list["N"][ip], line_list["b"][ip], line_list["l"][ip])
        params.append(line_list["N"][ip])
        params.append(line_list["b"][ip])
        params.append(line_list["l"][ip])
    params = np.array(params)
    chisq_soln = _chisq(params, l, flux, noise, ion_name, mode)

    """
    # Try to remove lines if it improves chisq
    while n_lines > 1:
        for i in range(n_lines):
            trial_params = params.copy()
            i_del = 3*i
            trial_params = np.delete(trial_params, [i_del, i_del+1, i_del+2], axis=0)
            chisq_trial = _chisq(trial_params, l, flux, noise, ion_name, mode)
            delta_chisq = abs(chisq_trial-chisq_soln)/chisq_trial
            if chisq_trial < chisq_soln:
                if verbose:
                    print("Full spectrum: Removed line %d (N=%g): chisq=%g, chisq_old=%g"%(i_del, params[3*i], chisq_trial, chisq_soln))
                params = trial_params.copy()
                chisq_soln = chisq_trial
                line_tracker[i] = 0
                n_lines = int(len(params)/3)
                break
            else:
                continue
        if i >= n_lines-2:
            break

    # Try combining adjacent lines if it improves the fit.
    while n_lines > 1:
        for i in range(n_lines-1):
            trial_params = params.copy()
            # Combines lines i and i+1
            ip = 3*i
            ip1 = 3*(i+1)
            N_i = 10**params[ip]
            N_i1 = 10**params[ip1]
            trial_params[ip] = np.log10(N_i + N_i1)
            trial_params[ip+1] = (N_i * params[ip+1] + N_i1 * params[ip+4]) / (N_i + N_i1)
            trial_params[ip+2] = (N_i * params[ip+2] + N_i1 * params[ip+5]) / (N_i + N_i1)
            trial_params = np.delete(trial_params, [ip+3, ip+4, ip+5], axis=0)
            chisq_trial = _chisq(trial_params, l, flux, noise, ion_name, mode)
            delta_chisq = abs(chisq_trial-chisq_soln)/chisq_trial
            if chisq_trial < chisq_soln:
                if verbose:
                    print("Full spectrum: Combining lines %d and %d (N=%g and %g): chisq=%g, chisq_old=%g"%(i, i+1, N_i, N_i1, chisq_trial, chisq_soln))
                params = trial_params.copy()
                chisq_soln = chisq_trial
                line_tracker[i+1] = 0
                n_lines = int(len(params)/3)
                break
            else:
                continue
        if i >= n_lines-2:
            break
    """

    # Try reducing columns. This can work because fitting each region separately can result in cumulative excess absorption  overall.
    chisq_trial = 0.
    f_reduce = 0.999
    while chisq_trial < chisq_soln:
        chisq_trial = chisq_soln
        trial_params = params.copy()
        trial_params[::3] *= f_reduce
        chisq_trial = _chisq(trial_params, l, flux, noise, ion_name, mode)
        #print(chisq_trial, chisq_soln, trial_params[::3])
        if chisq_trial < chisq_soln:
            if verbose:
                print(f"Multiplying all column densities by %d improves overall fit from chisq=%g to %g" % (f_reduce, chisq_soln, chisq_trial))
            params = trial_params.copy()
            chisq_soln = chisq_trial

    n_lines = int(len(params)/3)
    if verbose:
        print(f"Full spectrum: FINAL FIT {n_lines} lines in %d regions, chisq=%.3f"%(len(regions_l), chisq_soln))
        for ip in range(n_lines):
            print(ip, params[3*ip], params[3*ip+1], params[3*ip+2])

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
    Finds detection regions above some detection threshold and minimum width.

    Args:
        wavelengths (numpy array)
        fluxes (numpy array): flux values at each wavelength
        noise (numpy array): noise value at each wavelength
        min_region_width (int): minimum width of a detection region (pixels)
        N_sigma (float): detection threshold (std deviations)
        extend (boolean): default is False. Option to extend detected regions untill tau
                        returns to continuum.

    Returns:
        regions_l (numpy array): contains subarrays with start and end wavelengths
        regions_i (numpy array): contains subarrays with start and end indices
    """

    num_pixels = len(wavelengths)
    pixels = range(num_pixels)
    min_pix = 1
    max_pix = num_pixels - 1

    flux_ews = [0.0] * num_pixels
    noise_ews = [0.0] * num_pixels
    det_ratio = [-float("inf")] * num_pixels

    # flux_ews has units of wavelength since flux is normalised. so we can use it for optical depth space
    for i in range(min_pix, max_pix):
        flux_dec = 1.0 - fluxes[i]
        if flux_dec < noise[i]:
            flux_dec = 0.0
        flux_ews[i] = 0.5 * abs(wavelengths[i - 1] - wavelengths[i + 1]) * flux_dec
        noise_ews[i] = 0.5 * abs(wavelengths[i - 1] - wavelengths[i + 1]) * noise[i]

    # dev: no need to set end values = 0. since loop does not set end values
    flux_ews[0] = 0.0
    noise_ews[0] = 0.0

    # Range of standard deviations for Gaussian convolution
    std_min = 2
    std_max = 11

    # Convolve varying-width Gaussians with equivalent width of flux and noise
    xarr = np.array([p - (num_pixels - 1) / 2.0 for p in range(num_pixels)])

    # this part can remain the same, since it uses EW in wavelength units, not flux
    for std in range(std_min, std_max):
        gaussian = np.exp(-0.5 * (xarr / std) ** 2)

        flux_func = np.convolve(flux_ews, gaussian, "same")
        noise_func = np.convolve(np.square(noise_ews), np.square(gaussian), "same")

        # Select highest detection ratio of the Gaussians
        for i in range(min_pix, max_pix):
            noise_func[i] = 1.0 / np.sqrt(noise_func[i])
            if flux_func[i] * noise_func[i] > det_ratio[i]:
                det_ratio[i] = flux_func[i] * noise_func[i]

    if det_flag:
        return [], []

    # Select regions based on detection ratio at each point, combining nearby regions
    start = 0
    region_endpoints = []
    for i in range(num_pixels):
        if start == 0 and det_ratio[i] > 0 and fluxes[i] < 1.0:  ##greater 0
            start = i
        elif start != 0 and (det_ratio[i] < 0 or fluxes[i] > 1.0):
            # if (i - start) > min_region_width:
            end = i

            region_endpoints.append([start, end])
            start = 0

    significant_region_endpoints = []
    for reg in region_endpoints:
        det_ratio = np.array(det_ratio)
        significance = np.sqrt(np.sum(det_ratio[reg[0] : reg[1]] ** 2))
        # if reg[1]-reg[0] >10:
        # print(reg)
        # print(significance)
        # plt.plot(wavelengths[reg],fluxes[reg])

        if significance == np.inf:
            significance = 0
        if significance > N_sigma:  # and reg[1]>60 and reg[0]< (len(fluxes)-60):
            significant_region_endpoints.append(reg)
    # made extend a kwarg option
    # lines may not go down to 0 again before next line starts

    if extend:
        # Expand edges of region until flux goes above 1
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
    # print(significant_region_endpoints)
    # Change to return the region indices
    # Combine overlapping regions, check for detection based on noise value
    # and extend each region again by a buffer
    regions_l = []
    regions_i = []
    buffer = buffer
    for i in range(len(regions_expanded) - 1):
        # print(len(regions_expanded),i)
        if len(regions_expanded) == i:
            break
        start = regions_expanded[i][0]
        end = regions_expanded[i][1]
        # print(start,end)
        # print(regions_expanded[i+1])
        # print('difference'+str(regions_expanded[i+1][0]-end))
        if len(regions_expanded) == i + 1:
            break
        if (regions_expanded[i + 1][0] - end) < 5:
            regions_expanded[i][1] = regions_expanded[i + 1][1]
            regions_expanded = np.delete(regions_expanded, (i + 1), axis=0)
        # print(regions_expanded)

        end_init = end
    for i in range(len(regions_expanded)):
        start = regions_expanded[i][0]
        end = regions_expanded[i][1]

        if i > 0:
            start_min = regions_expanded[i-1][1] - buffer
        for j in range(start, end):
            flux_dec = 1.0 - fluxes[j]
            if start >= buffer:
                start -= buffer
            # cannot start too far within previous region
            if i > 0 and start < start_min:
                start = start_min
            if end < len(wavelengths) - buffer:
                end += buffer
            regions_expanded[i][0] = start
            regions_expanded[i][1] = end
            regions_l.append([wavelengths[start], wavelengths[end]])
            regions_i.append([start, end])

            break

    # combine regions if too overlapping 
    while len(regions_l) > 100000:  # this is not used; gives too large regions
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
    wavelengths, fluxes, noise, min_region_width=2, N_sigma=10.0, extend=False, verbose=False
):
    """
    Finds detection regions above some detection threshold and minimum width.

    Args:
        wavelengths (numpy array)
        fluxes (numpy array): flux values at each wavelength
        noise (numpy array): noise value at each wavelength
        min_region_width (int): minimum width of a detection region (pixels)
        N_sigma (float): detection threshold (std deviations)
        extend (boolean): default is False. Option to extend detected regions untill tau
                        returns to continuum.

    Returns:
        regions_l (numpy array): contains subarrays with start and end wavelengths
        regions_i (numpy array): contains subarrays with start and end indices
    """

    num_pixels = len(wavelengths)
    pixels = range(num_pixels)
    min_pix = 1
    max_pix = num_pixels - 1

    flux_ews = [0.0] * num_pixels
    noise_ews = [0.0] * num_pixels
    det_ratio = [-float("inf")] * num_pixels

    # flux_ews has units of wavelength since flux is normalised. so we can use it for optical depth space
    for i in range(min_pix, max_pix):
        flux_dec = 1.0 - fluxes[i]
        if flux_dec < noise[i]:
            flux_dec = 0.0
        flux_ews[i] = 0.5 * abs(wavelengths[i - 1] - wavelengths[i + 1]) * flux_dec
        noise_ews[i] = 0.5 * abs(wavelengths[i - 1] - wavelengths[i + 1]) * noise[i]

    # dev: no need to set end values = 0. since loop does not set end values
    flux_ews[0] = 0.0
    noise_ews[0] = 0.0

    # Range of standard deviations for Gaussian convolution
    std_min = 2
    std_max = 11

    # Convolve varying-width Gaussians with equivalent width of flux and noise
    xarr = np.array([p - (num_pixels - 1) / 2.0 for p in range(num_pixels)])

    # this part can remain the same, since it uses EW in wavelength units, not flux
    for std in range(std_min, std_max):

        gaussian = np.exp(-0.5 * (xarr / std) ** 2)

        flux_func = np.convolve(flux_ews, gaussian, "same")
        noise_func = np.convolve(np.square(noise_ews), np.square(gaussian), "same")

        # Select highest detection ratio of the Gaussians
        for i in range(min_pix, max_pix):
            noise_func[i] = 1.0 / np.sqrt(noise_func[i])
            if flux_func[i] * noise_func[i] > det_ratio[i]:
                det_ratio[i] = flux_func[i] * noise_func[i]

    # Select regions based on detection ratio at each point, combining nearby regions
    start = 0
    region_endpoints = []
    index = np.where(np.abs(det_ratio-np.max(det_ratio)) < N_sigma)[0]
    
    for i in range(num_pixels):
        if start == 0 and np.abs(det_ratio[i]-np.max(det_ratio)) < N_sigma: # and fluxes[i] < 1.0:
            start = i
        elif start != 0 and  np.abs(det_ratio[i]-np.max(det_ratio)) > N_sigma: # or fluxes[i] > 1.0):
            if (i - start) > min_region_width:
                end = i
                region_endpoints.append([start, end])
            start = 0

    # made extend a kwarg option
    # lines may not go down to 0 again before next line starts

    if extend:
        # Expand edges of region until flux goes above 1
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

    # Change to return the region indices
    # Combine overlapping regions, check for detection based on noise value
    # and extend each region again by a buffer
    regions_l = []
    regions_i = []
    bounding_regions_i = []
    buffer = 3
    for i in range(len(regions_expanded)):
        start = regions_expanded[i][0]
        end = regions_expanded[i][1]
        end_init = end
        # TODO: this part seems to merge regions if they overlap - try printing this out to see if it can be modified to not merge regions?
        if i < (len(regions_expanded) - 1) and np.abs(end -regions_expanded[i + 1][0])<2:
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

                if (start-18)<0:
                    start1 = 0
                else:
                    start1 = (start-18)
                if (start-8)< 0:
                    start2 = start
                else:
                    start2 = (start-8)
                if (end+8) >len(fluxes):
                    end1 = end
                else:
                    end1 = (end+8)
                if (end+18) > len(fluxes):
                    end2 = int(len(fluxes))
                else:
                    end2 = (end+18)
                
                bounding_regions_i.append([[start1,start2],[end1,end2]])

                #print(regions_l,regions_i)
                break

    #bounding_regions_i = [regions_i[0]-10,regions_i[-1]+10]
    return np.array(regions_l), np.array(regions_i), np.array(bounding_regions_i)





def model_tau(ion_name, p, l, mode="Voigt"):
    """
    Compute optical depth vs. wavelength for a set of lines.

    Args:
        p (numpy array): [logN,b,wavelength] for a set of lines, in
                         a flattened numpy array.
        l (numpy array): Wavelengths over which to compute model spectrum.
    Returns:
        total_tau:  Optical depths (vs. l) from the combined set of lines
    """
    p = np.array(p)
    total_tau = np.zeros(len(l), dtype=float)
    line = lines[ion_name]
    if len(p) == 0:
        return total_tau  # no lines yet, return zeros
    for ip in range(int(len(p) / 3)):
        _, tau = line_profile(
            line, 10 ** p[ip * 3], b=p[ip * 3 + 1], l0=p[ip * 3 + 2], l=l, mode=mode
        )
        total_tau += tau  # add up optical depths from all lines
    return total_tau


def EquivalentWidth(fluxes, waves):
    """
    Find the equivalent width of a line/region.

    Args:
        taus (numpy array): the optical depths.
        waves (numpy array): list of wavelength for region.
    Returns:
        Equivalent width in units of waves.
    """
    fluxes = np.asarray(fluxes, dtype=float)
    waves  = np.asarray(waves,  dtype=float)
    dwave          = np.empty_like(waves)
    dwave[1:-1]    = 0.5 * np.abs(waves[2:] - waves[:-2])
    dwave[0]       = np.abs(waves[1]  - waves[0])
    dwave[-1]      = np.abs(waves[-1] - waves[-2])
    return float(np.sum((1.0 - fluxes) * dwave))

def write_spectrum(
    spec_name,
    line,
    LOS_pos,
    lambda_rest,
    redshift,
    vels,
    fluxes,
    taus,
    noise,
    col_dens,
    phys_dens,
    temps,
    mets,
    vpec,
    overwrite=True,
):
    """
    Output spectrum to hdf5 file.

    Args:
        spec_name (str):      Name of file to write spectrum out to.
                              '.h5' will be appended to this.
        line (str):           The ion name, e.g. 'H1215'
        LOS_pos (list/array): (x,y,z) position of LOS,
                              with the LOS axis holding a value of -1.
        lambda_rest (float):  Rest wavelength of ion
        redshift (float):     Redshift of snapshot
        vels (list/array):    Velocities of pixels.
        fluxes (list/array):  Normalized fluxes of pixels; this should include
                              noise, smoothing, etc. so it's NOT =exp(-taus).
        taus (list/array):    Optical depths of pixels
        noise (list/array):   1-sigma noise array of pixels
        col_dens (list/array): Column densities for each pixel
        phys_dens (list/array): Tau-weighted physical densities for each pixel
        temps (list/array):   Tau-weighted gas temperatures for each pixel
        mets (list/array):    Tau-weighted metal mass fractions for each pixel
        vpec (list/array):    LOS peculiar velocity for each pixel

    Returns:

    """
    import os

    import h5py

    if os.path.isfile(spec_name) and not overwrite:
        if environment.verbose >= environment.VERBOSE_TACITURN:
            print(
                (
                    "WARNING: write_spectrum() failed: File %s exists, and overwrite set to False"
                    % spec_name
                )
            )
        return

    waves = lambda_rest * (redshift + 1.0) * (1.0 + vels / c)
    mets = np.log10(np.where(mets < 1.0e-10, 1.0e-10, mets))  # turn into log10(Z)
    if len(LOS_pos) == 2:
        LOS_pos = np.append(
            np.array(LOS_pos), -1.0
        )  # assumes if only 2 values are provided, they are (x,y), so we add -1 for z.

    with h5py.File("%s.h5" % spec_name, "w") as hf:
        lam0 = hf.create_dataset("lambda_rest", data=lambda_rest)
        lam0.attrs["ion_name"] = line  # store line name as attribute of rest wavelength
        hf.create_dataset("LOS_pos", data=np.array(LOS_pos))
        hf.create_dataset("redshift", data=redshift)
        hf.create_dataset("velocity", data=np.array(vels))
        hf.create_dataset("wavelength", data=np.array(waves))
        hf.create_dataset("flux", data=np.array(fluxes))
        hf.create_dataset("tau", data=np.array(taus))
        hf.create_dataset("noise", data=np.array(noise))
        hf.create_dataset("col_density", data=np.array(col_dens))
        hf.create_dataset("phys_density", data=np.array(phys_dens))
        hf.create_dataset("temperature", data=np.array(temps))
        hf.create_dataset("metallicity", data=np.array(mets))
        hf.create_dataset("vpec", data=np.array(vpec))

    return

def write_line_list(spec_name, line_list, regions_l, regions_i):
    """
    Append profile fit information to spectrum file.  Spectrum file must exist
    and contain the ion name attribute and wavelength list.

    Args:
        spec_name (str):      Name of existing file to append lines info.
                              Info placed in hdf5 group called 'line_list'.
        line_list (dict):     List of fitted absorption feature
                              as output by fit_profiles().
        regions_l (float):    [start,end] wavelengths of detection regions
        regions_i (int):      [start,end] pixels of detection regions

    Returns:

    """
    import h5py

    with h5py.File(spec_name, "r") as hf:
        line = hf["lambda_rest"].attrs["ion_name"]
        waves = np.array(hf["wavelength"])

    # create overall model spectrum from all lines combined
    tau_model = np.zeros(len(waves))
    for i in range(len(line_list["N"])):
        p = np.array([line_list["N"][i], line_list["b"][i], line_list["l"][i]])
        tau_model += model_tau(line, p, waves)
    model_flux = np.exp(-np.clip(tau_model, -30, 30))

    # load data into arrays
    regions_l0 = [x[0] for x in regions_l]
    regions_l1 = [x[1] for x in regions_l]
    regions_i0 = [x[0] for x in regions_i]
    regions_i1 = [x[1] for x in regions_i]
    ireg = line_list["region"]
    N = line_list["N"]
    dN = line_list["dN"]
    b = line_list["b"]
    db = line_list["db"]
    l = line_list["l"]
    dl = line_list["dl"]
    EW = line_list["EW"]
    chisq = line_list["Chisq"]

    with h5py.File(spec_name, "a") as hf:
        if "line_list" in hf.keys():
            if environment.verbose >= environment.VERBOSE_TACITURN:
                print("Deleting and replacing line_list in %s" % spec_name)
            del hf["line_list"]
        elif "lines" in hf.keys():
            del hf["lines"]
        lines = hf.create_group("line_list")
        lines.create_dataset("region", data=np.array(line_list["region"], dtype=int))
        lines.create_dataset("logN", data=np.array(N))
        lines.create_dataset("dlogN", data=np.array(dN))
        lines.create_dataset("b", data=np.array(b))
        lines.create_dataset("db", data=np.array(db))
        lines.create_dataset("l", data=np.array(l))
        lines.create_dataset("dl", data=np.array(dl))
        lines.create_dataset("EW", data=np.array(EW))
        lines.create_dataset("chisq", data=np.array(chisq))
        lines.create_dataset("model_flux", data=np.array(model_flux))
        lines.create_dataset("region_lam_start", data=np.array(regions_l0))
        lines.create_dataset("region_lam_end", data=np.array(regions_l1))
        lines.create_dataset("region_pix_start", data=np.array(regions_i0, dtype=int))
        lines.create_dataset("region_pix_end", data=np.array(regions_i1, dtype=int))

    return


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
):
    """
    Fit Voigt/other profiles to the given spectrum.  Begins with one
    line, then adds lines until desired chi-sq is achieved.

    Args:
        line (str):         The line to fit as listed in
                            `analysis.absorption_spectra.lines`, e.g. 'HI1215'.
        spectrum_file (str): Import l,flux,noise,gal_v_pos,vel_range from pygad spectrum file.
                            If this is left as None (default), must input [line, l, flux, noise].
        l (array-like):     The wavelengths of the input spectrum to fit.  One of vel or l must
                            be provided. If both are provided, vel is used to calculate l.
        vel (array-like):   The velocities of the input spectrum to fit.  See above note.
        flux (array-like):  The normalized flux as a function of wavelength,
                            i.e. the spectrum to fit.
        noise (array-like): The normalized 1-sigma noise vector as a function of wavelength.
        gal_v_pos (float):  Velocity space position of galaxy around which to fit, if desired
        vel_range (float):  Velocity space +/- range around galaxy to fit; default is whole spectrum
                            gal_v_pos must be specified for this option to be meaningful.
        chisq_lim (float):  Number of sigma below which chi-sq is considered
                            to be a "good fit" and no more lines are added.
                            If <0, then the value used is abs(chisq_lim)+0.1*n_lines,
                            where n_lines is the number of lines for that trial.
        chisq_unacceptable (float):  If chisq > chisq_unacceptable, throws out entire 
                            fit and tries with a new set of lines.
        chisq_factor (float):  chi-square must improve by this factor with each
                            added line, otherwise trial line is not accepted.
        max_lines (int):    Maximum number of lines allowed in a given detection
                            region, after which the fit declared done regardless of chisq.
                            If limit is hit, this may result in a poor fit.
        N_sigma_constr (float): Try to constrain minimizer to have model always above
                            flux minus N_sigma_constr * noise
        mode (str):         Type of line profile: Gaussian/Lorentzian/Voigt;
                            see absorption_spectra.line_profile().
        logN_bounds (list): Initial log(column density) is restricted to this range
        b_bounds (list):    Initial line width is restricted to this range (km/s).
                            If 'None', sets to thermal width at logT=4-7 for given ion.


    Returns:
        profiles:    Dictionary of [N, dN, b, db, l, dl, EW] of best-fit
                     profiles.
        tau_model:   Optical depths of best-fit model.

    """

    if spectrum_file is not None:
        # input spectrum taken from pygad-generated spectrum file
        #data = read_h5_into_dict(spectrum_file)
        #for key in data:
        #    setattr(self, key, data[key])
        #del data
        f = h5py.File(spectrum_file, 'r')
        try:
            gal_velocity_pos = np.array(f['galaxy_properties/vlos'])
        except:
            gal_velocity_pos = gal_v_pos
        if gal_velocity_pos is None or vel_range <= 0:
            print('Fitting full-box periodic spectrum from file',spectrum_file)
        else:
            print('Fitting spectrum from %s around galaxy at v=%g +/- %g km/s from' % (spectrum_file, gal_velocity_pos, vel_range))
        l = np.array(f['wavelength'])
        flux = np.array(f['flux'])
        noise = np.array(f['noise'])
        vel = np.array(f['velocity'])
        redshift = np.array(f['redshift'])
        f.close()
    elif line is not None and l is not None:
        # input spectrum provided on input
        print('Fitting profiles for %s spectrum with %d pixels' % (line, len(l)))
    else:
        print('Must provide either spectrum_file or [wave,flux,noise,vel]')
        exit

    # if not provided, make a reasonable guess at b bounds based on element
    if b_bounds[0] == 0.:
        T = UnitScalar(1.e4, "K")
        b_therm = thermal_b_param(line, T, "km/s")
        b_bounds = [0.25*float(b_therm), b_bounds[1]]

    gal_velocity_pos = None

    spec = Spectrum(line, redshift, l, flux, noise, vel, gal_velocity_pos=gal_velocity_pos, logN_bounds=logN_bounds, b_bounds=b_bounds)
    spec.fit_profiles(vel_range=vel_range, chisq_lim=chisq_lim, chisq_unacceptable=chisq_unacceptable, chisq_factor=chisq_factor, max_lines=max_lines, N_sigma_constr=N_sigma_constr)
    write_line_list(spectrum_file, spec.line_list, spec.regions_l, spec.regions_i)
    spec.plot_fit()


# run as standalone code 

#if __name__ == '__main__':
#
#    spectrum_file = sys.argv[1]
#    ion_name = sys.argv[2]
#    fit_profiles(ion_name, spectrum_file=spectrum_file, vel_range=600)
#
    #spec_file = 'sample_galaxy_811_MgII2796_270_deg_0.75r200.h5'
    
    

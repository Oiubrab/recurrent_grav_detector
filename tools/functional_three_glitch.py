# python script for all the defined functions in the scavanger of black holes script

import numpy as np
import matplotlib.pyplot as plt
import lalsimulation as lalsim 
from scipy import signal
import random as ran
from machine_run import *

# load in the time series construction constants
srate, f_min, f_max, duration, df, lo, hi, dur_sig, df_sig, lo_sig, hi_sig, N_fd = machine_configuration()

def frequencies(sampling_frequency, dur):
    """ Create a frequency series with the correct length and spacing.

    Parameters
    -------
    sampling_frequency: float
    duration: float
        duration of data

    Returns
    -------
    array_like: frequency series

    """
    number_of_samples = dur * sampling_frequency
    number_of_samples = int(np.round(number_of_samples))

    # prepare for FFT
    number_of_frequencies = (number_of_samples - 1) // 2
    delta_freq = 1. / dur

    frequencies = delta_freq * np.linspace(1, number_of_frequencies, number_of_frequencies)

    if len(frequencies) % 2 == 1:
        frequencies = np.concatenate(([0], frequencies, [sampling_frequency / 2.]))
    else:
        # no Nyquist frequency when N=odd
        frequencies = np.concatenate(([0], frequencies))

    return frequencies

def lal_binary_black_hole(
        mass_1, mass_2, luminosity_distance, a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl,
        iota, phase, ra, dec, geocent_time, psi, srate, dur_sig):
    """ A Binary Black Hole waveform model using lalsimulation

    Parameters
    ----------
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    luminosity_distance: float
        The luminosity distance in megaparsec
    a_1: float
        Dimensionless primary spin magnitude
    tilt_1: float
        Primary tilt angle
    phi_12: float

    a_2: float
        Dimensionless secondary spin magnitude
    tilt_2: float
        Secondary tilt angle
    phi_jl: float

    iota: float
        Orbital inclination
    phase: float
        The phase at coalescence
    ra: float
        The right ascension of the binary
    dec: float
        The declination of the object
    geocent_time: float
        The time at coalescence
    psi: float
        Orbital polarisation

    Returns
    -------
    The plus and cross polarisation strain modes
    """
    
    waveform_approximant = 'IMRPhenomPv2'
    reference_frequency = 50.0
    minimum_frequency = 30.0 # 20.0

    if mass_2 > mass_1:
        return None

    solar_mass = 1.98855 * 1e30
    parsec = 3.085677581 * 1e16
    luminosity_distance = luminosity_distance * 1e6 * parsec
    mass_1 = mass_1 * solar_mass
    mass_2 = mass_2 * solar_mass

    if tilt_1 == 0 and tilt_2 == 0:
        spin_1x = 0
        spin_1y = 0
        spin_1z = a_1
        spin_2x = 0
        spin_2y = 0
        spin_2z = a_2
    else:
        iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = \
            lalsim.SimInspiralTransformPrecessingNewInitialConditions(
                iota, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1, mass_2, reference_frequency, phase)

    longitude_ascending_nodes = 0.0
    eccentricity = 0.0
    mean_per_ano = 0.0

    waveform_dictionary = None

    approximant = lalsim.GetApproximantFromString(waveform_approximant)

    sampling_frequency = srate
    frequency_array = frequencies(sampling_frequency, dur_sig)
    maximum_frequency = frequency_array[-1]
    delta_frequency = frequency_array[1] - frequency_array[0]
    #print(frequency_array)

    # make the bbh signal
    hplus, hcross = lalsim.SimInspiralChooseFDWaveform(
        mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y,
        spin_2z, luminosity_distance, iota, phase,
        longitude_ascending_nodes, eccentricity, mean_per_ano, delta_frequency,
        minimum_frequency, maximum_frequency, reference_frequency,
        waveform_dictionary, approximant)
    
    h_plus = hplus.data.data
    h_cross = hcross.data.data

    h_plus = h_plus[:len(frequency_array)]
    h_cross = h_cross[:len(frequency_array)]
    
    return h_plus, h_cross, frequency_array

# This is a function to create some simulated Gaussian detector noise for a given power spectal density
def GetSimulatedData(power_spectral_density, N_fd, new_psd, df):
    """ GetSimulatedData - Function to create Gaussian noise with zero mean,
    colored by the design PSD of an Advanced detector.    
    """
  
    # Now construct Gaussian data series
    Real = np.random.normal(0,1,size=int(N_fd))*np.sqrt(new_psd/(4.*df))
    Imag = np.random.normal(0,1,size=int(N_fd))*np.sqrt(new_psd/(4.*df))
 
    # Create data series as data = real + i*imag
    detData = (Real + 1j*Imag)

    return detData

# This is the optimal SNR. Might want to replace later with matched filter SNR. 
def calculate_snr(freqsignal, PSD):
    
    # get the signal vals in our freq range
    sig = freqsignal[lo_sig:hi_sig]
    
    SNRsq = 4.*df_sig*np.sum(pow(abs(sig),2.)/ PSD)
    # Take sqrt
    SNR = np.sqrt(SNRsq)

    return SNR

# function to whiten data
def whiten(strain, psd, srate, lo, hi):
    Nt = len(strain)
    dt = 1. / srate

    # whitening: transform to freq domain, divide by asd, then transform back, 
    # taking care to get normalization right.
    hf = np.fft.rfft(strain)
    norm = 1./np.sqrt(1./(dt*2))
    white_hf = hf[lo:hi] / np.sqrt(psd) * norm
    white_ht = np.fft.irfft(white_hf, n=Nt) 
    return white_ht

# make your very own glitch   
def glitch_king(gauss_amp,gauss_sigma,Nt_noise,maxi=False):
	# set initial sigma
	gauss_length = gauss_sigma
	# randomise glitch amplitude or use maximum
	if maxi==False:
		amp = ran.uniform(-gauss_amp,gauss_amp)
	else:
		amp=gauss_amp
	
	# create initial gaussian glitch 
	gauss=(signal.gaussian(gauss_length,gauss_sigma)*amp)+1
	
	# widen the data if glitch is clipped
	while gauss[0] != 1:
		gauss_length += 100
		gauss=(signal.gaussian(gauss_length,gauss_sigma)*amp)+1
	
	#find the start and end indicies of the gaussian curve
	trigger=0
	for ex,y in enumerate(gauss,start=0):
		if y != 1 and trigger==0:
			trigger=1
			start=ex-1
		if trigger==1 and y==1:
			stop=ex
			break
	
	# find the range of the gaussian curve		
	ragnar = stop-start
	# create the glitch array
	glitch = np.ones(Nt_noise)-1
	# randomise injection point
	begin=ran.randint(0,Nt_noise-ragnar)
	# make the glitch
	glitch[begin:begin+ragnar]=gauss[start:stop]
	
	return glitch

from pycbc.filter import match as simple_match, optimized_match
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.types import TimeSeries, FrequencySeries
from Initial_Values import f_min, f_max


def perform_match(hnr:TimeSeries|FrequencySeries, hap:TimeSeries|FrequencySeries, f_lower=f_min, f_high=f_max,
                   optimized = False, return_phase = False)->tuple:
    """Function to cumpute the match of two given gravitational waves

    Args:
        hnr (TimeSeries|FrequencySeries): First Gravitational Wave 
        hap (TimeSeries|FrequencySeries): Second Gravitational Wave
        f_lower (float, optional): Low frequency cutoff. Defaults to f_min
        f_high (float, optional): High frequency cutoff. Defaults to f_max
        optimized (bool, optional): This parameter tells us to use simple_match or optimized_match. Defaults to False.
        return_phase (bool, optional): This parameter tells us to return the phase or not. Defaults to False.

    Returns:
        tuple: The match between the GWs
    """
    FD = isinstance(hnr, FrequencySeries) 
    # True if the gravitational waves are in the Frequency Domain and False if they are in the Time Domain
    if not FD: 
        hnr, hap = hnr.real(), hap.real() # The Time Domain only needs the real part of the GWs
    
    # Match the signal sizes
    length = max(len(hnr), len(hap))
    hnr.resize(length); hap.resize(length)

    # Choose the step to compute the PSD
    if FD: 
        delta_f = hnr.delta_f
    else:
        delta_f = 1/hnr.duration
        length = length//2 + 1
    
    psd = aLIGOZeroDetHighPower(length, delta_f, f_lower) # Compute PSD using the base LIGO noise at Zero Detuning and High Power
    
    # Compute Match
    match_kwargs = dict(vec1 = hnr, vec2 = hap, psd = psd, low_frequency_cutoff = f_lower, high_frequency_cutoff = f_high, return_phase = return_phase)
    return optimized_match(**match_kwargs) if optimized else simple_match(**match_kwargs, subsample_interpolation = True)
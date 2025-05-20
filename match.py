from pycbc.filter import match as simple_match, optimized_match
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.types import TimeSeries
from Initial_Values import f_min, f_max


def perform_match(h1:TimeSeries, h2:TimeSeries, f_lower=f_min, f_high=f_max,
                   optimized = False, return_phase = False)->tuple:
    """Function to cumpute the match of two given gravitational waves

    Args:
        h1 (TimeSeries|FrequencySeries): First Gravitational Wave 
        h2 (TimeSeries|FrequencySeries): Second Gravitational Wave
        f_lower (float, optional): Low frequency cutoff. Defaults to f_min
        f_high (float, optional): High frequency cutoff. Defaults to f_max
        optimized (bool, optional): This parameter tells us to use simple_match or optimized_match. Defaults to False.
        return_phase (bool, optional): This parameter tells us to return the phase or not. Defaults to False.

    Returns:
        tuple: The match between the GWs
    """
    h1, h2 = h1.real(), h2.real() # The Time Domain only needs the real part of the GWs
    
    # Match the signal sizes
    length = max(len(h1), len(h2))
    h1.resize(length); h2.resize(length) #TODO

    # Choose the step to compute the PSD
    delta_f = 1/h1.duration
    length = length//2 + 1
    
    psd = aLIGOZeroDetHighPower(length, delta_f, f_lower) # Compute PSD using the base LIGO noise at Zero Detuning and High Power
    
    # Compute Match
    match_kwargs = dict(vec1 = h1, vec2 = h2, psd = psd, low_frequency_cutoff = f_lower, high_frequency_cutoff = f_high, return_phase = return_phase)
    return optimized_match(**match_kwargs) if optimized else simple_match(**match_kwargs, subsample_interpolation = True)
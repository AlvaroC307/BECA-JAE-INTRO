from Initial_Values import Info_target, delta_T, f_min, f_ref, Spherical_Modes, n_workers
from classes import params
import lal
import lalsimulation as lalsim
import numpy as np
from math import cos, sin
from pycbc.types import TimeSeries


def Choose_modes():
    """
    Generate waveform parameters based on the chosen spherical modes.
    Returns:
        waveform_params: A dictionary containing the mode array or None if all modes are used.
    """
    if Spherical_Modes == "All":
        # Use all spherical modes
        waveform_params = None
    else:
    
        mode_list = Spherical_Modes

        # Create the waveform parameters structure
        waveform_params = lal.CreateDict()

        mode_array = lalsim.SimInspiralCreateModeArray()
        for l, m in mode_list:
            lalsim.SimInspiralModeArrayActivateMode(mode_array, l, m)
        lalsim.SimInspiralWaveformParamsInsertModeArray(waveform_params, mode_array)

    return waveform_params


def simulationTD(Approximant, parameters: params) -> tuple:
    """
    Simulate a binary system using the given approximant and parameters.
    Args:
        Approximant: The chosen approximant for the simulation.
        parameters (params): A class containing all the mandatory parameters needed to compute the gravitational wave.
    Returns:
        tuple: A tuple containing three numpy arrays: h_plus, h_cross, and the time array.
    """
    # Generate waveform parameters based on the chosen modes
    waveform_params = Choose_modes()

    with lal.no_swig_redirect_standard_output_error():
        # Generate the waveform using the given parameters
        hplus, hcross = lalsim.SimInspiralChooseTDWaveform(
            **parameters.__dict__,
            deltaT=delta_T,
            f_min=f_min,
            f_ref=f_ref,
            params=waveform_params,
            approximant=Approximant
        )

    # Extract time series for both plus (h_plus) and cross (h_cross) polarizations
    h_plus = hplus.data.data
    h_cross = hcross.data.data
    time = np.arange(len(h_plus)) * delta_T

    # Return the data as numpy arrays
    return h_plus, h_cross, time


# Compute the target gravitational wave to use it as a global variable
h_target = []  # List to store the total strain for each target gravitational wave

for Info in Info_target:
    """
    Info_target contains information about the target gravitational waves:
    - Info[0]: Approximant for the target wave
    - Info[1]: Parameters for the target wave
    - Info[2]: Polarization angle for the target wave
    """
    Approximant_target = Info[0]
    parameters_target = Info[1]
    pol_target = Info[2]

    # Simulate the target gravitational wave
    hp_target, hc_target, time_target = simulationTD(Approximant_target, parameters_target)

    # Convert the data from numpy arrays to PyCBC TimeSeries
    hp_target = TimeSeries(hp_target, delta_t=delta_T)
    hc_target = TimeSeries(hc_target, delta_t=delta_T)

    # Compute the total strain using the polarization angle
    h_target.append(hp_target * cos(2 * pol_target) + hc_target * sin(2 * pol_target))


def divide_list(data, n_workers):
    """
    Divide a list into approximately equal parts for parallel processing.
    Args:
        data: The list to be divided.
        n_workers: The number of parts to divide the list into.
    Returns:
        list: A list of sublists, each containing a portion of the original data.
    """
    k, m = divmod(len(data), n_workers)
    return [data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_workers)]


# Divide Info_target and h_target into parts for parallel processing
list_Info_target = Info_target  # Backup of the original Info_target
list_h_target = h_target        # Backup of the original h_target
Info_target = divide_list(Info_target, n_workers)
h_target = divide_list(h_target, n_workers)

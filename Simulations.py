import lal
import lalsimulation as lalsim
import numpy as np
from math import cos, sin
from pycbc.types import TimeSeries

from classes import params
import global_variables
from Initial_Values import Spherical_Modes
from Initial_Values import delta_T, f_min, f_ref, Info_target, Approximant_opt # Import the necessary parameters of the GW
# We import parameters_target so that we can compute the target GW and call it as a global function whenever we need it

def Choose_modes(): # Function to generate waveform_params using the choosen Spherical modes 

    if Spherical_Modes == "All":
        waveform_params = None
    elif Spherical_Modes == "Two": # TODO Make this in a better way 
        # Domminant Modes
        mode_list = [[2, 2], [2, -2]]

        # Create the waveform parameters structure
        waveform_params = lal.CreateDict()

        mode_array = lalsim.SimInspiralCreateModeArray()
        for l, m in mode_list:
            lalsim.SimInspiralModeArrayActivateMode(mode_array, l, m)
        lalsim.SimInspiralWaveformParamsInsertModeArray(waveform_params, mode_array)
    
    return waveform_params


def simulationTD(parameters: params)->tuple:
    """Simulation of a binary system using the approximant IMRPhenomTPHM

    Args:
        parameters (params): A class with all the mandatory parameters needed to compute the GW

    Returns:
        tuple: A tuple of three numpy arrays with all the information obtained from the simualtion, h_plus, h_cross and the list of every time
    """

    waveform_params = Choose_modes()
    
    global_variables.n_aprox_opt

    with lal.no_swig_redirect_standard_output_error():
    # Generate the waveform
        hplus, hcross = lalsim.SimInspiralChooseTDWaveform(**parameters.__dict__, 
        deltaT=delta_T, f_min=f_min, f_ref=f_ref, params=waveform_params, approximant=Approximant_opt[global_variables.n_aprox_opt])

    # Extract time series for both plus (h_plus) and cross (h_cross) polarizations and write them as numpy arrays
    h_plus = hplus.data.data
    h_cross = hcross.data.data 
    time = np.arange(len(h_plus)) * delta_T

    return h_plus, h_cross, time # The data is returned as Numpy arrays, we lost information of the epoch, etc which we won't use in this example


def simulationTD_target(Approximant, parameters: params)->tuple:
    """Simulation of a binary system using the approximant IMRPhenomTPHM

    Args:
        Approximant : The chosen approximant 
        parameters (params): A class with all the mandatory parameters needed to compute the GW

    Returns:
        tuple: A tuple of three numpy arrays with all the information obtained from the simualtion, h_plus, h_cross and the list of every time
    """

    waveform_params = Choose_modes()

    with lal.no_swig_redirect_standard_output_error():
    # Generate the waveform
        hplus, hcross = lalsim.SimInspiralChooseTDWaveform(**parameters.__dict__,
        deltaT=delta_T, f_min=f_min, f_ref=f_ref, params=waveform_params, approximant=Approximant
        )

    # Extract time series for both plus (h_plus) and cross (h_cross) polarizations and write them as numpy arrays
    h_plus = hplus.data.data
    h_cross = hcross.data.data 
    time = np.arange(len(h_plus)) * delta_T

    return h_plus, h_cross, time # The data is returned as Numpy arrays, we lost information of the epoch, etc which we won't use in this example


# Computation of the target gravitational wave to use it as a global function

h_target = [] # h_target is a list that contains all the TimeSeries for which we want to determine a Fitting Factor.
for Info in Info_target: 

    Approximant_target = Info[0]
    parameters_target = Info[1]
    pol_target = Info[2]    

    hp_target, hc_target, time_target = simulationTD_target(Approximant_target, parameters_target)    
    hp_target = TimeSeries(hp_target, delta_t=delta_T) # Convert the data obtained from a numpy array to a PyCBC TimeSeries
    hc_target = TimeSeries(hc_target, delta_t=delta_T) # Convert the data obtained from a numpy array to a PyCBC TimeSeries

    h_target.append(hp_target*cos(2*pol_target)+hc_target*sin(2*pol_target)) # We compute the total strain using the polarization of the wave


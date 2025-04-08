from classes import params
import lal
import numpy as np
import random as rnd
from math import pi, sqrt, cos, sin
import lalsimulation as lalsim
import h5py
from pycbc.types import TimeSeries

#-----------------------------DONT CHANGE----------------------------------
def M_c_and_q_m(mass_ratio, chirp_mass)->tuple:
        # Function to Calculate the original masses of the black holes given the mass ratio q_m=m1/m2 and the chirp mass

        mass2 = chirp_mass*((1+mass_ratio)/mass_ratio**3)**(1/5)
        mass1 = mass_ratio*mass2
    
        return (mass1, mass2)

def Eff_spin_and_spin1(mass1, mass2, eff_spin, spin2):
        """ Function to Calculate the third component of the spins of the second black holes 
        given the spin of the first one and the effective spin parameter """

        spin1 = (eff_spin*(mass1+mass2)-spin2*mass2)/mass1

        return spin1, spin2

def spin1p_mod_and_angle(spin1perp, angle_spin1):
    """ Function to Calculate the xy components of the large body spin given the perpendicular component and the angle"""

    spin1x = spin1perp*cos(angle_spin1)
    spin1y = spin1perp*sin(angle_spin1)

    return spin1x, spin1y
#----------------------------------DONT CHANGE----------------------------------

#-----------------------------LIST OF CHOICES OF PROGRAM OPTION----------------------------------

Target_Form = "NR_file", "Param_Space_Point", "Random_Space_Point"
Intrinsic_or_Extrinsic = "Extrinsic", "Intrinsic" 
Spherical_Modes = "All", "Two"

#-----------------------------CHOICES----------------------------------
 
Target_Form = Target_Form[1]
Intrinsic_or_Extrinsic = Intrinsic_or_Extrinsic[0]
Spherical_Modes = Spherical_Modes[0]

n_points = 5 # If Target_Form == "Random_Space_Point" this is the number of points to generate
n_workers = 5 # Number of cpus to use

#-------------------------FREQUENCY PARAMETERS OF THE SIMULATION. They are the same for every simulated GW--------------------------------------    

delta_T = 1.0/4096.0 
f_min = 15
f_max = 250
f_ref = f_min

#--------------------------APROXIMANTS USED IN THE OPTIMIZATION-------------------------------------    

Approximant_opt = ["IMRPhenomTPHM"] # Chosen Approximant (IMRPhenomTPHM, SEOBNRv4P, SpinTaylorT4)
for i in range(len(Approximant_opt)):
    Approximant_opt[i] = lalsim.GetApproximantFromString(Approximant_opt[i]) 

Approximant_target = ["IMRPhenomXO4a"]#, "IMRPhenomXPHM", "IMRPhenomXO4a"] # Chosen Approximant (IMRPhenomTPHM, SEOBNRv4P, SpinTaylorT4)
for i in range(len(Approximant_target)):
    Approximant_target[i] = lalsim.GetApproximantFromString(Approximant_target[i]) 

#-------------------------DATA OF THE TARGET GRAVITATIONAL WAVES--------------------------------------

r_target = 1e6 * lal.PC_SI # Distance to the binary system
PhiRef_target = 0 # Reference phase of the binary system

if Target_Form == "Param_Space_Point":

    Q_target = [1]
    chirp_mass_target = [40*lal.MSUN_SI]# IN SOLAR MASSES

    #mass1_target = [50,75,100]  # IN SOLAR MASSES
    #mass2_target = [50,75,100]  # IN SOLAR MASSES
    s1x_target = [0.0]
    s1y_target = [0.0]
    s1z_target = [0.0]
    s2z_target = [0.0]
    incl_target = [0] 
    LongAscNodes_target = [0]
    pol_target = np.linspace(0, pi/2, 9) # Polarization of the GW


    masses_target = []
    spin1_target = []
    spin2_target = []
    parameters_target = []

    for chirp_mass in chirp_mass_target:
        for Q in Q_target:
            masses_target.append(M_c_and_q_m(Q, chirp_mass))
                        

    """ for mass1 in mass1_target:
        for mass2 in mass2_target:
            masses_target.append((mass1*lal.MSUN_SI, mass2*lal.MSUN_SI)) # Masses of the Black Holes """

    for s1x in s1x_target:
        for s1y in s1y_target:
            for s1z in s1z_target:
                spin1_target.append((s1x, s1y, s1z)) # Spin of the first Black Hole
                                
    for s2z in s2z_target:
        spin2_target.append((0.0, 0.0, s2z)) # Spin of the second Nlack Hole
    
    for masses in masses_target:
        for spin1 in spin1_target:
            for spin2 in spin2_target:
                for incl in incl_target:
                    for longascnodes in LongAscNodes_target:
                        parameters_target.append(params(masses, spin1, spin2, r = r_target,
                                    incl = incl, phiRef = PhiRef_target, longAscNodes = longascnodes)) # Write this parameters as a params class

    Info_target = []
    for prms in parameters_target:
        for pol in pol_target:
            for App in Approximant_target:
                Info_target.append([App, prms, pol])


elif Target_Form == "Random_Space_Point":

    for k in range(n_workers):
        parameters_target = []
        pol_target = []
        for i in range(n_points):

            mass1_target = rnd.uniform(5, 150)*lal.MSUN_SI
            mass2_target = rnd.uniform(5, 150)*lal.MSUN_SI

            if mass1_target>=mass2_target: # This ensures that m1>m2. It's just a convention to make reading the results easier (Q>1)
                masses_target = (mass1_target, mass2_target) 
            elif mass2_target>mass1_target:
                masses_target = (mass2_target, mass1_target) 

            chi1_z = rnd.uniform(-1, 1)
            chi2_z = rnd.uniform(-1, 1)
            chi1_p = rnd.uniform(-sqrt(1-chi1_z**2), sqrt(1-chi1_z**2)) # The range is a function of s1z to make sure that |chi_1|<1
            chi1_p_angle = rnd.uniform(-pi, pi)

            spin1_target = (chi1_p*cos(chi1_p_angle), chi1_p*sin(chi1_p_angle), chi1_z) # Spin of the first Black Hole
            spin2_target = (0, 0, chi2_z) # Spin of the second Nlack Hole

            incl_target = rnd.uniform(0,2*pi)
            LongAscNodes_target = rnd.uniform(0, pi/2)


            parameters_target.append(params(masses_target, spin1_target, spin2_target, r = r_target,
                                            incl = incl_target, phiRef = PhiRef_target, longAscNodes = LongAscNodes_target)) # Write this parameters as a params class
            pol_target.append(rnd.uniform(0, pi/2)) # Polarization of the GW

        Info_target = []
        for i in range(n_points):
            for App in Approximant_target:
                Info_target.append([App, parameters_target[i], pol_target[i]])


elif Target_Form == "NR_file":

    NRfile = "q1a02t30_T_96_384.h5", "q2a02t30dP0dRm75_T_96_384.h5" # Filename with a NR waveform 

    NRfile = "./Input/NR_files" + NRfile[0] 

    #------------------------------------------
    mtotal = 60
    delta_F = 0.125
    #------------------------------------------

    NR_GW = h5py.File(NRfile, 'r')
    m1 = NR_GW.attrs['mass1']
    m2 = NR_GW.attrs['mass2']

    mass1_target = m1 * mtotal / (m1 + m2) * lal.MSUN_SI
    mass2_target = m2 * mtotal / (m1 + m2) * lal.MSUN_SI
    masses_target = (mass1_target, mass2_target) # Masses of the Black Holes

    spins = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(-1, 1, NRfile) 
    spin1_target = (spins[0], spins[1], spins[2]) # Spin of the first Black Hole
    spin2_target = (spins[3], spins[4], spins[5]) # Spin of the second Nlack Hole

    f_min = NR_GW.attrs['f_lower_at_1MSUN']/mtotal
    f_ref = f_min
    f_max = 2**(int(np.log(3000./delta_F)/np.log(2))+1) * delta_F

    NR_GW.close()

    #-----------------------------
    incl_target = 0  
    LongAscNodes_target = 0
    pol_target = 0
    #-------------------------

    parameters_target:params = params(masses_target, spin1_target, spin2_target, r = r_target,
                                   incl = incl_target, phiRef = PhiRef_target, longAscNodes = LongAscNodes_target) # Write this parameters as a params class

    Info_target = []
    for Approximant in Approximant_target:
        Info_target.append([Approximant, parameters_target, pol_target])


#----------------------------------DONT CHANGE----------------------------------

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


def simulationTD(Approximant, parameters: params)->tuple:
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
for Info in Info_target: # We use the global variable n_target to select the target GW we want to compute

    Approximant_target = Info[0]
    parameters_target = Info[1]
    pol_target = Info[2]    

    hp_target, hc_target, time_target = simulationTD(Approximant_target, parameters_target)    
    hp_target = TimeSeries(hp_target, delta_t=delta_T) # Convert the data obtained from a numpy array to a PyCBC TimeSeries
    hc_target = TimeSeries(hc_target, delta_t=delta_T) # Convert the data obtained from a numpy array to a PyCBC TimeSeries

    h_target.append(hp_target*cos(2*pol_target)+hc_target*sin(2*pol_target)) # We compute the total strain using the polarization of the wave


#----------------------------------Generate Waveforms and divide Info_target and h_target into n_workers parts----------------------------

def divide_list(data, n_workers):
    """Function to divide a list into n_workers parts (almost equal in size)"""
    k, m = divmod(len(data), n_workers)
    resize_data = [data[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n_workers)]
    return resize_data

list_Info_target = Info_target
list_h_target = h_target
Info_target = divide_list(Info_target, n_workers)
h_target = divide_list(h_target, n_workers)

#----------------------------------DONT CHANGE----------------------------------
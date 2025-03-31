import math

from Simulations import simulationTD, h_target
import global_variables as gl_var
from classes import params
from pycbc.types import TimeSeries
from match import perform_match
from Initial_Values import delta_T, f_min, f_max, Info_target


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

    spin1x = spin1perp*math.cos(angle_spin1)
    spin1y = spin1perp*math.sin(angle_spin1)

    return spin1x, spin1y


def opt_match_full(prms:list, grad)->float: # ----------------------TOTAL OPTIMIZATION------------------------------ 
    """ Function that we want to maximize. As nlopt minimizes the functions we instert a minus sign at the end of the computation
    Args:
        prms (list): List of parameters to optimize in this order: [Q_m=m1/m2, M_chirp, eff_spin_parameter, chi_2 (z component), 
                                                                    chi_perp, angle spin1x and spin1y, incl, LongAscNodes, pol]
        grad (_type_): We use a derivative-free method but this has to be written because of the systaxis of nlopt
    Returns:
        float: The match of the given parameters multiplied by -1 """

    if prms[0]>20 or prms[0]<1:#or (1/prms[0])>20:
        """ We make sure that the ratio of the masses is between the constraints. It's not really needed but better safe than sorry."""
        return 0

    masses = M_c_and_q_m(prms[0], prms[1]) # Obtain the masses for the given parameters
    spin1z, spin2z = Eff_spin_and_spin1(masses[0], masses[1], prms[2], prms[3]) # Obtain the spins in the direction z for the given parameters
    spin1x, spin1y = spin1p_mod_and_angle(prms[4], prms[5]) # Obtain the precession spins as xy components
    spin1 = (spin1x, spin1y, spin1z)

    # Penalization if the modulus of the spin parameters is more than one
    if (spin1[0]**2+spin1[1]**2+spin1[2]**2>1) or abs(spin2z)>1: 
        return 0 
    
    parameters = params(masses, spin1, (0, 0, spin2z), incl= prms[6], longAscNodes=prms[7]) # We write the parameters using the params class

    hp, hc, time = simulationTD(parameters) # Simulation of the GW
    hp, hc = TimeSeries(hp, delta_t = delta_T), TimeSeries(hc, delta_t = delta_T) # Writing the GW as a pycbc TimeSeries Class

    h = hp*math.cos(2*prms[8])+hc*math.sin(2*prms[8]) # We compute the total strain using the polarization of the wave

    match, _ = perform_match(h_target[gl_var.n_target], h, f_lower = f_min, f_high = f_max, optimized = False, return_phase = False)

    return -match # The minus sign is because nlopt minimizes the functions


def opt_match_first_step(prms:list, grad)->float: #--------- FIRST STEP. OPTIMIZATION OF THE COMPLETE LIST OF PARAMETERS----------- 
    """ Function that we want to maximize. As nlopt minimizes the functions we instert a minus sign at the end of the computation
    Args:
        prms (list): List of parameters to optimize in this order: [Q_m=m1/m2, M_chirp, eff_spin]
        grad (_type_): We use a derivative-free method but this has to be written because of the syntaxis of nlopt
    Returns:
        float: The match of the given parameters multiplied by -1 """

    match = - opt_match_full([prms[0], prms[1], prms[2], 0, 0, 0, 0, 0, 0], grad)

    return - match # The minus sign is already in opt_match_non_precession
    

def opt_match_second_step(prms:list, grad)->float: #--------- SECOND STEP. OPTIMIZATION OF THE COMPLETE LIST OF PARAMETERS----------- 
    """ Function that we want to maximize. As nlopt minimizes the functions we instert a minus sign at the end of the computation
    Args:
        prms (list): List of parameters to optimize in this order: [Q_m=m1/m2, M_chirp, eff_spin_parameter, chi_2 (third component)
                                                                        chi_p, inclination]
        grad (_type_): We use a derivative-free method but this has to be written because of the syntaxis of nlopt
    Returns:
        float: The match of the given parameters multiplied by -1 """

    match = -opt_match_full([prms[0], prms[1], prms[2], prms[3], prms[4], 0, prms[5], 0, 0], grad)

    return -match # The minus sign is because nlopt minimizes the functions



def opt_match_first_step_intrinsic(prms:list, grad)->float: #--------- FIRST STEP. OPTIMIZATION OF THE INTRINSIC PARAMETERS----------- 
    """ Function that we want to maximize. As nlopt minimizes the functions we instert a minus sign at the end of the computation
    Args:
        prms (list): List of parameters to optimize in this order: [Q_m=m1/m2, M_chirp, eff_spin]
        grad (_type_): We use a derivative-free method but this has to be written because of the syntaxis of nlopt
    Returns:
        float: The match of the given parameters multiplied by -1 """

    # We use the values of incl, longasconodes and polarization (extrinsic parameters) of the target to calculate the match
    incl_target = Info_target[gl_var.n_target][1].inclination
    longAscNodes_target = Info_target[gl_var.n_target][1].longAscNodes
    pol_target = Info_target[gl_var.n_target][2] 

    match = - opt_match_full([prms[0], prms[1], prms[2], 0, 0, 0, incl_target, longAscNodes_target, pol_target], grad)

    return - match # The minus sign is already in opt_match_non_precession


def opt_match_second_step_intrinic(prms:list, grad)->float: #--------- LAST STEP. OPTIMIZATION OF THE INTRINSIC PARAMETERS----------- 
    """ Function that we want to maximize. As nlopt minimizes the functions we instert a minus sign at the end of the computation
    Args:
        prms (list): List of parameters to optimize in this order: [Q_m=m1/m2, M_chirp, eff_spin_parameter, chi_2 (third component)
                                                                        chi_1(x component), chi_1(y component)]
        grad (_type_): We use a derivative-free method but this has to be written because of the syntaxis of nlopt
    Returns:
        float: The match of the given parameters multiplied by -1 """

    # We use the values of incl, longasconodes and polarization (extrinsic parameters) of the target to calculate the match
    incl_target = Info_target[gl_var.n_target][1].inclination
    longAscNodes_target = Info_target[gl_var.n_target][1].longAscNodes
    pol_target = Info_target[gl_var.n_target][2] 

    match = -opt_match_full([prms[0], prms[1], prms[2], prms[3], prms[4], prms[5], incl_target, longAscNodes_target, pol_target], grad)

    return - match # The minus sign is because nlopt minimizes the functions

import math
import nlopt
import lal
from termcolor import colored

from Initial_Values import simulationTD, Approximant_opt
import global_variables as gl_var
from classes import params
from pycbc.types import TimeSeries
from match import perform_match
from Initial_Values import delta_T, f_min, f_max, Info_target, h_target


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

    hp, hc, time = simulationTD(Approximant_opt[gl_var.n_aprox_opt],parameters) # Simulation of the GW
    hp, hc = TimeSeries(hp, delta_t = delta_T), TimeSeries(hc, delta_t = delta_T) # Writing the GW as a pycbc TimeSeries Class

    h = hp*math.cos(2*prms[8])+hc*math.sin(2*prms[8]) # We compute the total strain using the polarization of the wave

    match, _ = perform_match(h_target[gl_var.name_worker][gl_var.n_target], h, f_lower = f_min, f_high = f_max, optimized = False, return_phase = False)

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
    incl_target = Info_target[gl_var.name_worker][gl_var.n_target][1].inclination
    longAscNodes_target = Info_target[gl_var.name_worker][gl_var.n_target][1].longAscNodes
    pol_target = Info_target[gl_var.name_worker][gl_var.n_target][2] 

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
    incl_target = Info_target[gl_var.name_worker][gl_var.n_target][1].inclination
    longAscNodes_target = Info_target[gl_var.name_worker][gl_var.n_target][1].longAscNodes
    pol_target = Info_target[gl_var.name_worker][gl_var.n_target][2] 

    match = -opt_match_full([prms[0], prms[1], prms[2], prms[3], prms[4], prms[5], incl_target, longAscNodes_target, pol_target], grad)

    return - match # The minus sign is because nlopt minimizes the functions


#--------------------------------------------OPTIMIZATION FUNCTIONS FOR THE ONLY INTRINSIC PARAMETERS---------------------
def opt_first_intrinsic(prms_initial:list)->tuple:
    """ Function to Define the First Optimization
    Args:
        prms (list): List of parameters to optimize in this order: [Q_m=m1/m2, M_chirp, eff_spin]
    Returns:
        float: The match after the optimization
        list: The parameters to have the best possible match"""
    
    opt = nlopt.opt(nlopt.LN_NELDERMEAD, 3) # Define the first optimization
    opt.set_lower_bounds([1, lal.MSUN_SI, -1]) # Constraints
    opt.set_upper_bounds([20, 175 * lal.MSUN_SI, 1]) 
    opt.set_min_objective(opt_match_first_step_intrinsic) # Set Function to Optimize
    opt.set_xtol_rel(1e-2) # Tolerance used for computations
    #opt.set_ftol_rel(-1+1e-2) # Choice of Tolerance used for tests

    prms_final = opt.optimize(prms_initial) # Start The Optimization
    max_match = -opt.last_optimum_value() # Obtain the best value of the match

    print(colored(f"Number of Evaluations: {opt.get_numevals()} made by initial conditions: "
             f"Q = {prms_initial[0]}, M_chirp = {int(prms_initial[1]/lal.MSUN_SI)} solar masses, Chi_eff = {prms_initial[2]}.", "cyan"))

    return max_match, prms_final


def opt_second_intrinsic(prms_initial:list, detail:bool = True)->tuple:
    """ Function to Define the Second Optimization for Only Intrinsic Parameters
    Args:
        prms (list): List of parameters to optimize in this order: [Q_m=m1/m2, M_chirp, eff_spin, Chi_2z, Chi_p, Angle_Chip]
        detail (bool): Determines the detail of the optimization 
    Returns:
        float: The match after the optimization
        list: The parameters to have the best possible match"""
    
    opt = nlopt.opt(nlopt.LN_NELDERMEAD, 6)
    opt.set_lower_bounds([1, lal.MSUN_SI, -1, -1, 0, -math.pi]) # Constriants
    opt.set_upper_bounds([20, 175 * lal.MSUN_SI, 1, 1, 1, math.pi])
    opt.set_min_objective(opt_match_second_step_intrinic)

    if detail: # Determines the detail which we want when computing the optimization
        opt.set_xtol_rel(1e-4) # Tolerance used for THE FINAL computation
        #opt.set_ftol_rel(-1+1e-4) # Choice of Tolerance used for tests
    else:
        opt.set_xtol_rel(1e-3) # Tolerance used for computations

    prms_final = opt.optimize(prms_initial) # Start The Optimization
    max_match = -opt.last_optimum_value()

    if detail:
        print(colored(f"Number of Evaluations: {opt.get_numevals()}.", "cyan"))
    else:
        print(colored(f"Number of Evaluations: {opt.get_numevals()} made by initial conditions: "
             f"Chi_2z = {prms_initial[3]}, Chi_p = {prms_initial[4]}, Theta_precession = {str(round(prms_initial[5]/math.pi,2))}*pi.", "cyan"))

    return max_match, prms_final
#--------------------------------------------OPTIMIZATION FUNCTIONS FOR THE ONLY INTRINSIC PARAMETERS---------------------

#--------------------------------------------OPTIMIZATION FUNCTIONS FOR THE COMPLETE LIST OF PARAMETERS---------------------

def opt_first(prms_initial:list)->tuple:
    """ Function to Define the First Optimization
    Args:
        prms (list): List of parameters to optimize in this order: [Q_m=m1/m2, M_chirp, eff_spin]
    Returns:
        float: The match after the optimization
        list: The parameters to have the best possible match"""
    
    opt = nlopt.opt(nlopt.LN_NELDERMEAD, 3) # Define the first optimization
    opt.set_lower_bounds([1, lal.MSUN_SI, -1]) # Constraints
    opt.set_upper_bounds([20, 175 * lal.MSUN_SI, 1]) 
    opt.set_min_objective(opt_match_first_step) # Set Function to Optimize
    opt.set_xtol_rel(1e-2) # Tolerance used for computations
    #opt.set_ftol_rel(-1+1e-2) # Choice of Tolerance used for tests

    prms_final = opt.optimize(prms_initial) # Start The Optimization
    max_match = -opt.last_optimum_value() # Obtain the best value of the match

    print(colored(f"Number of Evaluations: {opt.get_numevals()} made by initial conditions: "
                 f"Q = {prms_initial[0]}, M_chirp = {int(prms_initial[1]/lal.MSUN_SI)} solar masses, Chi_eff = {prms_initial[2]}.", "cyan"))

    return max_match, prms_final


def opt_second_full(prms_initial:list)->tuple:
    """ Function to Define the Second Optimization for the complete set parameters
    Args:
        prms (list): List of parameters to optimize in this order: [Q_m=m1/m2, M_chirp, eff_spin, Chi_2z, Chi_p, Inclination]
    Returns:
        float: The match after the optimization
        list: The parameters to have the best possible match"""
    
    opt = nlopt.opt(nlopt.LN_NELDERMEAD, 6)
    opt.set_lower_bounds([1, lal.MSUN_SI, -1, -1, 0, 0]) # Constraints
    opt.set_upper_bounds([20, 175 * lal.MSUN_SI, 1, 1, 1, 2 * math.pi])
    opt.set_min_objective(opt_match_second_step)
    opt.set_xtol_rel(1e-3) # Tolerance used for computations
    #opt.set_ftol_rel(-1+1e-4) # Tolerance used for tests

    prms_final = opt.optimize(prms_initial) # Start The Optimization
    max_match = -opt.last_optimum_value()

    print(colored(f"Number of Evaluations: {opt.get_numevals()} made by initial conditions: "
             f"Chi_2z = {prms_initial[3]}, Chi_p = {prms_initial[4]}, incl = {str(round(prms_initial[5]/math.pi,2))}*pi.", "cyan"))

    return max_match, prms_final


def opt_third_full(prms_initial:list, detail:bool = True)->tuple:
    """ Function to Define the Third Optimization for the complete set of Parameters
    Args:
        prms (list): List of parameters to optimize in this order: [Q_m=m1/m2, M_chirp, eff_spin, Chi_2z, Chi_p, Angle_Spin1,
                                                                    Inclination, LongAscNodes, Polarization]
        detail (bool): Determines if the optimization is done in the optimization 
    Returns:
        float: The match after the optimization
        list: The parameters to have the best possible match"""
    
    opt = nlopt.opt(nlopt.LN_NELDERMEAD, 9) # Define the local optimization
    opt.set_lower_bounds([1, lal.MSUN_SI, -1, -1, 0, -math.pi, 0, 0, 0]) # Constraints
    opt.set_upper_bounds([20, 175 * lal.MSUN_SI, 1, 1, 1, math.pi, 2 * math.pi, math.pi/2, math.pi/2]) 
    opt.set_min_objective(opt_match_full)

    if detail: # Determines the detail which we want when computing the optimization
        opt.set_xtol_rel(1e-4) # Tolerance used for THE FINAL computation
        #opt.set_ftol_rel(-1+1e-4) # Choice of Tolerance used for tests
    else:
        opt.set_xtol_rel(1e-3) # Tolerance used for computations

    prms_final = opt.optimize(prms_initial) # We use the parameters obtained by the global optimization as the starting point
    max_match = -opt.last_optimum_value()

    if detail:
        print(colored(f"Number of Evaluations: {opt.get_numevals()}.", "cyan"))
    else:
        print(colored(f"Number of Evaluations: {opt.get_numevals()} made by initial conditions: "
                f"Theta_precession = {str(round(prms_initial[5]/math.pi,2))}*pi, longAscNodes = {str(round(prms_initial[7]/math.pi,2))}*pi, "
                f"polarization = {str(round(prms_initial[8]/math.pi,2))}*pi.", "cyan"))
    
    return max_match, prms_final

#--------------------------------------------OPTIMIZATION FUNCTIONS FOR THE COMPLETE LIST OF PARAMETERS---------------------



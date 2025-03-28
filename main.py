import lal
import math
import time
from pycbc.types import TimeSeries
import nlopt
import csv
from joblib import Parallel, delayed
import matplotlib.pyplot as plt 
from termcolor import colored

from Initial_Values import Intrinsic_or_Extrinsic
from Initial_Values import delta_T, incl_target, Approximant_opt, Info_target
from Simulations import simulationTD, h_target
from classes import params, chirp_mass_function
from pycbc.waveform.utils import coalign_waveforms
import global_variables as gl_var
import optimization_functions as func



def opt_first(prms_initial:list)->tuple:
    """ Function to Define the First Optimization
    Args:
        prms (list): List of parameters to optimize in this order: [Q_m=m1/m2, M_chirp, eff_spin]
    Returns:
        float: The match after the optimization
        list: The parameters to have the best possible match"""
    
    opt = nlopt.opt(nlopt.LN_NELDERMEAD, 3) # Define the first optimization
    opt.set_lower_bounds([1, lal.MSUN_SI, -1]) # Lower constraints
    opt.set_upper_bounds([20, 175 * lal.MSUN_SI, 1]) # High contraints 
    opt.set_min_objective(func.opt_match_mass_and_effspin) # Set Function to Optimization
    opt.set_xtol_rel(1e-2) # Tolerance used for computations
    #opt.set_ftol_rel(-1+1e-2) # Choice of Tolerance used for tests

    prms_final = opt.optimize(prms_initial) # Start The Optimization
    max_match = -opt.last_optimum_value() # Obtain the best value of the match

    print(colored(f"Number of Evaluations: {opt.get_numevals()} made by initial conditions: {prms_initial}", "cyan"))

    return max_match, prms_final


def opt_second_intrinsic(prms_initial:list, Detail:str = False)->tuple:
    """ Function to Define the Second Optimization for Only Intrinsic Parameters
    Args:
        prms (list): List of parameters to optimize in this order: [Q_m=m1/m2, M_chirp, eff_spin, Chi_2z, Chi_p, Angle_Chip]
    Returns:
        float: The match after the optimization
        list: The parameters to have the best possible match"""
    
    opt = nlopt.opt(nlopt.LN_NELDERMEAD, 6) # Define the local optimization
    opt.set_lower_bounds([1, lal.MSUN_SI, -1, -1, 0, -math.pi]) # Lower Constraints
    opt.set_upper_bounds([20, 175 * lal.MSUN_SI, 1, 1, 1, math.pi]) # Lower Constraints
    opt.set_min_objective(func.opt_match_precession_non_incl) # Set Function to Optimization

    if Detail: # Determines the detail which we want when computing the optimization
        opt.set_xtol_rel(1e-4) # Tolerance used for THE FINAL computation
        #opt.set_ftol_rel(-1+1e-4) # Choice of Tolerance used for tests
    else:
        opt.set_xtol_rel(1e-3) # Tolerance used for computations

    prms_final = opt.optimize(prms_initial) # Start The Optimization
    max_match = -opt.last_optimum_value() # Obtain the best value of the match

    print(colored(f"Number of Evaluations: {opt.get_numevals()} made by initial conditions: {prms_initial[4], prms_initial[5]}", "cyan"))

    return max_match, prms_final


def opt_second_full(prms_initial:list)->tuple:
    """ Function to Define the Second Optimization for the complete set Parameters
    Args:
        prms (list): List of parameters to optimize in this order: [Q_m=m1/m2, M_chirp, eff_spin, Chi_2z, Chi_p, Inclination]
    Returns:
        float: The match after the optimization
        list: The parameters to have the best possible match"""
    opt = nlopt.opt(nlopt.LN_NELDERMEAD, 6) # Define the local optimization
    opt.set_lower_bounds([1, lal.MSUN_SI, -1, -1, 0, 0]) # Same boundaries as before
    opt.set_upper_bounds([20, 175 * lal.MSUN_SI, 1, 1, 1, 2 * math.pi]) # Same boundaries as before
    opt.set_min_objective(func.opt_match_simple_precession)
    #opt.set_ftol_rel(-1+1e-4) # Tolerance used for tests
    opt.set_xtol_rel(1e-3) # Tolerance used for computations

    prms_final = opt.optimize(prms_initial) # We use the parameters obtained by the global optimization as the starting point
    max_match = -opt.last_optimum_value()

    print(colored(f"Number of Evaluations: {opt.get_numevals()} made by initial conditions: {prms_initial[3], prms_initial[4], prms_initial[5]}", "cyan"))

    return max_match, prms_final



def opt_third_full(prms_initial:list)->tuple:

    opt = nlopt.opt(nlopt.LN_NELDERMEAD, 9) # Define the local optimization
    opt.set_lower_bounds([1, lal.MSUN_SI, -1, -1, 0, -math.pi, 0, 0, 0]) # Same boundaries as before
    opt.set_upper_bounds([20, 175 * lal.MSUN_SI, 1, 1, 1, math.pi, 2 * math.pi, math.pi, math.pi]) # Same boundaries as before
    opt.set_min_objective(func.opt_match_precession)
    #opt.set_ftol_rel(-1+1e-4) # Tolerance used for tests
    opt.set_xtol_rel(1e-3)

    prms_final = opt.optimize(prms_initial) # We use the parameters obtained by the global optimization as the starting point
    max_match = -opt.last_optimum_value()

    print(colored(f"Number of Evaluations: {opt.get_numevals()} made by initial conditions: {prms_initial[5], prms_initial[7], prms_initial[8]}", "cyan"))

    return max_match, prms_final


def opt_fourth_full(prms_initial:list)->tuple:

    opt = nlopt.opt(nlopt.LN_NELDERMEAD, 9) # Define the local optimization
    opt.set_lower_bounds([1, lal.MSUN_SI, -1, -1, 0, -math.pi, 0, 0, 0]) # Same boundaries as before
    opt.set_upper_bounds([20, 175 * lal.MSUN_SI, 1, 1, 1, math.pi, 2 * math.pi, math.pi, math.pi]) # Same boundaries as before
    opt.set_min_objective(func.opt_match_precession)
    #opt.set_ftol_rel(-1+1e-4) # Tolerance used for tests
    
    opt.set_xtol_rel(1e-4) 

    prms_final = opt.optimize(prms_initial) # We use the parameters obtained by the global optimization as the starting point
    max_match = -opt.last_optimum_value()

    print(colored(f"Number of Evaluations: {opt.get_numevals()} made by initial conditions: {prms_initial[5], prms_initial[7], prms_initial[8]}", "cyan"))

    return max_match, prms_final




def main_Intrinsic(): 

    # We write the initial values for the optimization (it is not important if the first optimization is global)
    n_templates = 6

    # We write the initial values for the optimization (it is not important if the first optimization is global)
    mass1_template = [8*lal.MSUN_SI, 8*lal.MSUN_SI, 100*lal.MSUN_SI, 100*lal.MSUN_SI, 40*lal.MSUN_SI, 40*lal.MSUN_SI]
    mass2_template = [4*lal.MSUN_SI, 4*lal.MSUN_SI, 10*lal.MSUN_SI, 10*lal.MSUN_SI, 10*lal.MSUN_SI, 10*lal.MSUN_SI]
    eff_spin_template = [-0.5, 0.5, 0.7, -0.7, 0.7, -0,7]

    prms_initial = []
    for i in range(n_templates):
        mass_ratio_template = mass1_template[i]/mass2_template[i]
        chirp_mass_template = chirp_mass_function([mass1_template[i], mass2_template[i]])
        prms_initial.append([mass_ratio_template, chirp_mass_template, eff_spin_template[i]])

    print("Starting Hierarchical Optimization")
    print("First optimization: Masses and Effective Spin")
    time_initial=time.time()

    max_match = 0
    for prms in prms_initial:
        results_multiprocess = opt_first(prms)
        if results_multiprocess[0]>max_match:
            max_match = results_multiprocess[0]
            prms_final = results_multiprocess[1]
            best_prms_initial = prms

    print(f"The best initial parameters were {best_prms_initial}")
    
    print(f"The match of the first optimization is: {max_match}. It took {time.time()-time_initial} seconds.")
    print(f"The first optimization has obtain the following parameters: {prms_final[0], prms_final[1]/lal.MSUN_SI, prms_final[2]}.") # Print the parameters calculated by the global optimization

    print(f"Second Optimization: Spin2z and Relevant Precession Parameters")

    n_templates = 6
        
    spin2z_template = [0.0, -0.7, 0.7, 0.0, -0.7, 0.7]
    spin1perp_template = [0.7, 0.5, 0.5, 0.7, 0.5, 0.5]
    anglespin1_template = [-math.pi/2, -math.pi/2, -math.pi/2, math.pi/2, math.pi/2, math.pi/2]
    prms_initial = []
    for i in range(n_templates):
        prms_initial.append([prms_final[0], prms_final[1], prms_final[2], spin2z_template[i], spin1perp_template[i], anglespin1_template[i]])

    max_match = 0
    for prms in prms_initial:
        results_multiprocess = opt_second_intrinsic(prms, Detail = False)
        if results_multiprocess[0]>max_match:
            max_match = results_multiprocess[0]
            prms_final = results_multiprocess[1]
            best_prms_initial = prms

    print(f"The best initial parameters were {best_prms_initial}")
    print(f"The match of the second optimization is: {max_match}. It took {time.time()-time_initial} seconds.")


    max_match, prms_final = opt_second_intrinsic(prms_final, Detail = True) # Add detail to the optimization.





    masses_final = func.M_c_and_q_m(prms_final[0], prms_final[1])
    spin1z_final, spin2z_final = func.Eff_spin_and_spin1(masses_final[0], masses_final[1], prms_final[2], prms_final[3])
    spin1x_final, spin1y_final = func.spin1perp_and_angle(prms_final[4],prms_final[5])

    Comp_time = time.time()-time_initial

    print(colored(f"The match of the total optimization is: {max_match}. The complete optimization took {Comp_time} seconds.", "magenta"))
    
    # Waveforms Coaligned 

    optimized_params = params(masses_final, (spin1x_final,spin1y_final,spin1z_final), (0,0,spin2z_final))
    hp_test, hc_test, time_test = simulationTD(optimized_params)
    h_test = hp_test
    h1_aligned, h2_aligned = coalign_waveforms(h_target[gl_var.n_target], TimeSeries(h_test, delta_t=delta_T))

    # Plot of the coaligned waveforms
    plt.figure(figsize=(12, 5))
    plt.plot(h1_aligned.sample_times, h1_aligned, label = f'Target')
    plt.plot(h2_aligned.sample_times, h2_aligned, label = f'Template.', linestyle='dashed')
    plt.title(f'The mismatch between the gravitational waves is {1-max_match}')
    plt.xlabel('Time (s)')
    plt.ylabel('Strain')
    plt.legend()

    plt.savefig('./Graphics/main2.png', bbox_inches='tight') # Guardar la imagen como un png
    plt.show() # Mostrar la imagen


    return max_match, Comp_time, optimized_params


def main_Full(): 
                                                 

    # We write the initial values for the optimization (it is not important if the first optimization is global)
    n_workers = 6

    # We write the initial values for the optimization

    mass1_template = [8*lal.MSUN_SI, 8*lal.MSUN_SI, 100*lal.MSUN_SI, 100*lal.MSUN_SI, 40*lal.MSUN_SI, 40*lal.MSUN_SI]
    mass2_template = [4*lal.MSUN_SI, 4*lal.MSUN_SI, 10*lal.MSUN_SI, 10*lal.MSUN_SI, 10*lal.MSUN_SI, 10*lal.MSUN_SI]
    eff_spin_template = [-0.5, 0.5, 0.7, -0.7, 0.7, -0,7]


    prms_initial = []
    for i in range(n_workers):
        mass_ratio_template = mass1_template[i]/mass2_template[i]
        chirp_mass_template = chirp_mass_function([mass1_template[i], mass2_template[i]])
        prms_initial.append([mass_ratio_template, chirp_mass_template, eff_spin_template[i]])

    print("Starting Hierarchical Optimization")
    print("First optimization: Masses and Effective Spin")
    time_initial=time.time()

    max_match = 0
    for prms in prms_initial:
        results_multiprocess = opt_first(prms)
        if results_multiprocess[0]>max_match:
            max_match = results_multiprocess[0]
            prms_final = results_multiprocess[1]
            best_prms_initial = prms

    print(f"The best initial parameters were {best_prms_initial}")
    
    # max_match, prms_final = opt_first(prms_initial[3])
    
    print(f"The match of the first optimization is: {max_match}. It took {time.time()-time_initial} seconds.")
    print(f"The first optimization has obtain the following parameters: {prms_final[0], prms_final[1]/lal.MSUN_SI, prms_final[2]}.") # Print the parameters calculated by the global optimization

    print(f"Second Optimization: Spin2z, Chi_p and Inclination")

    spin2z_template = [0.0, 0.0 ,-0.4, 0.4, -0.7, 0.7]
    chi_p_template = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    incl_template = [0, math.pi/2 ,math.pi/2, math.pi/2, 3*math.pi/2, 3*math.pi/2]

    prms_initial = []
    for i in range(n_workers):
        prms_initial.append([prms_final[0], prms_final[1], prms_final[2], spin2z_template[i], chi_p_template[i], incl_template[i]])

    max_match = 0
    for prms in prms_initial:
        results_multiprocess = opt_second_full(prms)
        if results_multiprocess[0]>max_match:
            max_match = results_multiprocess[0]
            prms_final = results_multiprocess[1]
            best_prms_initial = prms

    print(f"The best initial parameters were {best_prms_initial}")


    print(f"The match of the second optimization is: {max_match}. It took {time.time()-time_initial} seconds.")
    print(f"The second optimization has obtain the following parameters: {prms_final[0], prms_final[1]/lal.MSUN_SI, prms_final[2], prms_final[3], prms_final[4]}.") # Print the parameters calculated by the global optimization
    print(f"inclination: {prms_final[4]}, incl_target: {incl_target}")

    print(f"Third Optimization: Spin1_Angle, LongAscNodes and Polarization")


    anglespin1_template = [-math.pi/2, -math.pi/2, -math.pi/2, math.pi/2, math.pi/2, math.pi/2]
    LongAscNodes_template = [0, math.pi/4, 0, math.pi/4, 0, math.pi/4]
    pol_template = [0, 0, math.pi/4, math.pi/4, math.pi/4, 0] #Polarization of the GW (Periodic [0,pi/2])

    prms_initial = []
    for i in range(n_workers):
        prms_initial.append([prms_final[0], prms_final[1], prms_final[2], prms_final[3],
                              prms_final[4], anglespin1_template[i], prms_final[5], LongAscNodes_template[i], pol_template[i]])

    max_match = 0
    for prms in prms_initial:
        results_multiprocess = opt_third_full(prms)
        if results_multiprocess[0]>max_match:
            max_match = results_multiprocess[0]
            prms_final = results_multiprocess[1]
            best_prms_initial = prms

    print(f"The best initial parameters were {best_prms_initial}")

    Comp_time = time.time()-time_initial

    print(f"The match of the total optimization is: {max_match}. The complete optimization took {Comp_time} seconds.")

    max_match, prms_final = opt_fourth_full(prms_final) # Add detail to the optimization.

    # max_match, prms_final = opt_third_precession(prms_initial[2])

    masses_final = func.M_c_and_q_m(prms_final[0], prms_final[1])
    spin1z_final, spin2z_final = func.Eff_spin_and_spin1(masses_final[0], masses_final[1], prms_final[2], prms_final[3])
    spin1x_final, spin1y_final = func.spin1perp_and_angle(prms_final[4], prms_final[5])
    pol_final = prms_final[8]

    Comp_time = time.time()-time_initial

    print(f"The match of the total optimization is: {max_match}. The complete optimization took {Comp_time} seconds.")

    # Waveforms Coaligned 

    optimized_params = params(masses_final, (spin1x_final, spin1y_final, spin1z_final), (0,0,spin2z_final),
                                                       incl = prms_final[6], longAscNodes = prms_final[7])

    hp_test, hc_test, time_test = simulationTD(optimized_params)
    hp_test, hc_test = TimeSeries(hp_test, delta_t=delta_T), TimeSeries(hc_test, delta_t=delta_T)
    h_test = hp_test*math.cos(2*pol_final)+hc_test*math.sin(2*pol_final)

    h1_aligned, h2_aligned = coalign_waveforms(h_target[gl_var.n_target], h_test)

    # Plot of the coaligned waveforms
    plt.figure(figsize=(12, 5))
    plt.plot(h1_aligned.sample_times, h1_aligned, label = f'Target')
    plt.plot(h2_aligned.sample_times, h2_aligned, label = f'Template.', linestyle='dashed')
    plt.title(f'The mismatch between the gravitational waves is {1-max_match}')
    plt.xlabel('Time (s)')
    plt.ylabel('Strain')
    plt.legend()

    plt.savefig('./Graphics/main2.png', bbox_inches='tight') # Guardar la imagen como un png
    plt.show() # Mostrar la imagen

    return max_match, Comp_time, optimized_params, pol_final


if __name__ == '__main__': #Llamar al main()


    Fitting_Factor, Comp_time, optimized_params, pol = [],[],[],[]

    if Intrinsic_or_Extrinsic == "Extrinsic":
        for i in range(len(Approximant_opt)):
            for j in range(len(Info_target)):
                results = main_Full()
                Fitting_Factor.append(1-results[0])
                Comp_time.append(results[1])
                optimized_params.append(results[2])
                pol.append(results[3])

                gl_var.n_target+=1

            gl_var.n_aprox_opt+=1
            gl_var.n_target = 0

    elif Intrinsic_or_Extrinsic == "Intrinsic":
        for i in range(len(Approximant_opt)):
            for j in range(len(Info_target)):
                results = main_Intrinsic()
                Fitting_Factor.append(1-results[0])
                Comp_time.append(results[1])
                optimized_params.append(results[2])

                gl_var.n_target+=1

            gl_var.n_aprox_opt +=1
            gl_var.n_target = 0
            



    print(colored(f"1-match = {Fitting_Factor}, Computing time = {Comp_time}", "green")) # TODO
    print(colored(f"Final Parameters: {optimized_params}", "green"))


    """ print(colored(f"Chirp_Mass: {parameters_target.chirp_mass()}, Eff_spin={parameters_target.eff_spin()}, Eff_p={parameters_target.spin1perp()}, angle={parameters_target.anglespin1()}", "red"))

    print(colored(f"Chirp_Mass: {optimized_params.chirp_mass()}, Eff_spin={optimized_params.eff_spin()}, Eff_p={optimized_params.spin1perp()}, angle={optimized_params.anglespin1()}", "red"))  

    if extrinsic == "Yes":
        print(colored(f"Pol: {pol}, Pol_target: {pol_target}", "red")) """

    """     print(f"{parameters_target.Q()}, "
            f"{parameters_target.chirp_mass()}, "
            f"{parameters_target.eff_spin()}, "
            f"{parameters_target.s2z}, "
            f"{parameters_target.s1x}, "
            f"{parameters_target.s1y}, "
            f"{parameters_target.inclination}",
            f"{parameters_target.longAscNodes}")

    print(f"IMRPhenomTPHM, {1-Fitting_Factor}, {Comp_time}, " 
            f"{prms[0]}, "
            f"{prms[1]}, "
            f"{prms[2]}, "
            f"{prms[3]}, "
            f"{prms[4]}, "
            f"{prms[5]}, "
            f"{prms[6]}, "
            f"{prms[7]}") """

    
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
from Initial_Values import delta_T, Approximant_opt, Info_target
from Simulations import simulationTD, h_target
from classes import params, chirp_mass_function
from pycbc.waveform.utils import coalign_waveforms
import global_variables as gl_var
import optimization_functions as func

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
    opt.set_min_objective(func.opt_match_first_step_intrinsic) # Set Function to Optimize
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
    opt.set_min_objective(func.opt_match_second_step_intrinic)

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
    opt.set_min_objective(func.opt_match_first_step) # Set Function to Optimize
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
    opt.set_min_objective(func.opt_match_second_step)
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
    opt.set_min_objective(func.opt_match_full)

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

#--------------------------------------------MAIN FUNCTIONS---------------------

def main_optimization_intrinsic(): 
    """Function to obtain the Fitting Factor optimizing over the intrinsic parameters
    Returns:
        max_match: Fitting Factor
        Comp_time: Time it takes for the computation to complete
        optimized_params (params): Final value of the Parameters which we optimize
    """

    n_templates = 6 # Number of Templates 

    # We write the initial values for the first optimization
    mass1_template = [8*lal.MSUN_SI, 8*lal.MSUN_SI, 100*lal.MSUN_SI, 100*lal.MSUN_SI, 40*lal.MSUN_SI, 40*lal.MSUN_SI]
    mass2_template = [4*lal.MSUN_SI, 4*lal.MSUN_SI, 10*lal.MSUN_SI, 10*lal.MSUN_SI, 10*lal.MSUN_SI, 10*lal.MSUN_SI]
    eff_spin_template = [-0.5, 0.5, 0.7, -0.7, 0.7, -0,7]

    prms_initial = []
    for i in range(n_templates):
        mass_ratio_template = mass1_template[i]/mass2_template[i]
        chirp_mass_template = chirp_mass_function([mass1_template[i], mass2_template[i]])
        prms_initial.append([mass_ratio_template, chirp_mass_template, eff_spin_template[i]]) # Create the templates

    print(colored("Starting Hierarchical Optimization", "green"))
    print(colored("First optimization: Masses and Effective Spin", "green"))
    time_initial = time.time()

    max_match = 0
    for prms in prms_initial:
        results_multiprocess = opt_first_intrinsic(prms) # Optimize the different templates
        if results_multiprocess[0]>max_match: # Keep the best value for the match
            max_match = results_multiprocess[0]
            prms_final = results_multiprocess[1]
            best_prms_initial = prms

    #print(f"The best initial parameters were {best_prms_initial}")
    print(f"The match of the first optimization is: {max_match}. It took {time.time()-time_initial} seconds.")
    print(colored("Second Optimization: Spin2z and Precession Parameters", "green"))

    # Number of templates
    n_templates = 6

    # We write the initial values for the first optimization    
    spin2z_template = [0.0, -0.7, 0.7, 0.0, -0.7, 0.7]
    spin1perp_template = [0.7, 0.5, 0.5, 0.7, 0.5, 0.5]
    anglespin1_template = [-math.pi/2, -math.pi/2, -math.pi/2, math.pi/2, math.pi/2, math.pi/2]

    prms_initial = []
    for i in range(n_templates):
        prms_initial.append([prms_final[0], prms_final[1], prms_final[2],
                              spin2z_template[i], spin1perp_template[i], anglespin1_template[i]]) # Create the templates

    max_match = 0
    for prms in prms_initial:
        results_multiprocess = opt_second_intrinsic(prms, detail=False) # Optimize the different templates
        if results_multiprocess[0]>max_match: # Keep the best value for the match
            max_match = results_multiprocess[0]
            prms_final = results_multiprocess[1]
            best_prms_initial = prms

    #print(f"The best initial parameters were {best_prms_initial}")
    print(f"The match of the second optimization is: {max_match}. It took {time.time()-time_initial} seconds.")
    print(colored("Third Optimization: Increasing Accuracy", "green"))

    max_match, prms_final = opt_second_intrinsic(prms_final, detail = True) # Add detail to the optimization.
    Comp_time = time.time()-time_initial
    print(colored(f"The match of the total optimization is: {max_match}. The complete optimization took {Comp_time} seconds.", "magenta"))



    # Obtaining the values for representing the Gravitational Wave
    masses_final = func.M_c_and_q_m(prms_final[0], prms_final[1])
    spin1z_final, spin2z_final = func.Eff_spin_and_spin1(masses_final[0], masses_final[1], prms_final[2], prms_final[3])
    spin1x_final, spin1y_final = func.spin1p_mod_and_angle(prms_final[4],prms_final[5])

    # Waveforms Coaligned 

    optimized_params = params(masses_final, (spin1x_final,spin1y_final,spin1z_final), (0,0,spin2z_final))
    hp_test, hc_test, time_test = simulationTD(optimized_params)
    h_test = hp_test # TODO
    h1_aligned, h2_aligned = coalign_waveforms(h_target[gl_var.n_target], TimeSeries(h_test, delta_t=delta_T))

    # Plot of the coaligned waveforms
    plt.figure(figsize=(12, 5))
    plt.plot(h1_aligned.sample_times, h1_aligned, label = f'Target')
    plt.plot(h2_aligned.sample_times, h2_aligned, label = f'Template.', linestyle='dashed')
    plt.title(f'The mismatch between the gravitational waves is {1-max_match}')
    plt.xlabel('Time (s)')
    plt.ylabel('Strain')
    plt.legend()
    plt.savefig('./Graphics/main_intrinsic.png', bbox_inches='tight') 
    #plt.show() 


    return max_match, Comp_time, prms_final


def main_optimization_full(): 
    """Function to obtain the Fitting Factor optimizing over the both intrinsic and extrinsic parameters
    Returns:
        max_match: Fitting Factor
        Comp_time: Time it takes for the computation to complete
        optimized_params (params): Final value of the Parameters which we optimize
        optimized_pol: Final value of the polarization which we optimize
    """                           

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
        prms_initial.append([mass_ratio_template, chirp_mass_template, eff_spin_template[i]]) # Create the templates

    print(colored("Starting Hierarchical Optimization", "green"))
    print(colored("First optimization: Masses and Effective Spin", "green"))
    time_initial=time.time()

    max_match = 0
    for prms in prms_initial:
        results_multiprocess = opt_first(prms) # Optimize the different templates
        if results_multiprocess[0]>max_match: # Keep the best value for the match
            max_match = results_multiprocess[0]
            prms_final = results_multiprocess[1]
            best_prms_initial = prms

    # print(f"The best initial parameters were {best_prms_initial}")
    print(f"The match of the first optimization is: {max_match}. It took {time.time()-time_initial} seconds.")
    print(colored("Second Optimization: Spin2z, Chi_p and Inclination", "green"))

    # We write the initial values for the optimization
    spin2z_template = [0.0, 0.0 ,-0.4, 0.4, -0.7, 0.7]
    chi_p_template = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    incl_template = [0, math.pi/2 ,math.pi/2, math.pi/2, 3*math.pi/2, 3*math.pi/2]

    prms_initial = []
    for i in range(n_workers):
        prms_initial.append([prms_final[0], prms_final[1], prms_final[2],
                              spin2z_template[i], chi_p_template[i], incl_template[i]]) # Create the templates

    max_match = 0
    for prms in prms_initial:
        results_multiprocess = opt_second_full(prms) # Optimize the different templates
        if results_multiprocess[0]>max_match: # Keep the best value for the match
            max_match = results_multiprocess[0]
            prms_final = results_multiprocess[1]
            best_prms_initial = prms

    #print(f"The best initial parameters were {best_prms_initial}")
    print(f"The match of the second optimization is: {max_match}. It took {time.time()-time_initial} seconds.")
    print(colored("Third Optimization: Theta_precession, LongAscNodes and Polarization", "green"))

    # We write the initial values for the optimization
    anglespin1_template = [-math.pi/2, -math.pi/2, -math.pi/2, math.pi/2, math.pi/2, math.pi/2]
    LongAscNodes_template = [0, math.pi/4, 0, math.pi/4, 0, math.pi/4]
    pol_template = [0, 0, math.pi/4, math.pi/4, math.pi/4, 0] 

    prms_initial = []
    for i in range(n_workers):
        prms_initial.append([prms_final[0], prms_final[1], prms_final[2],
                            prms_final[3], prms_final[4], anglespin1_template[i],
                            prms_final[5], LongAscNodes_template[i], pol_template[i]]) # Create the templates

    max_match = 0
    for prms in prms_initial:
        results_multiprocess = opt_third_full(prms, detail=False) # Optimize the different templates
        if results_multiprocess[0]>max_match: # Keep the best value for the match
            max_match = results_multiprocess[0]
            prms_final = results_multiprocess[1]
            best_prms_initial = prms

    #print(f"The best initial parameters were {best_prms_initial}")
    print(f"The match of the total optimization is: {max_match}. The complete optimization took {time.time()-time_initial} seconds.")
    print(colored("Fourth Optimization: Increasing Accuracy", "green"))

    max_match, prms_final = opt_third_full(prms_final, detail=True) # Add detail to the optimization.
    Comp_time = time.time()-time_initial
    print(colored(f"The match of the total optimization is: {max_match}. The complete optimization took {Comp_time} seconds.", "magenta"))



    # Obtaining the parameters to plot the gravitational Waveforms
    masses_final = func.M_c_and_q_m(prms_final[0], prms_final[1])
    spin1z_final, spin2z_final = func.Eff_spin_and_spin1(masses_final[0], masses_final[1], prms_final[2], prms_final[3])
    spin1x_final, spin1y_final = func.spin1p_mod_and_angle(prms_final[4], prms_final[5])

    # Waveforms Coaligned 

    optimized_params = params(masses_final, (spin1x_final, spin1y_final, spin1z_final), (0,0,spin2z_final),
                                                       incl=prms_final[6], longAscNodes=prms_final[7])
    optimized_pol = prms_final[8]

    hp_test, hc_test, time_test = simulationTD(optimized_params)
    hp_test, hc_test = TimeSeries(hp_test, delta_t=delta_T), TimeSeries(hc_test, delta_t=delta_T)
    h_test = hp_test*math.cos(2*optimized_pol)+hc_test*math.sin(2*optimized_pol)
    h1_aligned, h2_aligned = coalign_waveforms(h_target[gl_var.n_target], h_test)

    # Plot of the coaligned waveforms
    plt.figure(figsize=(12, 5))
    plt.plot(h1_aligned.sample_times, h1_aligned, label = f'Target')
    plt.plot(h2_aligned.sample_times, h2_aligned, label = f'Template.', linestyle='dashed')
    plt.title(f'The mismatch between the gravitational waves is {1-max_match}')
    plt.xlabel('Time (s)')
    plt.ylabel('Strain')
    plt.legend()
    plt.savefig('./Graphics/main_full.png', bbox_inches='tight') 
    #plt.show() 

    return max_match, Comp_time, prms_final


def main(): # Main Function. It executes main_optimization_full or main_optimization_intrinsic depending on the input of the user. 

    Fitting_Factor, Comp_time, prms_final = [],[],[]

    if Intrinsic_or_Extrinsic == "Extrinsic": # Uses the optimization of every possible parameter
        for i in range(len(Approximant_opt)): # Obtaining the result for different approximations in the optimizations
            for j in range(len(Info_target)): # Obtaining the result for different target parameters
                results = main_optimization_full()
                Fitting_Factor.append(1-results[0])
                Comp_time.append(results[1])
                prms_final.append(results[2]) 

                gl_var.n_target+=1

            gl_var.n_aprox_opt+=1
            gl_var.n_target = 0

    elif Intrinsic_or_Extrinsic == "Intrinsic": # Uses the optimization of every possible parameter
        for i in range(len(Approximant_opt)): # Obtaining the result for different approximations in the optimizations
            for j in range(len(Info_target)): # Obtaining the result for different target parameters
                results = main_optimization_intrinsic()
                Fitting_Factor.append(1-results[0])
                Comp_time.append(results[1])
                prms_final.append(results[2])

                gl_var.n_target+=1

            gl_var.n_aprox_opt +=1
            gl_var.n_target = 0

    return Fitting_Factor, Comp_time, prms_final



if __name__ == '__main__': 

    Fitting_Factor, Comp_time, prms_final = main()            

    if Intrinsic_or_Extrinsic == "Extrinsic":

        file_test = open('./Data/Testing_Full.csv', "w", newline="")
        csv_test = csv.writer(file_test)

        csv_test.writerow(["FF","T_comp","Q","M_c", "Chi_eff", "Chi_2z", "Chi_p", "theta_p", "incl", "longascnodes", "pol", 
                       "Q_0","M_c_0", "Chi_eff_0", "Chi_2z_0", "Chi_p_0", "theta_p_0", "incl_0", "longascnodes_0", "pol_0"])

        j = 0
        for Approximant_optimization in Approximant_opt:
            for Info in Info_target: # Obtaining the result for different target parameters

                csv_test.writerow([Fitting_Factor[j], Comp_time[j], prms_final[j][0], prms_final[j][1], prms_final[j][2],
                                prms_final[j][3], prms_final[j][4], prms_final[j][5], prms_final[j][6], prms_final[j][7], prms_final[j][8], Info[1].Q(),
                                Info[1].chirp_mass(), Info[1].eff_spin(), Info[1].s2z, Info[1].spin1p_mod(), Info[1].spin1p_angle(),
                                Info[1].inclination, Info[1].longAscNodes, Info[2]])
                j+=1

    elif Intrinsic_or_Extrinsic == "Intrinsic":

        file_test = open('./Data/Testing_Intrinsic.csv', "w", newline="")
        csv_test = csv.writer(file_test)

        csv_test.writerow(["FF","T_comp","Q","M_c", "Chi_eff", "Chi_2z", "Chi_p", "theta_p", 
                       "Q_0","M_c_0", "Chi_eff_0", "Chi_2z_0", "Chi_p_0", "theta_p_0"])


        j = 0
        for Approximant_optimization in Approximant_opt:
            for Info in Info_target: # Obtaining the result for different target parameters

                csv_test.writerow([Fitting_Factor[j], Comp_time[j], prms_final[j][0], prms_final[j][1], prms_final[j][2],
                                    prms_final[j][3], prms_final[j][4], prms_final[j][5], Info[1].Q(), Info[1].chirp_mass(),
                                    Info[1].eff_spin(), Info[1].s2z, Info[1].spin1p_mod(), Info[1].spin1p_angle(),])
                j+=1






    
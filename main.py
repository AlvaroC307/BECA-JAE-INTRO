import lal
import math
import time
import csv
from termcolor import colored
from joblib import Parallel, delayed


import indistinguishability as ind
from Initial_Values import Intrinsic_or_Extrinsic
from Target import Info_target
from Initial_Values import Approximant_opt, n_workers, n_points_per_worker
from classes import chirp_mass_function
import global_variables as gl_var
import optimization_functions as func


#--------------------------------------------FUNCTION TO SAVE THE DIFFERENT DATA----------------------

def Save_FF_Data(Fitting_Factor: list, Comp_time: list, prms_final: list, list_Info_target: list):
    """
    Save the data of the optimization in a CSV file. The data is saved in a folder called Data.
    Args:
        Fitting_Factor: List of fitting factors.
        Comp_time: List of computation times.
        prms_final: List of final optimized parameters.
        list_Info_target: List of target information.
    """
    if Intrinsic_or_Extrinsic == "Extrinsic":
        file_test = open('./Data/Testing_Full.csv', "a", newline="")
        csv_test = csv.writer(file_test)

        # Write the header of the CSV file
        csv_test.writerow(["1-FF", "T_comp", "Q", "M_c (solar masses)", "Chi_eff", "Chi_2z", "Chi_p", "theta_p", "incl",
                           "longascnodes", "pol", "Q_0", "M_c_0 (solar masses)", "Chi_eff_0", "Chi_2z_0", "Chi_p_0",
                           "theta_p_0", "incl_0", "longascnodes_0", "pol_0"])

        for Approximant_optimization in Approximant_opt:
            for j, Info in enumerate(list_Info_target):  # Save results for different target parameters
                csv_test.writerow([1 - Fitting_Factor[j], Comp_time[j], prms_final[j][0], prms_final[j][1] / lal.MSUN_SI,
                                   prms_final[j][2], prms_final[j][3], prms_final[j][4], prms_final[j][5],
                                   prms_final[j][6], prms_final[j][7], prms_final[j][8], Info[1].Q(),
                                   Info[1].chirp_mass() / lal.MSUN_SI, Info[1].eff_spin(), Info[1].s2z,
                                   Info[1].spin1p_mod(), Info[1].spin1p_angle(), Info[1].inclination,
                                   Info[1].longAscNodes, Info[2]])
                
    elif Intrinsic_or_Extrinsic == "Intrinsic":
        file_test = open('./Data/Testing_Intrinsic.csv', "a", newline="")
        csv_test = csv.writer(file_test)

        # Write the header of the CSV file
        csv_test.writerow(["FF", "T_comp", "Q", "M_c (solar masses)", "Chi_eff", "Chi_2z", "Chi_p", "theta_p",
                           "Q_0", "M_c_0 (solar masses)", "Chi_eff_0", "Chi_2z_0", "Chi_p_0", "theta_p_0"])

        for Approximant_optimization in Approximant_opt:
            for j, Info in enumerate(list_Info_target):  # Save results for different target parameters
                csv_test.writerow([1 - Fitting_Factor[j], Comp_time[j], prms_final[j][0], prms_final[j][1] / lal.MSUN_SI,
                                   prms_final[j][2], prms_final[j][3], prms_final[j][4], prms_final[j][5],
                                   Info[1].Q(), Info[1].chirp_mass() / lal.MSUN_SI, Info[1].eff_spin(), Info[1].s2z,
                                   Info[1].spin1p_mod(), Info[1].spin1p_angle()])

    file_test.close()  # Close the file
                

def Save_SNR_Data(Fitting_Factor: list, overlap: list, min_SNR: list, min_old_SNR: list, list_Info_target: list):
    """
    Save the data of the minimum SNR in a CSV file. The data is saved in a folder called Data.
    Args:
        Fitting_Factor: List of fitting factors.
        overlap: List of overlaps.
        min_SNR: List of minimum SNR values.
        min_old_SNR: List of old minimum SNR values.
        list_Info_target: List of target information.
    """
    file_SNR = open('./Data/Min_SNR.csv', "a", newline="")
    csv_SNR = csv.writer(file_SNR)

    # Write the header of the CSV file
    csv_SNR.writerow(["1-FF", "1-Overlap", "SNR", "old_SNR", "Q_0", "M_c_0 (solar masses)", "Chi_eff_0", "Chi_2z_0",
                      "Chi_p_0", "theta_p_0", "incl_0", "longascnodes_0", "pol_0"])

    for i, Info in enumerate(list_Info_target):
        csv_SNR.writerow([1 - Fitting_Factor[i], 1 - overlap[i], min_SNR[i], min_old_SNR[i], Info[1].Q(),
                          Info[1].chirp_mass() / lal.MSUN_SI, Info[1].eff_spin(), Info[1].s2z, Info[1].spin1p_mod(),
                          Info[1].spin1p_angle(), Info[1].inclination, Info[1].longAscNodes, Info[2]])

    file_SNR.close()  # Close the file


#--------------------------------------------MAIN FUNCTIONS---------------------

def main_optimization_intrinsic(): 
    """
    Perform optimization over intrinsic parameters to obtain the Fitting Factor.
    Returns:
        max_match: Fitting Factor.
        Comp_time: Time taken for the computation.
        prms_final: Final optimized parameters.
    """

    # --------------------INITIAL VALUES FOR THE OPTIMIZATION-------------------

    # FIRST OPTIMIZATION: Masses and Effective Spin
    mass1_template = [8*lal.MSUN_SI, 8*lal.MSUN_SI, 100*lal.MSUN_SI, 100*lal.MSUN_SI, 40*lal.MSUN_SI, 40*lal.MSUN_SI]
    mass2_template = [4*lal.MSUN_SI, 4*lal.MSUN_SI, 10*lal.MSUN_SI, 10*lal.MSUN_SI, 10*lal.MSUN_SI, 10*lal.MSUN_SI]
    eff_spin_template = [-0.5, 0.5, 0.7, -0.7, 0.7, -0.7]

    # SECOND OPTIMIZATION: Spin2z, Chi_p and Inclination
    spin2z_template = [0.0, -0.7, 0.7, 0.0, -0.7, 0.7]
    spin1perp_template = [0.7, 0.5, 0.5, 0.7, 0.5, 0.5]
    anglespin1_template = [-math.pi/2, -math.pi/2, -math.pi/2, math.pi/2, math.pi/2, math.pi/2]

    # --------------------INITIAL VALUES FOR THE OPTIMIZATION-------------------


    prms_initial = []
    for i in range(len(mass1_template)): # Create templates
        mass_ratio_template = mass1_template[i]/mass2_template[i]
        chirp_mass_template = chirp_mass_function([mass1_template[i], mass2_template[i]])
        prms_initial.append([mass_ratio_template, chirp_mass_template, eff_spin_template[i]])

    print(colored("Starting Hierarchical Optimization", "green"))
    time_initial = time.time()

    max_match = 0
    for prms in prms_initial:
        results_multiprocess = func.opt_first_intrinsic(prms) # Optimize templates
        if results_multiprocess[0]>max_match: # Keep the best match
            max_match = results_multiprocess[0]
            prms_final = results_multiprocess[1]

    #print(f"The match of the first optimization is: {max_match}. It took {time.time()-time_initial} seconds. Cpu: {gl_var.name_worker}")

    prms_initial = []
    for i in range(len(spin2z_template)): # Create templates
        prms_initial.append([prms_final[0], prms_final[1], prms_final[2],
                              spin2z_template[i], spin1perp_template[i], anglespin1_template[i]]) 

    max_match = 0
    for prms in prms_initial:
        results_multiprocess = func.opt_second_intrinsic(prms, detail=False) # Optimize templates
        if results_multiprocess[0]>max_match: # Keep the best match
            max_match = results_multiprocess[0]
            prms_final = results_multiprocess[1]

    #print(f"The match of the second optimization is: {max_match}. It took {time.time()-time_initial} seconds. Cpu: {gl_var.name_worker}")

    max_match, prms_final = func.opt_second_intrinsic(prms_final, detail = True) # Add detail to the optimization.
    Comp_time = time.time()-time_initial
    print(colored(f"The Fitting Factor of the total optimization is: {max_match}. The complete optimization took {Comp_time} seconds. Cpu: {gl_var.name_worker}", "magenta"))

    return max_match, Comp_time, prms_final


def main_optimization_full(): 
    """
    Perform optimization over intrinic and extrinsic parameters to obtain the Fitting Factor.
    Returns:
        max_match: Fitting Factor
        Comp_time: Time taken for the computation
        prms_final: Final optimized parameters
    """
                       
    #-------------------INITIAL VALUES FOR THE OPTIMIZATION-------------------

    # FIRST OPTIMIZATION: Masses and Effective Spin
    mass1_template = [8*lal.MSUN_SI, 8*lal.MSUN_SI, 100*lal.MSUN_SI, 100*lal.MSUN_SI, 40*lal.MSUN_SI,
                        40*lal.MSUN_SI, 5*lal.MSUN_SI, 5*lal.MSUN_SI, 20*lal.MSUN_SI, 20*lal.MSUN_SI,
                        40*lal.MSUN_SI, 40*lal.MSUN_SI, 100*lal.MSUN_SI, 100*lal.MSUN_SI, 80*lal.MSUN_SI,
                        80*lal.MSUN_SI, 120*lal.MSUN_SI, 120*lal.MSUN_SI, 70*lal.MSUN_SI, 70*lal.MSUN_SI]
    mass2_template = [4*lal.MSUN_SI, 4*lal.MSUN_SI, 10*lal.MSUN_SI, 10*lal.MSUN_SI, 10*lal.MSUN_SI,
                        10*lal.MSUN_SI, 5*lal.MSUN_SI, 5*lal.MSUN_SI, 20*lal.MSUN_SI, 20*lal.MSUN_SI,
                        30*lal.MSUN_SI, 30*lal.MSUN_SI, 70*lal.MSUN_SI, 70*lal.MSUN_SI, 80*lal.MSUN_SI,
                        80*lal.MSUN_SI, 15*lal.MSUN_SI, 15*lal.MSUN_SI, 50*lal.MSUN_SI, 50*lal.MSUN_SI]
    eff_spin_template = [0.5, -0.5, 0.7, -0.7, 0.7,
                        -0.7, 0.3, -0.3, 0.4, -0.4,
                        0.6, -0.6, 0.3, 0.3, 0.7,
                        -0.7, 0.4, -0.4, 0.5, -0.5]

    # SECOND OPTIMIZATION: Spin2z, Chi_p and Inclination
    spin2z_template = [0.0, 0.0 ,-0.4, 0.4, -0.7, 0.7, 0.0, 0.0 ,-0.4, 0.4, -0.7, 0.7,
                       0.0, 0.0 ,-0.4, 0.4, -0.7, 0.7, 0.2 -0.2]
    chi_p_template = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                      0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.7, 0.7]
    incl_template = [0, math.pi/2 ,math.pi/2, math.pi/2, 3*math.pi/2, 3*math.pi/2, 0, math.pi/2 ,math.pi/2, math.pi/2, 3*math.pi/2, 3*math.pi/2,
                     0, math.pi/2 ,math.pi/2, math.pi/2, 3*math.pi/2, 3*math.pi/2, math.pi/4, 3*math.pi/4]

    # THIRD OPTIMIZATION: Theta_precession, LongAscNodes and Polarization
    anglespin1_template = [-math.pi/2, -math.pi/2, -math.pi/2, math.pi/2, math.pi/2, math.pi/2,
                           -math.pi/3, -math.pi/3, -math.pi/3, math.pi/3, math.pi/3, math.pi/3,
                           -math.pi/4, -math.pi/4, -math.pi/4, math.pi/4, math.pi/4, math.pi/4, 0., 0.]
    LongAscNodes_template = [0, math.pi/4, 3*math.pi/2, 0, math.pi/4, 3*math.pi/2,
                             0, math.pi/4, 3*math.pi/2, 0, math.pi/4, 3*math.pi/2,
                             0, math.pi/4, 3*math.pi/2, 0, math.pi/4, 3*math.pi/2, math.pi, math.pi]
    pol_template = [math.pi/4, 0, math.pi/3, math.pi/4, 0, math.pi/3,
                    math.pi/4, 0, math.pi/3, math.pi/4, 0, math.pi/3,
                    math.pi/4, 0, math.pi/3, math.pi/4, 0, math.pi/3, math.pi/6, math.pi/2] 

    #-------------------INITIAL VALUES FOR THE OPTIMIZATION-------------------


    prms_initial = []
    for i in range(len(mass1_template)): # Create templates
        mass_ratio_template = mass1_template[i]/mass2_template[i]
        chirp_mass_template = chirp_mass_function([mass1_template[i], mass2_template[i]])
        prms_initial.append([mass_ratio_template, chirp_mass_template, eff_spin_template[i]])

    print(colored("Starting Hierarchical Optimization", "green"))
    time_initial=time.time()

    max_match = 0
    for prms in prms_initial:
        results_multiprocess = func.opt_first(prms) # Optimize templates
        if results_multiprocess[0]>max_match: # Keep the best match
            max_match = results_multiprocess[0]
            prms_final = results_multiprocess[1]

    #print(f"The match of the first optimization is: {max_match}. It took {time.time()-time_initial} seconds. Cpu: {gl_var.name_worker}")

    prms_initial = []
    for i in range(len(spin2z_template)): # Create templates
        prms_initial.append([prms_final[0], prms_final[1], prms_final[2],
                              spin2z_template[i], chi_p_template[i], incl_template[i]]) 

    max_match = 0
    for prms in prms_initial:
        results_multiprocess = func.opt_second_full(prms) # Optimize templates
        if results_multiprocess[0]>max_match: # Keep the best match
            max_match = results_multiprocess[0]
            prms_final = results_multiprocess[1]

    #print(f"The match of the second optimization is: {max_match}. It took {time.time()-time_initial} seconds. Cpu: {gl_var.name_worker}")

    prms_initial = []
    for i in range(len(anglespin1_template)): # Create templates
        prms_initial.append([prms_final[0], prms_final[1], prms_final[2],
                            prms_final[3], prms_final[4], anglespin1_template[i],
                            prms_final[5], LongAscNodes_template[i], pol_template[i]]) 

    max_match = 0
    for prms in prms_initial:
        results_multiprocess = func.opt_third_full(prms, detail=False) # Optimize templates
        if results_multiprocess[0]>max_match: # Keep the best match
            max_match = results_multiprocess[0]
            prms_final = results_multiprocess[1]

    #print(f"The match of the total optimization is: {max_match}. The complete optimization took {time.time()-time_initial} seconds. Cpu: {gl_var.name_worker}")

    max_match, prms_final = func.opt_third_full(prms_final, detail=True) # Add detail to the optimization.
    Comp_time = time.time()-time_initial
    print(colored(f"The Fitting Factor of the total optimization is: {max_match}. The complete optimization took {Comp_time} seconds. Cpu: {gl_var.name_worker}", "magenta"))


    return max_match, Comp_time, prms_final


def main(name_worker:int): # Main Function. It executes main_optimization_full or main_optimization_intrinsic depending on the input of the user. 
    """
    Main Function. It executes main_optimization_full or main_optimization_intrinsic depending on the input of the user.
    Args:
        name_worker: Identifier for the worker in multiprocessing.
    """

    gl_var.name_worker = name_worker # Assign a unique number to the worker for multiprocessing

    # Initialize lists to store the results of the optimization
    Fitting_Factor, Comp_time, prms_final = [], [], [] 
    target = []

    if Intrinsic_or_Extrinsic == "Extrinsic": # Perform optimization over all possible parameters
        for i in range(len(Approximant_opt)): # Iterate over different approximations
            for Info in Info_target[gl_var.name_worker]: # Iterate over different targets
                results = main_optimization_full() # Perform full optimization
                Fitting_Factor.append(results[0])
                Comp_time.append(results[1])
                prms_final.append(results[2]) 
                target.append(Info[gl_var.n_target])

                gl_var.n_target+=1 # Increment the target counter
                print(f"The {gl_var.n_target + gl_var.n_aprox_opt*n_points_per_worker} optimization has been completed. CPU: {gl_var.name_worker}.")

            gl_var.n_aprox_opt+=1 # Increment the approximation counter
            gl_var.n_target = 0 # Reset the target counter

    elif Intrinsic_or_Extrinsic == "Intrinsic": # Perform optimization over intrinsic parameters only
        for i in range(len(Approximant_opt)): # Iterate over different approximations
            for Info in Info_target[gl_var.name_worker]: # Iterate over different targets
                results = main_optimization_intrinsic() # Perform intrinsic optimization
                Fitting_Factor.append(results[0])
                Comp_time.append(results[1])
                prms_final.append(results[2])
                target.append(Info[gl_var.n_target])

                gl_var.n_target+=1 # Increment the target counter
                print(f"The {gl_var.n_target + gl_var.n_aprox_opt*n_points_per_worker} optimization has been completed. CPU: {gl_var.name_worker}.")

            gl_var.n_aprox_opt +=1 # Increment the approximation counter
            gl_var.n_target = 0 # Reset the target counter

    gl_var.n_aprox_opt-=1 # Decrement the approximation counter to avoid exceeding the number of approximations

    # Calculate the overlap of the targets using different approximants 
    overlap = ind.overlap() 

    # Calculate the minimum SNR values (old and new formulae)
    min_SNR, min_old_SNR = ind.minimun_SNR(Fitting_Factor, overlap) 

    # Save the data in a list
    SNR_Data = [overlap, min_SNR, min_old_SNR] 

    return Fitting_Factor, Comp_time, prms_final, SNR_Data, target



if __name__ == '__main__': 
    """
    Entry point of the program. Executes the optimization process using multiprocessing.
    """

    # Perform multiprocessing for optimization
    results_multiprocess = Parallel(n_jobs=n_workers)(delayed(main)(i) for i in range(n_workers)) # Multiprocessing the optimization
    print(colored("The optimization has been completed.", "red"))
    
    # Initialize lists to aggregate results from all workers
    Fitting_Factor, Comp_time, prms_final = [], [], []
    overlap, min_SNR, min_old_SNR = [], [], []
    target = []

    # Aggregate results from all workers
    for list_worker in results_multiprocess: 
        for i in range(len(list_worker[0])):
            Fitting_Factor.append(list_worker[0][i])
            Comp_time.append(list_worker[1][i])
            prms_final.append(list_worker[2][i])

            overlap.append(list_worker[3][0][i])
            min_SNR.append(list_worker[3][1][i])
            min_old_SNR.append(list_worker[3][2][i])

            target.append(list_worker[4][i])

    # Save the data in a CSV file
    Save_FF_Data(Fitting_Factor, Comp_time, prms_final, target) # Save the data in a csv file   
    Save_SNR_Data(Fitting_Factor, overlap, min_SNR, min_old_SNR, target) # Save the data in a csv file

    print(colored("The data has been saved in the folder Data.", "red"))
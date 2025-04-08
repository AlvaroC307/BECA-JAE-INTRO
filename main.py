import lal
import math
import time
from pycbc.types import TimeSeries
import csv
import matplotlib.pyplot as plt 
from termcolor import colored
from joblib import Parallel, delayed


import indistinguishability as ind
from Initial_Values import Intrinsic_or_Extrinsic
from Initial_Values import delta_T, Approximant_opt, Info_target, n_workers, list_Info_target, h_target
from Initial_Values import simulationTD
from classes import params, chirp_mass_function
from pycbc.waveform.utils import coalign_waveforms
import global_variables as gl_var
import optimization_functions as func


#--------------------------------------------MAIN FUNCTIONS---------------------

def main_optimization_intrinsic(): 
    """Function to obtain the Fitting Factor optimizing over the intrinsic parameters
    Returns:
        max_match: Fitting Factor
        Comp_time: Time it takes for the computation to complete
        optimized_params (params): Final value of the Parameters which we optimize
    """

    # --------------------INITIAL VALUES FOR THE OPTIMIZATION-------------------
    # FIRST OPTIMIZATION: Masses and Effective Spin
    mass1_template = [8*lal.MSUN_SI, 8*lal.MSUN_SI, 100*lal.MSUN_SI, 100*lal.MSUN_SI, 40*lal.MSUN_SI, 40*lal.MSUN_SI]
    mass2_template = [4*lal.MSUN_SI, 4*lal.MSUN_SI, 10*lal.MSUN_SI, 10*lal.MSUN_SI, 10*lal.MSUN_SI, 10*lal.MSUN_SI]
    eff_spin_template = [-0.5, 0.5, 0.7, -0.7, 0.7, -0,7]
    # SECOND OPTIMIZATION: Spin2z, Chi_p and Inclination
    spin2z_template = [0.0, -0.7, 0.7, 0.0, -0.7, 0.7]
    spin1perp_template = [0.7, 0.5, 0.5, 0.7, 0.5, 0.5]
    anglespin1_template = [-math.pi/2, -math.pi/2, -math.pi/2, math.pi/2, math.pi/2, math.pi/2]
    # --------------------INITIAL VALUES FOR THE OPTIMIZATION-------------------


    prms_initial = []
    for i in range(len(mass1_template)): # Creating the templates
        mass_ratio_template = mass1_template[i]/mass2_template[i]
        chirp_mass_template = chirp_mass_function([mass1_template[i], mass2_template[i]])
        prms_initial.append([mass_ratio_template, chirp_mass_template, eff_spin_template[i]]) # Create the templates

    print(colored("Starting Hierarchical Optimization", "green"))
    print(colored("First optimization: Masses and Effective Spin", "green"))
    time_initial = time.time()

    max_match = 0
    for prms in prms_initial:
        results_multiprocess = func.opt_first_intrinsic(prms) # Optimize the different templates
        if results_multiprocess[0]>max_match: # Keep the best value for the match
            max_match = results_multiprocess[0]
            prms_final = results_multiprocess[1]


    print(f"The match of the first optimization is: {max_match}. It took {time.time()-time_initial} seconds.")
    print(colored("Second Optimization: Spin2z and Precession Parameters", "green"))

    prms_initial = []
    for i in range(len(spin2z_template)): # Creating the templates
        prms_initial.append([prms_final[0], prms_final[1], prms_final[2],
                              spin2z_template[i], spin1perp_template[i], anglespin1_template[i]]) # Create the templates

    max_match = 0
    for prms in prms_initial:
        results_multiprocess = func.opt_second_intrinsic(prms, detail=False) # Optimize the different templates
        if results_multiprocess[0]>max_match: # Keep the best value for the match
            max_match = results_multiprocess[0]
            prms_final = results_multiprocess[1]


    print(f"The match of the second optimization is: {max_match}. It took {time.time()-time_initial} seconds.")
    print(colored("Third Optimization: Increasing Accuracy", "green"))

    max_match, prms_final = func.opt_second_intrinsic(prms_final, detail = True) # Add detail to the optimization.
    Comp_time = time.time()-time_initial
    print(colored(f"The match of the total optimization is: {max_match}. The complete optimization took {Comp_time} seconds.", "magenta"))


    #---------------------PLOTTING THE TARGET GW AGAINST THE OBTAINED GRAVITATIONAL WAVE--------------------

    # Obtaining the values for representing the Gravitational Wave
    masses_final = func.M_c_and_q_m(prms_final[0], prms_final[1])
    spin1z_final, spin2z_final = func.Eff_spin_and_spin1(masses_final[0], masses_final[1], prms_final[2], prms_final[3])
    spin1x_final, spin1y_final = func.spin1p_mod_and_angle(prms_final[4],prms_final[5])

    # Waveforms Coaligned 
    optimized_params = params(masses_final, (spin1x_final,spin1y_final,spin1z_final), (0,0,spin2z_final))
    hp_test, hc_test, time_test = simulationTD(Approximant_opt[gl_var.n_aprox_opt], optimized_params)
    h_test = hp_test*math.cos(Info_target[gl_var.name_worker][gl_var.n_target][2]) + hc_test*math.sin(Info_target[gl_var.name_worker][gl_var.n_target][2]) 

    h1_aligned, h2_aligned = coalign_waveforms(h_target[gl_var.name_worker][gl_var.n_target], TimeSeries(h_test, delta_t=delta_T))

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

    #---------------------PLOTTING THE TARGET GW AGAINST THE OBTAINED GRAVITATIONAL WAVE--------------------

    return max_match, Comp_time, prms_final


def main_optimization_full(): 
    """Function to obtain the Fitting Factor optimizing over the both intrinsic and extrinsic parameters
    Returns:
        max_match: Fitting Factor
        Comp_time: Time it takes for the computation to complete
        optimized_params (params): Final value of the Parameters which we optimize
        optimized_pol: Final value of the polarization which we optimize
    """                           

    #-------------------INITIAL VALUES FOR THE OPTIMIZATION-------------------
    # FIRST OPTIMIZATION: Masses and Effective Spin
    mass1_template = [8*lal.MSUN_SI, 8*lal.MSUN_SI, 100*lal.MSUN_SI, 100*lal.MSUN_SI, 40*lal.MSUN_SI, 40*lal.MSUN_SI]
    mass2_template = [4*lal.MSUN_SI, 4*lal.MSUN_SI, 10*lal.MSUN_SI, 10*lal.MSUN_SI, 10*lal.MSUN_SI, 10*lal.MSUN_SI]
    eff_spin_template = [-0.5, 0.5, 0.7, -0.7, 0.7, -0,7]
    # SECOND OPTIMIZATION: Spin2z, Chi_p and Inclination
    spin2z_template = [0.0, 0.0 ,-0.4, 0.4, -0.7, 0.7]
    chi_p_template = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    incl_template = [0, math.pi/2 ,math.pi/2, math.pi/2, 3*math.pi/2, 3*math.pi/2]
    # THIRD OPTIMIZATION: Theta_precession, LongAscNodes and Polarization
    anglespin1_template = [-math.pi/2, -math.pi/2, -math.pi/2, math.pi/2, math.pi/2, math.pi/2]
    LongAscNodes_template = [0, math.pi/4, 0, math.pi/4, 0, math.pi/4]
    pol_template = [0, 0, math.pi/4, math.pi/4, math.pi/4, 0] 
    #-------------------INITIAL VALUES FOR THE OPTIMIZATION-------------------


    prms_initial = []
    for i in range(len(mass1_template)): # Creating the templates
        mass_ratio_template = mass1_template[i]/mass2_template[i]
        chirp_mass_template = chirp_mass_function([mass1_template[i], mass2_template[i]])
        prms_initial.append([mass_ratio_template, chirp_mass_template, eff_spin_template[i]]) # Create the templates

    print(colored("Starting Hierarchical Optimization", "green"))
    print(colored("First optimization: Masses and Effective Spin", "green"))
    time_initial=time.time()

    max_match = 0
    for prms in prms_initial:
        results_multiprocess = func.opt_first(prms) # Optimize the different templates
        if results_multiprocess[0]>max_match: # Keep the best value for the match
            max_match = results_multiprocess[0]
            prms_final = results_multiprocess[1]


    print(f"The match of the first optimization is: {max_match}. It took {time.time()-time_initial} seconds.")
    print(colored("Second Optimization: Spin2z, Chi_p and Inclination", "green"))



    prms_initial = []
    for i in range(len(spin2z_template)): # Creating the templates
        prms_initial.append([prms_final[0], prms_final[1], prms_final[2],
                              spin2z_template[i], chi_p_template[i], incl_template[i]]) # Create the templates

    max_match = 0
    for prms in prms_initial:
        results_multiprocess = func.opt_second_full(prms) # Optimize the different templates
        if results_multiprocess[0]>max_match: # Keep the best value for the match
            max_match = results_multiprocess[0]
            prms_final = results_multiprocess[1]


    print(f"The match of the second optimization is: {max_match}. It took {time.time()-time_initial} seconds.")
    print(colored("Third Optimization: Theta_precession, LongAscNodes and Polarization", "green"))


    prms_initial = []
    for i in range(len(anglespin1_template)): # Creating the templates
        prms_initial.append([prms_final[0], prms_final[1], prms_final[2],
                            prms_final[3], prms_final[4], anglespin1_template[i],
                            prms_final[5], LongAscNodes_template[i], pol_template[i]]) # Create the templates

    max_match = 0
    for prms in prms_initial:
        results_multiprocess = func.opt_third_full(prms, detail=False) # Optimize the different templates
        if results_multiprocess[0]>max_match: # Keep the best value for the match
            max_match = results_multiprocess[0]
            prms_final = results_multiprocess[1]


    print(f"The match of the total optimization is: {max_match}. The complete optimization took {time.time()-time_initial} seconds.")
    print(colored("Fourth Optimization: Increasing Accuracy", "green"))

    max_match, prms_final = func.opt_third_full(prms_final, detail=True) # Add detail to the optimization.
    Comp_time = time.time()-time_initial
    print(colored(f"The match of the total optimization is: {max_match}. The complete optimization took {Comp_time} seconds.", "magenta"))


    #---------------------PLOTTING THE TARGET GW AGAINST THE OBTAINED GRAVITATIONAL WAVE--------------------

    # Obtaining the parameters to plot the gravitational Waveforms
    masses_final = func.M_c_and_q_m(prms_final[0], prms_final[1])
    spin1z_final, spin2z_final = func.Eff_spin_and_spin1(masses_final[0], masses_final[1], prms_final[2], prms_final[3])
    spin1x_final, spin1y_final = func.spin1p_mod_and_angle(prms_final[4], prms_final[5])

    # Waveforms Coaligned 
    optimized_params = params(masses_final, (spin1x_final, spin1y_final, spin1z_final), (0,0,spin2z_final),
                                                       incl=prms_final[6], longAscNodes=prms_final[7])
    optimized_pol = prms_final[8]

    hp_test, hc_test, time_test = simulationTD(Approximant_opt[gl_var.n_aprox_opt], optimized_params)
    hp_test, hc_test = TimeSeries(hp_test, delta_t=delta_T), TimeSeries(hc_test, delta_t=delta_T)
    h_test = hp_test*math.cos(2*optimized_pol)+hc_test*math.sin(2*optimized_pol)
    h1_aligned, h2_aligned = coalign_waveforms(h_target[gl_var.name_worker][gl_var.n_target], h_test)

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

    #---------------------PLOTTING THE TARGET GW AGAINST THE OBTAINED GRAVITATIONAL WAVE--------------------

    return max_match, Comp_time, prms_final


def main(name_worker:int): # Main Function. It executes main_optimization_full or main_optimization_intrinsic depending on the input of the user. 

    gl_var.name_worker = name_worker # Giving nuber to the worker of the multiprocessing

    Fitting_Factor, Comp_time, prms_final = [],[],[]

    if Intrinsic_or_Extrinsic == "Extrinsic": # Uses the optimization of every possible parameter
        for i in range(len(Approximant_opt)): # Obtaining the result for different approximations in the optimizations
            for j in range(len(Info_target[gl_var.name_worker])): # Obtaining the result for different target parameters
                results = main_optimization_full()
                Fitting_Factor.append(results[0])
                Comp_time.append(results[1])
                prms_final.append(results[2]) 

                gl_var.n_target+=1 
                print(f"The {gl_var.n_target + gl_var.n_aprox_opt*len(Info_target[gl_var.name_worker])} optimization has been completed.")

            gl_var.n_aprox_opt+=1
            gl_var.n_target = 0

    elif Intrinsic_or_Extrinsic == "Intrinsic": # Uses the optimization of every possible parameter
        for i in range(len(Approximant_opt)): # Obtaining the result for different approximations in the optimizations
            for j in range(len(Info_target[gl_var.name_worker])): # Obtaining the result for different target parameters
                results = main_optimization_intrinsic()
                Fitting_Factor.append(results[0])
                Comp_time.append(results[1])
                prms_final.append(results[2])

                gl_var.n_target+=1
                print(f"The {gl_var.n_target + gl_var.n_aprox_opt*len(Info_target[gl_var.name_worker])} optimization has been completed.")

            gl_var.n_aprox_opt +=1
            gl_var.n_target = 0
    gl_var.n_aprox_opt-=1

    return Fitting_Factor, Comp_time, prms_final



def Save_Data(Fitting_Factor:list, Comp_time:list, prms_final:list):

    if Intrinsic_or_Extrinsic == "Extrinsic":

        file_test = open('./Data/Testing_Full.csv', "a", newline="")
        csv_test = csv.writer(file_test)

        csv_test.writerow(["1-FF","T_comp","Q","M_c (solar masses)", "Chi_eff", "Chi_2z", "Chi_p", "theta_p", "incl", "longascnodes", "pol", 
                       "Q_0","M_c_0 (solar masses)", "Chi_eff_0", "Chi_2z_0", "Chi_p_0", "theta_p_0", "incl_0", "longascnodes_0", "pol_0"])

        
        for Approximant_optimization in Approximant_opt:
            for j, Info in enumerate(list_Info_target): # Obtaining the result for different target parameters

                csv_test.writerow([1-Fitting_Factor[j], Comp_time[j], prms_final[j][0], prms_final[j][1]/lal.MSUN_SI, prms_final[j][2],
                                prms_final[j][3], prms_final[j][4], prms_final[j][5], prms_final[j][6], prms_final[j][7], prms_final[j][8], Info[1].Q(),
                                Info[1].chirp_mass()/lal.MSUN_SI, Info[1].eff_spin(), Info[1].s2z, Info[1].spin1p_mod(), Info[1].spin1p_angle(),
                                Info[1].inclination, Info[1].longAscNodes, Info[2]])

    elif Intrinsic_or_Extrinsic == "Intrinsic":

        file_test = open('./Data/Testing_Intrinsic.csv', "a", newline="")
        csv_test = csv.writer(file_test)

        csv_test.writerow(["FF","T_comp","Q","M_c (solar masses)", "Chi_eff", "Chi_2z", "Chi_p", "theta_p", 
                       "Q_0","M_c_0 (solar masses)", "Chi_eff_0", "Chi_2z_0", "Chi_p_0", "theta_p_0"])


        
        for Approximant_optimization in Approximant_opt:
            for j, Info in enumerate(list_Info_target): # Obtaining the result for different target parameters

                csv_test.writerow([1-Fitting_Factor[j], Comp_time[j], prms_final[j][0], prms_final[j][1]/lal.MSUN_SI, prms_final[j][2],
                                    prms_final[j][3], prms_final[j][4], prms_final[j][5], Info[1].Q(), Info[1].chirp_mass()/lal.MSUN_SI,
                                    Info[1].eff_spin(), Info[1].s2z, Info[1].spin1p_mod(), Info[1].spin1p_angle(),])



if __name__ == '__main__': 

    results_multiprocess = Parallel(n_jobs=n_workers)(delayed(main)(i) for i in range(n_workers))

    # Write the results as a list
    Fitting_Factor = []
    Comp_time = []
    prms_final = []
    for list_worker in results_multiprocess:    
        for i in range(len(list_worker[0])):
            Fitting_Factor.append(list_worker[0][i])
            Comp_time.append(list_worker[1][i])
            prms_final.append(list_worker[2][i])

    Save_Data(Fitting_Factor, Comp_time, prms_final)    

    overlap = ind.overlap()
    min_SNR, min_old_SNR = ind.minimun_SNR(Fitting_Factor, overlap)

    file_SNR = open('./Data/Min_SNR.csv', "a", newline="")
    csv_SNR = csv.writer(file_SNR)

    csv_SNR.writerow(["1-FF","1-Overlap","SNR", "old_SNR", "Q_0","M_c_0 (solar masses)", "Chi_eff_0", "Chi_2z_0",
                       "Chi_p_0", "theta_p_0", "incl_0", "longascnodes_0", "pol_0"])
    
    for i, Info in enumerate(list_Info_target):
        csv_SNR.writerow([1-Fitting_Factor[i], 1-overlap[i], min_SNR[i], min_old_SNR[i] ,Info[1].Q(), Info[1].chirp_mass()/lal.MSUN_SI,
                        Info[1].eff_spin(), Info[1].s2z, Info[1].spin1p_mod(), Info[1].spin1p_angle(),
                        Info[1].inclination, Info[1].longAscNodes, Info[2]])



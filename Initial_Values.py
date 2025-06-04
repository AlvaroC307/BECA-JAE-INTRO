from classes import params
import lal
import numpy as np
import random as rnd
from math import pi, sqrt, cos, sin
import lalsimulation as lalsim
import h5py

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

def spinp_mod_and_angle(spin_perp, angle_spin):
    """ Function to Calculate the xy components of the large body spin given the perpendicular component and the angle"""

    spin_x = spin_perp*cos(angle_spin)
    spin_y = spin_perp*sin(angle_spin)

    return spin_x, spin_y
#----------------------------------DONT CHANGE----------------------------------

#-----------------------------LIST OF CHOICES OF PROGRAM OPTION----------------------------------

Target_Form = "NR_file", "Param_Space_Point", "Random_Space_Point"
Intrinsic_or_Extrinsic = "Extrinsic", "Intrinsic", "Non-Precessing" 
Spherical_Modes = "All", [[2, 2]]

#-----------------------------CHOICES----------------------------------
  
Target_Form = Target_Form[1]
Intrinsic_or_Extrinsic = Intrinsic_or_Extrinsic[0]
Spherical_Modes = Spherical_Modes[1]

n_points_per_worker = 1 # Number of points to simulate per worker
n_workers = 1 # Number of cpus to use

#-------------------------FREQUENCY PARAMETERS OF THE SIMULATION. They are the same for every simulated GW--------------------------------------    

delta_T = 1.0/4096.0 
f_min = 15
f_max = 250
f_ref = f_min

#--------------------------APROXIMANTS USED IN THE OPTIMIZATION-------------------------------------    

Approximant_opt = ["IMRPhenomTPHM"] # Chosen Approximant (IMRPhenomTPHM, SEOBNRv4P, SpinTaylorT4)
for i in range(len(Approximant_opt)):
    Approximant_opt[i] = lalsim.GetApproximantFromString(Approximant_opt[i]) 

Approximant_target = ["SEOBNRv4PHM"] # Chosen Approximant (IMRPhenomTPHM, IMRPhenomPv2, SEOBNRv4P, IMRPhenomXPHM), "NRSur7dq2"
for i in range(len(Approximant_target)):
    Approximant_target[i] = lalsim.GetApproximantFromString(Approximant_target[i]) 

#-------------------------DATA OF THE TARGET GRAVITATIONAL WAVES--------------------------------------

r_target = 1e6 * lal.PC_SI # Distance to the binary system
PhiRef_target = 0 # Reference phase of the binary system

if Target_Form == "Param_Space_Point":

    Q_target = [3.14]
    chirp_mass_target = [29.800*lal.MSUN_SI] # IN SOLAR MASSES


    eff_spin_BBH = 0.437

    s2z_target = [0.03]

    masses_target = M_c_and_q_m(Q_target[0], chirp_mass_target[0]) # Masses of the Black Holes
    s1z_target = [Eff_spin_and_spin1(masses_target[0], masses_target[1], eff_spin_BBH, s2z_target[0])[0]] # Spin of the first Black Hole

    #spin1xy_third_BBH = spinp_mod_and_angle(0.4478833617861279, 0.1825993585981389)

    s1x_target = [0.] # Spin of the first Black Hole
    s1y_target = [0.] # Spin of the first Black Hole
    s2x_target = [0.] # Spin of the second Black Hole
    s2y_target = [0.] # Spin of the second Black Hole

    incl_target = [0.] # Inclination of the binary system 
    LongAscNodes_target = [0.] # Longitude of the Ascending Node
    pol_target = [0.] # Polarization of the GW


    masses_target = []
    spin1_target = []
    spin2_target = []
    parameters_target = []                       

    for i in range(len(Q_target)):
        masses_target.append(M_c_and_q_m(Q_target[i], chirp_mass_target[i]))
        spin1_target.append((s1x_target[i], s1y_target[i], s1z_target[i])) # Spin of the first Black Hole
        spin2_target.append((s2x_target[i], s2y_target[i], s2z_target[i])) # Spin of the second Nlack Hole
        parameters_target.append(params(masses_target[i], spin1_target[i], spin2_target[i], r = r_target,
                                        incl = incl_target[i], phiRef = PhiRef_target, longAscNodes = LongAscNodes_target[i]))
        
    list_Info_target = []
    for i in range(len(Q_target)):
        for App in Approximant_target:
            list_Info_target.append([App, parameters_target[i], pol_target[i]])


elif Target_Form == "Random_Space_Point":

    parameters_target = []
    pol_target = []
    for i in range(n_points_per_worker*n_workers):

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
        chi2_p = rnd.uniform(-sqrt(1-chi2_z**2), sqrt(1-chi2_z**2)) # The range is a function of s1z to make sure that |chi_1|<1
        chi2_p_angle = rnd.uniform(-pi, pi)

        spin1_target = (chi1_p*cos(chi1_p_angle), chi1_p*sin(chi1_p_angle), chi1_z) # Spin of the first Black Hole
        spin2_target = (chi2_p*cos(chi1_p_angle), chi2_p*sin(chi1_p_angle), chi2_z) # Spin of the second Nlack Hole

        incl_target = rnd.uniform(0, 2*pi)
        LongAscNodes_target = rnd.uniform(0, pi/2)


        parameters_target.append(params(masses_target, spin1_target, spin2_target, r = r_target,
                                            incl = incl_target, phiRef = PhiRef_target, longAscNodes = LongAscNodes_target)) # Write this parameters as a params class
        pol_target.append(rnd.uniform(0, pi/2)) # Polarization of the GW

    list_Info_target = []
    for i in range(n_points_per_worker*n_workers):
        for App in Approximant_target:
            list_Info_target.append([App, parameters_target[i], pol_target[i]])


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

    list_Info_target = []
    for Approximant in Approximant_target:
        list_Info_target.append([Approximant, parameters_target, pol_target])

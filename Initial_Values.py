from classes import params
import lal
import numpy as np
import csv
import random as rnd
from math import pi, sqrt, cos, sin
import lalsimulation as lalsim
import h5py

def Read_Target()->list:
    # Leer el fichero de inputs y de constantes
    csv_Target = open('./Input/Target.csv', 'r')
    Reader_Target = csv.reader(csv_Target)

    # Escribir el fichero de inputs en una lista
    list_Target = []
    for row in Reader_Target:
        content = row[1]
        list_Target.append(content)
    csv_Target.close()
    
    return list_Target


#---------------------------------------------------------------
# LIST OF CHOICES OF PROGRAM OPTION

Target_Form = "NR_file", "Param_Space_Point", "Random_Space_Point"
Intrinsic_or_Extrinsic = "Extrinsic", "Intrinsic" 
Spherical_Modes = "All", "Two"

#---------------------------------------------------------------
# CHOICES

Target_Form = Target_Form[2]
Intrinsic_or_Extrinsic = Intrinsic_or_Extrinsic[0]
Spherical_Modes = Spherical_Modes[0]

n_points = 2

#---------------------------------------------------------------    
# FREQUENCY PARAMETERS OF THE SIMULATION. They are the same for every simulated GW

delta_T = 1.0/4096.0 
f_min = 15
f_max = 250
f_ref = f_min

#---------------------------------------------------------------    
# APROXIMANTS USED IN THE OPTIMIZATION
Approximant_opt = ["IMRPhenomTPHM"] # Chosen Approximant (IMRPhenomTPHM, SEOBNRv4P, SpinTaylorT4)
for i in range(len(Approximant_opt)):
    Approximant_opt[i] = lalsim.GetApproximantFromString(Approximant_opt[i]) 

Approximant_target = ["IMRPhenomTPHM"]#, "IMRPhenomXPHM", "IMRPhenomXO4a"] # Chosen Approximant (IMRPhenomTPHM, SEOBNRv4P, SpinTaylorT4)
for i in range(len(Approximant_target)):
    Approximant_target[i] = lalsim.GetApproximantFromString(Approximant_target[i]) 

#---------------------------------------------------------------
# TARGET GRAVITATIONAL WAVE

list_Target = Read_Target()

r_target = float(list_Target[8]) * lal.PC_SI # Distance to the binary system

PhiRef_target = eval(list_Target[10])

if Target_Form == "Param_Space_Point":

    mass1_target = float(list_Target[0]) * lal.MSUN_SI 
    mass2_target = float(list_Target[1]) * lal.MSUN_SI
    masses_target = (mass1_target, mass2_target) # Masses of the Black Holes

    spin1_target = (float(list_Target[2]), float(list_Target[3]), float(list_Target[4])) # Spin of the first Black Hole
    spin2_target = (float(list_Target[5]), float(list_Target[6]), float(list_Target[7])) # Spin of the second Nlack Hole

    incl_target = eval(list_Target[9])  
    LongAscNodes_target = eval(list_Target[11])

    parameters_target:params = params(masses_target, spin1_target, spin2_target, r = r_target,
                                    incl = incl_target, phiRef = PhiRef_target, longAscNodes = LongAscNodes_target) # Write this parameters as a params class

    pol_target = eval(list_Target[12]) # Polarization of the GW

elif Target_Form == "Random_Space_Point":

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

    incl_target = eval(list_Target[9])  
    LongAscNodes_target = eval(list_Target[11])

    parameters_target:params = params(masses_target, spin1_target, spin2_target, r = r_target,
                                   incl = incl_target, phiRef = PhiRef_target, longAscNodes = LongAscNodes_target) # Write this parameters as a params class

    pol_target = eval(list_Target[12]) # Polarization of the GW


Info_target = []
for Approximant in Approximant_target:
    Info_target.append([Approximant, parameters_target, pol_target])

if Target_Form == "Random_Space_Point":
    Info_target = []
    for i in range(n_points):
        Info_target.append([Approximant_target[0], parameters_target[i], pol_target[i]])
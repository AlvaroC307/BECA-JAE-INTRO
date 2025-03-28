from classes import params
import lal
import numpy as np
import csv
from math import pi
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

Target_Form = "NR_file", "Param_Space_Point"
Intrinsic_or_Extrinsic = "Extrinsic", "Intrinsic" 
Spherical_Modes = "All", "Two"

#---------------------------------------------------------------
# CHOICES

Target_Form = Target_Form[0]
Intrinsic_or_Extrinsic = Intrinsic_or_Extrinsic[1]
Spherical_Modes = Spherical_Modes[0]

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

Approximant_target = ["IMRPhenomTPHM", "IMRPhenomXPHM", "IMRPhenomXO4a"] # Chosen Approximant (IMRPhenomTPHM, SEOBNRv4P, SpinTaylorT4)
for i in range(len(Approximant_target)):
    Approximant_target[i] = lalsim.GetApproximantFromString(Approximant_target[i]) 

#---------------------------------------------------------------
# TARGET GRAVITATIONAL WAVE

list_Target = Read_Target()

r_target = float(list_Target[8]) * lal.PC_SI # Distance to the binary system

incl_target = eval(list_Target[9])
PhiRef_target = eval(list_Target[10])
LongAscNodes_target = eval(list_Target[11])

if Target_Form == "Param_Space_Point":

    mass1_target = float(list_Target[0]) * lal.MSUN_SI 
    mass2_target = float(list_Target[1]) * lal.MSUN_SI
    masses_target = (mass1_target, mass2_target) # Masses of the Black Holes

    spin1_target = (float(list_Target[2]), float(list_Target[3]), float(list_Target[4])) # Spin of the first Black Hole
    spin2_target = (float(list_Target[5]), float(list_Target[6]), float(list_Target[7])) # Spin of the second Nlack Hole

    parameters_target:params = params(masses_target, spin1_target, spin2_target, r = r_target,
                                    incl = incl_target, phiRef = PhiRef_target, longAscNodes = LongAscNodes_target) # Write this parameters as a params class

    pol_target = eval(list_Target[12]) # Polarization of the GW



elif Target_Form == "NR_file":

    NRfile = "q1a02t30_T_96_384.h5", "q2a02t30dP0dRm75_T_96_384.h5" # Filename with a NR waveform 

    NRfile = "./Mismatch_test/" + NRfile[0] # TODO Cambiar la carpeta de Mismatch_test a NRfiles

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

    parameters_target:params = params(masses_target, spin1_target, spin2_target, r = r_target,
                                   incl = incl_target, phiRef = PhiRef_target, longAscNodes = LongAscNodes_target) # Write this parameters as a params class

    pol_target = eval(list_Target[12]) # Polarization of the GW


Info_target = []
for Approximant in Approximant_target:
    Info_target.append([Approximant, parameters_target, pol_target])
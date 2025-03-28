import lalsimulation as ls
import lal
import numpy as np
import sys
import argparse
import pickle
import csv

import os
import time
import lal_mismatch as lmm


initial_computing_time = time.time()

# script arguments

# NOTE: normally, would loop over an array of the values below in order to optimise the match
masses = np.array([60.]) # Original: masses = np.array([60., 150.])
#nrfile = './q2a02t30dP0dRm75_T_96_384.h5'
nrfile = './q1a02t30_T_96_384.h5'


""" pols = np.linspace(0.,np.pi*0.25,4)
phases = np.linspace(0.,2.*np.pi,4,endpoint=False)
incs = np.array([ 0.0,  0.3, 0.6, 0.9, 1.2, 1.5 ])  """

pols = np.array([0., np.pi/4])
phases = np.array([0.])
incs = np.array([0., np.pi/4, np.pi/2])

fmin = 20.
fmax = 512.
delta_F = 0.125


# set up models

modes_NR = [[l,m] for l in range(2,5) for m in range(-l,l+1)]
modes_model = [[2,2],[2,1],[2,-1],[2,-2],[3,3],[3,2],[3,-2],[3,-3],[4,4],[4,-4]]

approx_xp = ls.SimInspiralGetApproximantFromString('IMRPhenomXPHM')
approx_st = ls.SimInspiralGetApproximantFromString('IMRPhenomXPHM')
approx_o4 = ls.SimInspiralGetApproximantFromString('IMRPhenomXO4a')

Nmodels = 3

params_xp = lal.CreateDict()
params_st = lal.CreateDict()
params_o4 = lal.CreateDict()

modearrayxp = ls.SimInspiralCreateModeArray()
modearrayst = ls.SimInspiralCreateModeArray()
modearrayo4 = ls.SimInspiralCreateModeArray()
for mode in modes_model:
    ls.SimInspiralModeArrayActivateMode(modearrayxp, mode[0], mode[1])
    ls.SimInspiralModeArrayActivateMode(modearrayst, mode[0], mode[1])
    ls.SimInspiralModeArrayActivateMode(modearrayo4, mode[0], mode[1])

ls.SimInspiralWaveformParamsInsertModeArray(params_xp, modearrayxp)
ls.SimInspiralWaveformParamsInsertModeArray(params_st, modearrayst)
ls.SimInspiralWaveformParamsInsertModeArray(params_o4, modearrayo4)

sys.stdout.write('mass\tinclination\tphase_s\tkappa_s\toptSNR\tmatch_xp\tmatch_st\tmatch_o4\n')
sys.stdout.flush()

# set phases for optimization                                                                                                                                                                               
test_phases = np.linspace(0,2*np.pi,7,endpoint=False)

# loop over polarisations, phases and inclinations:
N = Nmodels + 5
M = len(phases) + len(pols)
match_dict = {}
for Mtotal in masses:
    match_dict[Mtotal] = {}
    for inclination in incs:
        match_dict[Mtotal][inclination] = np.zeros((M, N))
        i = 0
        for phase_s in phases:
            for kappa_s in pols:
                
                # get NR signal                                                                                                                                                                                
                signal_pars, signal = lmm.GenerateNRsignal(nrfile, Mtotal, inclination, phiRef=phase_s, modes=modes_NR, delta_F=delta_F)
                
                fmin_match = max(fmin, signal_pars['f_min']*1.35)
                          
                
                # compute mismatch
                
                try:
                    template_pars_xp = lmm.GenerateWaveformPars(signal_pars, inclination, approx_xp, LALparams = params_xp)
                    output_XP = lmm.pol_phase_optimized_match(signal, template_pars_xp, ls.SimNoisePSDaLIGOZeroDetHighPower,
                                                              fmin_match, fmax, kappa_s, optimization_method='NM',
                                                              test_phi=test_phases, rotate_spins=1)
                    match_XP = output_XP[2]
                    optSNR = output_XP[1]
                except RuntimeError:
                    match_XP = -1.
                    optSNR = -1.
                    
                try:
                    template_pars_st = lmm.GenerateWaveformPars(signal_pars, inclination, approx_st, LALparams = params_st)
                    ls.SimInspiralWaveformParamsInsertPhenomXHMReleaseVersion(template_pars_st['LALparams'], 122019);
                    ls.SimInspiralWaveformParamsInsertPhenomXPrecVersion(template_pars_st['LALparams'], 320)
                    ls.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(template_pars_st['LALparams'], 2)
                    output_ST = lmm.pol_phase_optimized_match(signal, template_pars_st, ls.SimNoisePSDaLIGOZeroDetHighPower,
                                                              fmin_match, fmax, kappa_s, optimization_method='NM',
                                                              test_phi=test_phases, rotate_spins=1)
                    match_ST = output_ST[2]
                    optSNR = output_ST[1]
                except RuntimeError:
                    match_XP = -1.
                    optSNR = -1.
                    
                try:
                    template_pars_o4 = lmm.GenerateWaveformPars(signal_pars, inclination, approx_o4, LALparams = params_o4)
                    output_O4 = lmm.pol_phase_optimized_match(signal, template_pars_o4, ls.SimNoisePSDaLIGOZeroDetHighPower,
                                                              fmin_match, fmax, kappa_s, optimization_method='NM',
                                                              test_phi=test_phases, rotate_spins=1)
                    match_O4 = output_O4[2]
                    optSNR = output_O4[1]
                except RuntimeError:
                    match_O4 = -1.
                    optSNR = -1.

                match_dict[Mtotal][inclination][i] = np.array([Mtotal, inclination, phase_s, kappa_s, optSNR, match_XP, match_ST, match_O4])

                i = i+1
                
# calculate snr-weighted mismatch
snr_weighted_matches = {}
for Mtotal in masses:
    snr_weighted_matches[Mtotal] = {}
    for inclination in incs:
        snrs = match_dict[Mtotal][inclination][:,4]

        xp_matches = match_dict[Mtotal][inclination][:,5]
        st_matches = match_dict[Mtotal][inclination][:,6]
        o4_matches = match_dict[Mtotal][inclination][:,7]

        weightedmatch_xp = np.cbrt(np.sum((snrs*xp_matches)**3)/np.sum(snrs**3)) 
        weightedmatch_st = np.cbrt(np.sum((snrs*st_matches)**3)/np.sum(snrs**3)) 
        weightedmatch_o4 = np.cbrt(np.sum((snrs*o4_matches)**3)/np.sum(snrs**3))
            
        snr_weighted_matches[Mtotal][inclination] = np.array([weightedmatch_xp, weightedmatch_st, weightedmatch_o4])

# need to save the values of optSNR and the match for each of the different models
#pickle.dump(snr_weighted_matches,open('./match_data/results.pkl','wb'))

file_comparison = open("./Comparison.csv", "w", newline="")
csv_comparison = csv.writer(file_comparison)

csv_comparison.writerow(["Computing_Time", "XP", "ST", "O4", "Mtotal", "Inclination"])

for Mtotal in masses:
    for inclination in incs:
        csv_comparison.writerow([time.time()-initial_computing_time, snr_weighted_matches[Mtotal][inclination][0],
                                  snr_weighted_matches[Mtotal][inclination][1], snr_weighted_matches[Mtotal][inclination][2],
                                  Mtotal, inclination])

file_comparison.close()

print(snr_weighted_matches)
print(f"the time it took to complete the program was: {time.time()-initial_computing_time} segundos.")

import sys
import math
from pycbc.types import TimeSeries

from match import perform_match
from Simulations import h_target, simulationTD
from Initial_Values import Info_target, delta_T, f_min, f_max, Intrinsic_or_Extrinsic


def overlap():

    overlap = []
    i = 0
    for Info in Info_target:

        hp, hc, time = simulationTD(Info[1]) # Simulation of the GW
        hp, hc = TimeSeries(hp, delta_t = delta_T), TimeSeries(hc, delta_t = delta_T) # Writing the GW as a pycbc TimeSeries Class
        h = hp*math.cos(2*Info[2])+hc*math.sin(2*Info[2]) # We compute the total strain using the polarization of the wave

        match, _ = perform_match(h_target[i], h, f_lower = f_min, f_high = f_max, optimized = False, return_phase = False)
        overlap.append(match)

        i+=1

    return overlap

def minimun_SNR(FF, overlap):

    if len(FF)!=len(overlap):

        print("There is an error. The overlap and the Fitting Factor have different lengths.")
        sys.exit()

    SNR = []
    old_SNR = []

    for i in range(len(FF)):
        
        if FF[i]<overlap[i]:
            SNR.append("FF<Overlap")
            old_SNR.append(math.sqrt(n_param/(2*(1-overlap[i]))))

        else:

            if Intrinsic_or_Extrinsic == "Intrinsic":
                n_param = 6
            elif Intrinsic_or_Extrinsic == "Extrinsic":
                n_param = 9
            
            Coef = (n_param*(1-(2/(9*n_param))+1.3*math.sqrt(2/(9*n_param)))**3)/2

            SNR_sq = Coef/(FF[i]-overlap[i])
            SNR.append(math.sqrt(SNR_sq))

            old_SNR.append(math.sqrt(n_param/(2*(1-overlap[i]))))

    return SNR, old_SNR


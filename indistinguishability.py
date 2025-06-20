import sys
import math

from match import match_modes
import global_variables as gl_var
from Initial_Values import Approximant_template, mode_list_template, Intrinsic_or_Extrinsic
from Target import simulationTD_modes, modes_target, Info_target


def overlap(): 
    """
    Computes the overlap between the target gravitational wave (h_target) and 
    the simulated gravitational wave for each worker.

    Returns:
        list: A list of overlap values for each target gravitational wave.
    """
    overlap = []
    for i, Info in enumerate(Info_target[gl_var.name_worker]): 
        # Iterate over the target gravitational waves assigned to the current worker

        # Simulate the gravitational wave using the chosen approximant and parameters
        modes = simulationTD_modes(Approximant_template[gl_var.n_aprox_opt], mode_list_template, Info[1])
        match = match_modes(modes, modes_target[gl_var.name_worker][i], Info[1].params_ext(), Info[1].params_ext(), Info[2], Info[2])

        overlap.append(match)  # Append the match value to the overlap list

    return overlap


def minimun_SNR(FF, overlap):
    """
    Computes the minimum Signal-to-Noise Ratio (SNR) required for a given 
    Fitting Factor (FF) and overlap.

    Args:
        FF (list): List of Fitting Factor values.
        overlap (list): List of overlap values.

    Returns:
        tuple: A tuple containing two lists:
            - SNR: The computed SNR values.
            - old_SNR: The old SNR values based on a simpler formula.
    """
    # Check if the lengths of FF and overlap are consistent
    if len(FF) != len(overlap):
        print("There is an error. The overlap and the Fitting Factor have different lengths.")
        sys.exit()

    SNR = []  # List to store the computed SNR values
    old_SNR = []  # List to store the old SNR values

    # Determine the number of parameters based on whether the analysis is intrinsic or extrinsic
    if Intrinsic_or_Extrinsic == "Intrinsic":
        n_param = 6
    elif Intrinsic_or_Extrinsic == "Extrinsic":
        n_param = 9
    elif Intrinsic_or_Extrinsic == "Non-Precessing":
        n_param = 4

    for i in range(len(FF)):
        # If the Fitting Factor is less than the overlap, don't use the new formula
        if FF[i] < overlap[i]:
            SNR.append("FF<Overlap")  # Indicate that FF is less than overlap
        else:
            # Compute the SNR
            Coef = (n_param * (1 - (2 / (9 * n_param)) + 1.3 * math.sqrt(2 / (9 * n_param)))**3) / 2
            SNR_sq = Coef / (FF[i] - overlap[i])  # Compute the square of the SNR
            SNR.append(math.sqrt(SNR_sq))  # Append the computed SNR value

        # Compute the old SNR value for comparison
        old_SNR.append(math.sqrt(n_param / (2 * (1 - overlap[i]))))

    return SNR, old_SNR
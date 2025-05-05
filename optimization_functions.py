import math
import nlopt
import lal
from pycbc.types import TimeSeries

import global_variables as gl_var
from classes import params
from match import perform_match
from Initial_Values import M_c_and_q_m, Eff_spin_and_spin1, spin1p_mod_and_angle
from Initial_Values import delta_T, f_min, f_max, Approximant_opt
from Target import simulationTD, h_target, Info_target

#---------------------------------------------- Functions for optimization ----------------------------------#

def opt_match_full(prms: list, grad) -> float:
    """
    Total optimization function to maximize the match between the target and simulated gravitational waves.
    Args:
        prms (list): List of parameters to optimize in this order:
                     [Q_m=m1/m2, M_chirp, eff_spin_parameter, chi_2 (z component),
                      chi_perp, angle spin1x and spin1y, incl, LongAscNodes, pol]
        grad: Placeholder for gradient (not used as derivative-free methods are applied).
    Returns:
        float: Negative match value (as nlopt minimizes functions).
    """
    # Compute masses and spins based on input parameters
    masses = M_c_and_q_m(prms[0], prms[1])
    spin1z, spin2z = Eff_spin_and_spin1(masses[0], masses[1], prms[2], prms[3])
    spin1x, spin1y = spin1p_mod_and_angle(prms[4], prms[5])
    spin1 = (spin1x, spin1y, spin1z)

    # Penalize if spin magnitudes exceed physical limits
    if (spin1[0]**2 + spin1[1]**2 + spin1[2]**2 > 1) or abs(spin2z) > 1:
        return 0

    # Create parameter object and simulate gravitational wave
    parameters = params(masses, spin1, (0, 0, spin2z), incl=prms[6], longAscNodes=prms[7])
    hp, hc, time = simulationTD(Approximant_opt[gl_var.n_aprox_opt], parameters)
    hp, hc = TimeSeries(hp, delta_t=delta_T), TimeSeries(hc, delta_t=delta_T)

    # Compute total strain using polarization
    h = hp * math.cos(2 * prms[8]) + hc * math.sin(2 * prms[8])

    # Compute match between target and simulated wave
    match, _ = perform_match(h_target[gl_var.name_worker][gl_var.n_target], h,
                             f_lower=f_min, f_high=f_max, optimized=False, return_phase=False)

    return -match  # Negative sign because nlopt minimizes


def opt_match_first_step(prms: list, grad) -> float:
    """
    First step optimization for a subset of parameters.
    Args:
        prms (list): Parameters to optimize: [Q_m=m1/m2, M_chirp, eff_spin].
        grad: Placeholder for gradient (not used).
    Returns:
        float: Negative match value.
    """
    return opt_match_full([prms[0], prms[1], prms[2], 0, 0, 0, 0, 0, 0], grad)


def opt_match_second_step(prms: list, grad) -> float:
    """
    Second step optimization for a larger subset of parameters.
    Args:
        prms (list): Parameters to optimize: [Q_m=m1/m2, M_chirp, eff_spin, chi_2, chi_p, inclination].
        grad: Placeholder for gradient (not used).
    Returns:
        float: Negative match value.
    """
    return opt_match_full([prms[0], prms[1], prms[2], prms[3], prms[4], 0, prms[5], 0, 0], grad)


def opt_match_first_step_intrinsic(prms: list, grad) -> float:
    """
    First step optimization for intrinsic parameters only.
    Args:
        prms (list): Parameters to optimize: [Q_m=m1/m2, M_chirp, eff_spin].
        grad: Placeholder for gradient (not used).
    Returns:
        float: Negative match value.
    """
    # Use target extrinsic parameters for optimization
    incl_target = Info_target[gl_var.name_worker][gl_var.n_target][1].inclination
    longAscNodes_target = Info_target[gl_var.name_worker][gl_var.n_target][1].longAscNodes
    pol_target = Info_target[gl_var.name_worker][gl_var.n_target][2]

    return opt_match_full([prms[0], prms[1], prms[2], 0, 0, 0, incl_target, longAscNodes_target, pol_target], grad)


def opt_match_second_step_intrinsic(prms: list, grad) -> float:
    """
    Second step optimization for intrinsic parameters only.
    Args:
        prms (list): Parameters to optimize: [Q_m=m1/m2, M_chirp, eff_spin, chi_2, chi_1x, chi_1y].
        grad: Placeholder for gradient (not used).
    Returns:
        float: Negative match value.
    """
    # Use target extrinsic parameters for optimization
    incl_target = Info_target[gl_var.name_worker][gl_var.n_target][1].inclination
    longAscNodes_target = Info_target[gl_var.name_worker][gl_var.n_target][1].longAscNodes
    pol_target = Info_target[gl_var.name_worker][gl_var.n_target][2]

    return opt_match_full([prms[0], prms[1], prms[2], prms[3], prms[4], prms[5], incl_target, longAscNodes_target, pol_target], grad)


#---------------------------------------------- Functions for optimization ----------------------------------#
#---------------------------------------------- Setting the Optimizers (only intrinsic)----------------------#


def opt_first_intrinsic(prms_initial: list) -> tuple:
    """
    Perform the first optimization for intrinsic parameters.
    Args:
        prms_initial (list): Initial guess for parameters: [Q_m, M_chirp, eff_spin].
    Returns:
        tuple: Maximum match and optimized parameters.
    """
    opt = nlopt.opt(nlopt.LN_NELDERMEAD, 3)
    opt.set_lower_bounds([1, lal.MSUN_SI, -1])
    opt.set_upper_bounds([20, 175 * lal.MSUN_SI, 1])
    opt.set_min_objective(opt_match_first_step_intrinsic)

    # Compute overlap for tolerance adjustment
    Info = Info_target[gl_var.name_worker][gl_var.n_target]
    hp, hc, time = simulationTD(Approximant_opt[gl_var.n_aprox_opt], Info[1])
    hp, hc = TimeSeries(hp, delta_t=delta_T), TimeSeries(hc, delta_t=delta_T)
    h = hp * math.cos(2 * Info[2]) + hc * math.sin(2 * Info[2])
    overlap, _ = perform_match(h_target[gl_var.name_worker][gl_var.n_target], h,
                               f_lower=f_min, f_high=f_max, optimized=False, return_phase=False)

    # Adjust tolerance based on overlap
    if (1 - overlap) < 0.02:
        opt.set_ftol_abs(-overlap)
    else:
        opt.set_xtol_rel(1e-3)

    prms_final = opt.optimize(prms_initial)
    max_match = -opt.last_optimum_value()

    return max_match, prms_final


def opt_second_intrinsic(prms_initial: list, detail: bool = True) -> tuple:
    """
    Perform the second optimization for intrinsic parameters.
    Args:
        prms_initial (list): Initial guess for parameters: [Q_m, M_chirp, eff_spin, chi_2, chi_p, angle_chip].
        detail (bool): Whether to use detailed optimization (higher precision).
    Returns:
        tuple: Maximum match and optimized parameters.
    """
    opt = nlopt.opt(nlopt.LN_NELDERMEAD, 6)
    opt.set_lower_bounds([1, lal.MSUN_SI, -1, -1, 0, -math.pi])
    opt.set_upper_bounds([20, 175 * lal.MSUN_SI, 1, 1, 1, math.pi])
    opt.set_min_objective(opt_match_second_step_intrinsic)

    opt.set_xtol_rel(1e-5 if detail else 1e-4)

    prms_final = opt.optimize(prms_initial)
    max_match = -opt.last_optimum_value()

    return max_match, prms_final

#---------------------------------------------- Setting the Optimizers (only intrinsic)----------------------#
#---------------------------------------------- Setting the Optimizers (full parameters)---------------------#

def opt_first(prms_initial: list) -> tuple:
    """
    Perform the first optimization for all parameters.
    Args:
        prms_initial (list): Initial guess for parameters: [Q_m, M_chirp, eff_spin].
    Returns:
        tuple: Maximum match and optimized parameters.
    """
    opt = nlopt.opt(nlopt.LN_NELDERMEAD, 3)
    opt.set_lower_bounds([1, lal.MSUN_SI, -1])
    opt.set_upper_bounds([20, 175 * lal.MSUN_SI, 1])
    opt.set_min_objective(opt_match_first_step)

    # Compute overlap for tolerance adjustment
    Info = Info_target[gl_var.name_worker][gl_var.n_target]
    hp, hc, time = simulationTD(Approximant_opt[gl_var.n_aprox_opt], Info[1])
    hp, hc = TimeSeries(hp, delta_t=delta_T), TimeSeries(hc, delta_t=delta_T)
    h = hp * math.cos(2 * Info[2]) + hc * math.sin(2 * Info[2])
    overlap, _ = perform_match(h_target[gl_var.name_worker][gl_var.n_target], h,
                               f_lower=f_min, f_high=f_max, optimized=False, return_phase=False)

    # Adjust tolerance based on overlap
    if (1 - overlap) < 0.02:
        opt.set_ftol_abs(-overlap)
    else:
        opt.set_xtol_rel(1e-3)

    prms_final = opt.optimize(prms_initial)
    max_match = -opt.last_optimum_value()

    return max_match, prms_final


def opt_second_full(prms_initial: list) -> tuple:
    """
    Perform the second optimization for all parameters.
    Args:
        prms_initial (list): Initial guess for parameters: [Q_m, M_chirp, eff_spin, chi_2, chi_p, inclination].
    Returns:
        tuple: Maximum match and optimized parameters.
    """
    opt = nlopt.opt(nlopt.LN_NELDERMEAD, 6)
    opt.set_lower_bounds([1, lal.MSUN_SI, -1, -1, 0, 0])
    opt.set_upper_bounds([20, 175 * lal.MSUN_SI, 1, 1, 1, 2 * math.pi])
    opt.set_min_objective(opt_match_second_step)
    opt.set_xtol_rel(1e-3)

    prms_final = opt.optimize(prms_initial)
    max_match = -opt.last_optimum_value()

    return max_match, prms_final


def opt_third_full(prms_initial: list, detail: bool = True) -> tuple:
    """
    Perform the third optimization for all parameters.
    Args:
        prms_initial (list): Initial guess for parameters: [Q_m, M_chirp, eff_spin, chi_2, chi_p, angle_spin1,
                                                            inclination, longAscNodes, polarization].
        detail (bool): Whether to use detailed optimization (higher precision).
    Returns:
        tuple: Maximum match and optimized parameters.
    """
    opt = nlopt.opt(nlopt.LN_NELDERMEAD, 9)
    opt.set_lower_bounds([1, lal.MSUN_SI, -1, -1, 0, -math.pi, 0, 0, 0])
    opt.set_upper_bounds([20, 175 * lal.MSUN_SI, 1, 1, 1, math.pi, 2 * math.pi, math.pi / 2, math.pi / 2])
    opt.set_min_objective(opt_match_full)

    opt.set_xtol_rel(1e-5 if detail else 1e-4)

    prms_final = opt.optimize(prms_initial)
    max_match = -opt.last_optimum_value()

    return max_match, prms_final

#---------------------------------------------- Setting the Optimizers (full parameters)---------------------#

from Initial_Values import list_Info_target, delta_T, f_min, f_ref, n_workers, pad, mode_list_target
from classes import params
import lal
import lalsimulation as lalsim
import numpy as np
from math import cos, sin
from pycbc.types import TimeSeries


def planck_window(N, eps, left=True, right=False, xp=np):

    P = np.ceil(eps * N).astype(int)
    n = xp.arange(1, P)
    w_decay = xp.zeros(P)
    w_decay[1:] = 1 / (1 + xp.exp(P/n - P/(P-n)))
    w = xp.ones(N)
    if right: w[-P:] = xp.flip(w_decay)
    if left : w[:P]  = w_decay
    return w


def taper_and_pad_mode(time, h_lm, eps_left=1e-3, taper_right=False):
    """
    Taper the left edge with a Planck window, zero-pad both ends **and**
    return a boolean mask that marks the untouched portion of the signal.
    """
    dt   = time[1] - time[0]

    # 1. window ────────────────────────────────────────────────────
    win   = planck_window(len(h_lm), eps_left, left=True, right=taper_right)
    h_tp  = h_lm * win

    # 2. zero-pad ─────────────────────────────────────────────────
    h_pad = np.pad(h_tp, (pad, pad))
    t_pad = time[0] - pad*dt + np.arange(len(h_pad)) * dt

    # 3. physical-sample mask  (True where window == 1)
    mask_phys = np.zeros_like(h_pad, dtype=bool)
    mask_phys[pad + np.flatnonzero(win == 1)] = True

    return t_pad, h_pad, mask_phys



def Choose_modes(mode_list):
    """
    Generate waveform parameters based on the chosen spherical modes.
    Args:
        mode_list: A list of tuples representing the spherical modes to be used, e.g., [(2, 2), (2, -2)].
                   If "All" is passed, all spherical modes will be used.
    Returns:
        waveform_params: A dictionary containing the mode array or None if all modes are used.
    """
    if mode_list == "All":
        # Use all spherical modes
        waveform_params = None
        return waveform_params
    
    else:
    
        # Create the waveform parameters structure
        waveform_params = lal.CreateDict()
        mode_array = lalsim.SimInspiralCreateModeArray()
        for l, m in mode_list:
            lalsim.SimInspiralModeArrayActivateMode(mode_array, l, m)
        lalsim.SimInspiralWaveformParamsInsertModeArray(waveform_params, mode_array)

        return waveform_params


def simulationTD_modes(Approximant, mode_list, parameters: params) -> dict:


    waveform_params = Choose_modes(mode_list)
    params_for_ChooseTDModes = parameters.params_for_ChooseTDModes()

    lmax=max([l for l, m in mode_list]) if mode_list != "All" else None

    hlm = lalsim.SimInspiralChooseTDModes(**params_for_ChooseTDModes,
            deltaT=delta_T,
            f_min=f_min,
            f_ref=f_ref,
            LALpars=waveform_params, lmax=lmax,
            approximant=Approximant
        )
    
    # Extraer el modo (2,2)
    mode22 = lalsim.SphHarmTimeSeriesGetMode(hlm, 2, 2)

    time = np.arange(mode22.data.length) * mode22.deltaT

    # locate the sample of maximum amplitude
    peak_idx_0 = np.argmax(np.abs(mode22.data.data))
    t_peak   = time[peak_idx_0]

    # time array shifted so that t = 0 at the 2‑2 peak
    t_shifted = time - t_peak #TODO

    modes = {}
    for (l, m) in mode_list:
        ts = lalsim.SphHarmTimeSeriesGetMode(hlm, l, m)
        if ts is not None:
            t_pad, h_pad, mask_phys = taper_and_pad_mode(
                    t_shifted,                      # original grid
                    ts.data.data.astype(np.complex128),
                    pad )
            modes[(l, m)] = {"time": t_pad, "h_lm": h_pad}

    return modes


# Compute the target gravitational wave to use it as a global variable
modes_target = []  # List to store the total strain for each target gravitational wave

for Info in list_Info_target:
    #Info_target contains information about the target gravitational waves:
    # Simulate the target gravitational wave
    modes_target.append(simulationTD_modes(Approximant=Info[0], mode_list=mode_list_target, parameters=Info[1]))


def divide_list(data, n_workers):
    """
    Divide a list into approximately equal parts for parallel processing.
    Args:
        data: The list to be divided.
        n_workers: The number of parts to divide the list into.
    Returns:
        list: A list of sublists, each containing a portion of the original data.
    """
    k, m = divmod(len(data), n_workers)
    return [data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_workers)]


# Divide Info_target and h_target into parts for parallel processing
list_modes_target = modes_target        # Backup of the original h_target
Info_target = divide_list(list_Info_target, n_workers)
modes_target = divide_list(modes_target, n_workers)
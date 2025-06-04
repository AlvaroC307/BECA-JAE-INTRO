from pycbc.filter import match as simple_match, optimized_match
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.types import TimeSeries
from scipy.interpolate import interp1d 
from Initial_Values import f_min, f_max


import numpy as np


def complex_interp(t, t_arr, arr):
    """Cubic-spline complex interpolation that preserves complex structure."""
    re = interp1d(t_arr, arr.real, kind="cubic", fill_value=0.0, bounds_error=False)
    im = interp1d(t_arr, arr.imag, kind="cubic", fill_value=0.0, bounds_error=False)
    return re(t) + 1j * im(t)


def planck_window(N, eps, left=True, right=False, xp=np):
    P = np.ceil(eps * N).astype(int)
    n = xp.arange(1, P)
    w_decay = xp.zeros(P)
    w_decay[1:] = 1 / (1 + xp.exp(P/n - P/(P-n)))
    w = xp.ones(N)
    if right: w[-P:] = xp.flip(w_decay)
    if left : w[:P]  = w_decay
    return w


def taper_and_pad_mode(time, h_lm, pad, eps_left=1e-3, taper_right=False):
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


def align_modes(
    modes_A: dict[tuple[int, int], dict],
    modes_B: dict[tuple[int, int], dict],
    t0: float,
    t1: float,
    *,
    low_frequency_cutoff: float = f_min,
    high_frequency_cutoff: float = f_max,
    N: int = 50_000,              #   5× faster than 5e5, same accuracy
    amp_tol: float = 1e-30,
) -> dict[tuple[int, int], dict]:
    """
    Align every (l,m) mode in *modes_B* to *modes_A*.

    Returns
    -------
    aligned_modes_B : dict    {(l,m): {"time":…, "h_lm":…}}
    """
    # --------------------------------------------------  (2,2) matching
    if (2, 2) not in modes_A or (2, 2) not in modes_B:
        raise KeyError('(2,2) mode missing in one of the inputs')

    t_B   = modes_B[(2, 2)]['time']
    dt_B  = t_B[1] - t_B[0]

    h22_A = complex_interp(t_B, modes_A[(2,2)]['time'],
                                 modes_A[(2,2)]['h_lm'])
    h22_B = modes_B[(2,2)]['h_lm']

    A_ts = TimeSeries(h22_A.real, delta_t=dt_B)
    B_ts = TimeSeries(h22_B.real, delta_t=dt_B)

    _, shift_idx, alpha = optimized_match(
        A_ts, B_ts,
        psd=None,
        low_frequency_cutoff=low_frequency_cutoff,
        high_frequency_cutoff=high_frequency_cutoff,
        return_phase=True
    )

    tau = shift_idx * dt_B               #   sub-sample time shift  [s]

    # --------------------------------------------------  ψ / φ options
    opts = [(0.0,           alpha/2.0),
            (np.pi, (alpha - np.pi)/2.0)]
    opts_full = opts + [(ψ, φ+np.pi) for ψ, φ in opts]   # for odd m only

    # --------------------------------------------------  interpolants
    interp_A = {k: lambda x, v=v: complex_interp(x, v['time'], v['h_lm'])
                for k, v in modes_A.items()}
    interp_B = {k: lambda x, v=v: complex_interp(x, v['time'], v['h_lm'])
                for k, v in modes_B.items()}

    t_test = np.linspace(t0, t1, N)

    # pick (2,1) if loud enough, else (4,4)
    try:
        amp_21 = max(np.abs(interp_A[(2,1)](t_test)).max(),
                     np.abs(interp_B[(2,1)](t_test - tau)).max())
        use_21 = amp_21 > amp_tol
    except KeyError:
        use_21 = False

    test_mode = (2,1) if use_21 else (4,4)
    
    if test_mode not in interp_A or test_mode not in interp_B:
        psi = phi = 0.0
    else:
        ell, m = test_mode
        cand = opts_full if (m % 2) else opts

        A = interp_A[test_mode](t_test)
        best_err = np.inf
        psi = phi = 0.0
        for ψ, φ in cand:
            phase = ψ + m*φ
            B = interp_B[test_mode](t_test - tau) * np.exp(-1j*phase)
            err = np.trapz(np.abs(A-B)**2, t_test)
            if err < best_err:
                best_err, psi, phi = err, ψ, (φ + np.pi)%(2*np.pi) - np.pi

    # --------------------------------------------------  apply shifts
    aligned = {}
    for (ell, m), mode in modes_B.items():
        dt = mode['time'][1] - mode['time'][0]

        # split into two real TimeSeries, shift each, then recombine
        ts_re = TimeSeries(mode['h_lm'].real, delta_t=dt)
        ts_im = TimeSeries(mode['h_lm'].imag, delta_t=dt)

        re_shift = ts_re.cyclic_time_shift(tau).numpy()
        im_shift = ts_im.cyclic_time_shift(tau).numpy()

        h_shift  = (re_shift + 1j*im_shift) * np.exp(-1j * (psi + m*phi))
        # --- build a matching time grid (length == len(h_shift)) ----
        t0      = mode["time"][0]          # same epoch as original
        time_al = t0 + np.arange(len(h_shift)) * dt


        aligned[(ell, m)] = {
            "time": time_al,
            "h_lm": h_shift.astype(np.complex128)
        }

    return aligned











def perform_match(h1:TimeSeries, h2:TimeSeries, f_lower=f_min, f_high=f_max,
                   optimized = False, return_phase = False)->tuple:
    """Function to cumpute the match of two given gravitational waves

    Args:
        h1 (TimeSeries|FrequencySeries): First Gravitational Wave 
        h2 (TimeSeries|FrequencySeries): Second Gravitational Wave
        f_lower (float, optional): Low frequency cutoff. Defaults to f_min
        f_high (float, optional): High frequency cutoff. Defaults to f_max
        optimized (bool, optional): This parameter tells us to use simple_match or optimized_match. Defaults to False.
        return_phase (bool, optional): This parameter tells us to return the phase or not. Defaults to False.

    Returns:
        tuple: The match between the GWs
    """
    h1, h2 = h1.real(), h2.real() # The Time Domain only needs the real part of the GWs
    
    # Match the signal sizes
    length = max(len(h1), len(h2))
    h1.resize(length); h2.resize(length) 

    # Choose the step to compute the PSD
    delta_f = 1/h1.duration
    length = length//2 + 1
    
    psd = aLIGOZeroDetHighPower(length, delta_f, f_lower) # Compute PSD using the base LIGO noise at Zero Detuning and High Power

    """ #--------------

    psi, phi, tau = align_with_pycbc(modes_P, modes_S, t0 = t_pk - 6, t1 = t_pk - 3,
                                psd = aLIGOZeroDetHighPower(length, delta_f, f_lower),
                                freq_psd = freq_psd)
    print("tau, phi and psi:", tau, phi, psi)    
            
    modes_S_shift = {}
    for (ell, mm), v in modes_S.items():
        rot = np.exp(-1j * (psi+ mm *phi))                       # phase rotation
        modes_S_shift[(ell, mm)] = {
            "time": v["time"] + tau,                            # time shift
            "h_lm": v["h_lm"] * rot,                          # complex strain
    }

    #-------------- """

    # Compute Match
    match_kwargs = dict(vec1 = h1, vec2 = h2, psd = psd, low_frequency_cutoff = f_lower, high_frequency_cutoff = f_high, return_phase = return_phase)
    return optimized_match(**match_kwargs) if optimized else simple_match(**match_kwargs, subsample_interpolation = True)
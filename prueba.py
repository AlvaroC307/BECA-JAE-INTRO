from match import align_modes, perform_match
from Target import taper_and_pad_mode
import lalsimulation as lalsim
import lal
from classes import params
import math
import numpy as np

def chirp_mass_function(masses:list)->float: 
    """Function to calculate the Chirp Mass of a binary system
    Args:
        masses (list or tuple): A list or tuple with the masses of the two black holes
    """
    return ((masses[0]*masses[1])**(3/5))/((masses[0]+masses[1])**(1/5))


def eff_spin_function(masses:list, spins:list)->float:
    """Function to Calculate the effective spin parameter of a binary system
    Args:
        masses (list): A list with the masses of the two black holes
        spins (list): A list with the dimensionless spin parameters of the two black holes
    """
    return (spins[0]*masses[0]+spins[1]*masses[1])/(masses[0]+masses[1])


class params_variant: # Class with all the basic information mandatory to simulate a GW
    def __init__(self, masses:tuple, spin1:tuple = (0,0,0), spin2:tuple= (0,0,0), r: float = 1e6 * lal.PC_SI, phiRef:float = 0):
        # Atributo de instancia (Se vincula a la instancia)
        self.m1: float = masses[0] # Mass of the first Black Hole
        self.m2: float = masses[1] # Mass of the second Black Hole

        self.S1x:float = spin1[0] # First Component of the dimensionless spin parameter of the first Black Hole
        self.S1y:float = spin1[1] # Second Component of the dimensionless spin parameter of the first Black Hole
        self.S1z:float = spin1[2] # Third Component of the dimensionless spin parameter of the first Black Hole

        self.S2x:float = spin2[0] # First Component of the dimensionless spin parameter of the second Black Hole
        self.S2y:float = spin2[1] # Second Component of the dimensionless spin parameter of the second Black Hole
        self.S2z:float = spin2[2] # Third Component of the dimensionless spin parameter of the second Black Hole

        self.r:float = r # Distance to the binary system
        self.phiRef:float = phiRef # Orbital phase at reference, half of main GW phase at reference


    def Q(self) -> float: # Chirp Mass of the binary system
        return self.m1/self.m2


    def chirp_mass(self) -> float: # Chirp Mass of the binary system
        return chirp_mass_function([self.m1, self.m2])
    

    def eff_spin(self) -> float: # Effective Spin Parameter of the binary system
        return eff_spin_function([self.m1, self.m2], [self.s1z, self.s2z])
    
    
    def spin1p_mod(self) -> float: # Module of the perpendicular component of the spin of the first black hole
        return math.sqrt(self.s1x**2+self.s1y**2)
    

    def spin1p_angle(self) -> float: # Angle of the perpendicular component of the spin of the first black hole. spin1x + i spin1y 
        return math.atan2(self.s1y, self.s1x) 

    def spin2p_mod(self) -> float: # Module of the perpendicular component of the spin of the second black hole
        return math.sqrt(self.s2x**2+self.s2y**2)

    def spin2p_angle(self) -> float: # Angle of the perpendicular component of the spin of the second black hole. spin1x + i spin1y 
        return math.atan2(self.s2y, self.s2x) 


    def __str__(self) -> str: # String to format how does print(params) work
        return f"mass1: {self.m1} | mass2: {self.m2} | spin1: {self.s1x, self.s1y, self.s1z}| spin2: {self.s2x, self.s2y, self.s2z}, | distance: {self.distance}, |phiRef: {self.phiRef}"
    



delta_T = 1.0 / 4096.0
f_min = 15
f_max = 250
f_ref = f_min
Approximant_A = lalsim.GetApproximantFromString("IMRPhenomTPHM")
Approximant_B = lalsim.GetApproximantFromString("SEOBNRv4P")

masses = (30*lal.MSUN_SI, 30*lal.MSUN_SI)
spin1 = (0.6, 0.6, 0.3)
spin2 = (0.7, -0.7, 0.0)
r = 1e6*lal.PC_SI 
parameters = params_variant(masses, spin1, spin2, r=r)

incl = 0
longascnodes = 0.

all_parameters = params(masses, spin1, spin2, r=r, incl=incl, longAscNodes=longascnodes)


pad=0


mode_list_imr = [(2, 2), (2, -2), (2, 1), (2, -1)]
mode_list_seobnr = [(2, 2), (2, 1)]

# Create the waveform parameters structure
waveform_params = lal.CreateDict()
mode_array = lalsim.SimInspiralCreateModeArray()
for l, m in mode_list_imr:
    lalsim.SimInspiralModeArrayActivateMode(mode_array, l, m)
lalsim.SimInspiralWaveformParamsInsertModeArray(waveform_params, mode_array)


hlm_A= lalsim.SimInspiralChooseTDModes(**parameters.__dict__,
            deltaT=delta_T,
            f_min=f_min,
            f_ref=f_ref,
            LALpars=waveform_params, lmax=2,
            approximant=Approximant_A
        )


# Extraer el modo (2,2)
mode22_A = lalsim.SphHarmTimeSeriesGetMode(hlm_A, 2, 2)

N_A      = mode22_A.data.length
t_A_0      = np.arange(N_A) * mode22_A.deltaT

# locate the sample of maximum amplitude
peak_idx_0 = np.argmax(np.abs(mode22_A.data.data))
t_peak_0   = t_A_0[peak_idx_0]

# time array shifted so that t = 0 at the 2‑2 peak
t_A_shifted = t_A_0 - t_peak_0
time_grid1 = t_A_shifted 

modes_A = {}
for (l, m) in mode_list_imr:
    ts = lalsim.SphHarmTimeSeriesGetMode(hlm_A, l, m)
    if ts is not None:
        t_pad, h_pad, mask_phys1 = taper_and_pad_mode(
                time_grid1,                      # original grid
                ts.data.data.astype(np.complex128),
                pad )
        modes_A[(l, m)] = {"time": t_pad, "h_lm": h_pad}


# Create the waveform parameters structure
waveform_params = lal.CreateDict()
mode_array = lalsim.SimInspiralCreateModeArray()
for l, m in mode_list_seobnr:
    lalsim.SimInspiralModeArrayActivateMode(mode_array, l, m)
lalsim.SimInspiralWaveformParamsInsertModeArray(waveform_params, mode_array)

hlm_B= lalsim.SimInspiralChooseTDModes(**parameters.__dict__,
            deltaT=delta_T,
            f_min=f_min,
            f_ref=f_ref,
            LALpars=waveform_params, lmax=2,
            approximant=Approximant_B
        )



# Extraer el modo (2,2)
mode22_B = lalsim.SphHarmTimeSeriesGetMode(hlm_A, 2, 2)

N_B      = mode22_B.data.length
t_B      = np.arange(N_B) * mode22_B.deltaT

# locate the sample of maximum amplitude
peak_idx_1 = np.argmax(np.abs(mode22_B.data.data))
t_peak_1   = t_B[peak_idx_1]

# time array shifted so that t = 0 at the 2‑2 peak
t_B_shifted = t_B - t_peak_1
time_grid2 = t_B_shifted 

modes_B = {}
for (l, m) in mode_list_seobnr:
    ts = lalsim.SphHarmTimeSeriesGetMode(hlm_B, l, m)
    if ts is not None:
        t_pad, h_pad, mask_phys2 = taper_and_pad_mode(
                time_grid2,                      # original grid
                ts.data.data.astype(np.complex128),
                pad )
        modes_B[(l, m)] = {"time": t_pad, "h_lm": h_pad}




modes_B_shift = align_modes(modes_A, modes_B, t0=t_peak_0, t1=t_peak_1, 
                      low_frequency_cutoff=f_min, high_frequency_cutoff=f_max)



from pycbc.types import TimeSeries

h_A = 0j

for (l, m) in modes_A.keys():
    Ylm = lal.SpinWeightedSphericalHarmonic(incl, longascnodes, -2, l, m)


    h_A += modes_A[(l, m)]["h_lm"]*Ylm

h_B = 0j

for (l, m) in modes_B_shift.keys():
    Ylm = lal.SpinWeightedSphericalHarmonic(incl, longascnodes, -2, l, m)

    h_B += modes_B_shift[(l, m)]["h_lm"]*Ylm




hp_A=TimeSeries(h_A.real, delta_t=delta_T)
hp_B=TimeSeries(h_B.real, delta_t=delta_T)
hc_A=TimeSeries(h_A.imag, delta_t=delta_T)
hc_B=TimeSeries(h_B.imag, delta_t=delta_T)


import matplotlib.pyplot as plt

""" # Plot the plus/cross polarization of the gravitational wave
plt.figure(figsize=(10, 5))
plt.plot(modes_A[(l,m)]["time"], h_A.real, label='h+')
plt.plot(modes_A[(l,m)]["time"], h_A.imag, label='hx')
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.title(f'Gravitational Waveform {Approximant_A}')
plt.legend()
plt.grid()
plt.show()

# Plot the plus/cross polarization of the gravitational wave
plt.figure(figsize=(10, 5))
plt.plot(modes_B_shift[(l,m)]["time"], h_B.real, label='h+')
plt.plot(modes_B_shift[(l,m)]["time"], h_B.imag, label='hx')
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.title(f'Gravitational Waveform {Approximant_A}')
plt.legend()
plt.grid()
plt.show() """




#---------------------------------MISMATCH

match, _ = perform_match(hp_A, hp_B, f_min, f_max, optimized = False)

print(f"El mismatch de optimized match es: {1-match}")
match, _ = perform_match(hc_A, hc_B, f_min, f_max, optimized = False)

print(f"El mismatch de optimized match es: {1-match}")










def Choose_modes(mode_list):
    """
    Generate waveform parameters based on the chosen spherical modes.
    Returns:
        waveform_params: A dictionary containing the mode array or None if all modes are used.
    """

    # Create the waveform parameters structure
    waveform_params = lal.CreateDict()
    mode_array = lalsim.SimInspiralCreateModeArray()
    for l, m in mode_list:
        lalsim.SimInspiralModeArrayActivateMode(mode_array, l, m)
    lalsim.SimInspiralWaveformParamsInsertModeArray(waveform_params, mode_array)

    return waveform_params


def simulationTD(app, Approximant, parameters: params) -> tuple:
    """
    Simulate a binary system using the given approximant and parameters.
    Args:
        Approximant: The chosen approximant for the simulation.
        parameters (params): A class containing all the mandatory parameters needed to compute the gravitational wave.
    Returns:
        tuple: A tuple containing three numpy arrays: h_plus, h_cross, and the time array.
    """
    # Generate waveform parameters based on the chosen modes
    waveform_params = Choose_modes(app)

    with lal.no_swig_redirect_standard_output_error():
        # Generate the waveform using the given parameters
        hplus, hcross = lalsim.SimInspiralChooseTDWaveform(
            **parameters.__dict__,
            deltaT=delta_T,
            f_min=f_min,
            f_ref=f_ref,
            params=waveform_params,
            approximant=Approximant
        )

    # Extract time series for both plus (h_plus) and cross (h_cross) polarizations
    h_plus = hplus.data.data
    h_cross = hcross.data.data
    time = np.arange(len(h_plus)) * delta_T

    # Return the data as numpy arrays
    return h_plus, h_cross, time




hp_A_original, hc_A_original, times = simulationTD(mode_list_imr, Approximant_A, all_parameters)
hp_B_original, hc_B_original, times = simulationTD(mode_list_seobnr, Approximant_B, all_parameters)

hp_A_original = TimeSeries(hp_A_original, delta_t=delta_T)
hp_B_original = TimeSeries(hp_B_original, delta_t=delta_T)
hc_A_original = TimeSeries(hc_A_original, delta_t=delta_T)
hc_B_original = TimeSeries(hc_B_original, delta_t=delta_T)

match_original, _ = perform_match(hp_A_original, hp_B_original, f_min, f_max, optimized = False)
print(f"El mismatch de original es: {1-match_original}")
match_original, _ = perform_match(hc_A_original, hc_B_original, f_min, f_max, optimized = False)
print(f"El mismatch de original es: {1-match_original}")
import lal
import math

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


class params: # Class with all the basic information mandatory to simulate a GW
    def __init__(self, masses:tuple, spin1:tuple = (0,0,0), spin2:tuple= (0,0,0), r: float = 1e6 * lal.PC_SI,
                  incl:float = 0, phiRef:float = 0, longAscNodes:float=0, ecc:float = 0, meanPerAno:float = 0):
        # Atributo de instancia (Se vincula a la instancia)
        self.m1: float = masses[0] # Mass of the first Black Hole
        self.m2: float = masses[1] # Mass of the second Black Hole

        self.s1x:float = spin1[0] # First Component of the dimensionless spin parameter of the first Black Hole
        self.s1y:float = spin1[1] # Second Component of the dimensionless spin parameter of the first Black Hole
        self.s1z:float = spin1[2] # Third Component of the dimensionless spin parameter of the first Black Hole

        self.s2x:float = spin2[0] # First Component of the dimensionless spin parameter of the second Black Hole
        self.s2y:float = spin2[1] # Second Component of the dimensionless spin parameter of the second Black Hole
        self.s2z:float = spin2[2] # Third Component of the dimensionless spin parameter of the second Black Hole

        self.distance:float = r # Distance to the binary system
        self.inclination:float = incl # Angle of inclination of the system with respect to the earth
        self.phiRef:float = phiRef # Orbital phase at reference, half of main GW phase at reference
        self.longAscNodes:float = longAscNodes # longitude of ascending nodes

        self.eccentricity:float = ecc # Eccentricity of the binary system
        self.meanPerAno : float = meanPerAno # Parameter for the simulation of a system with eccentricity


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



    def params_for_ChooseTDModes(self) -> dict:

        params_intrinsic = {
            "m1": self.m1,
            "m2": self.m2,
            "S1x": self.s1x,
            "S1y": self.s1y,
            "S1z": self.s1z,
            "S2x": self.s2x,
            "S2y": self.s2y,
            "S2z": self.s2z,
            "r": self.distance,
            "phiRef": self.phiRef,
        }

        return params_intrinsic


    def params_ext(self) -> dict:
        """Function to return the extra parameters needed for the simulation of a binary system with eccentricity"""
        params_extra = {
            "inclination": self.inclination,
            "longAscNodes": self.longAscNodes,
            "eccentricity": self.eccentricity,
            "meanPerAno": self.meanPerAno
        }
        return params_extra


    def __str__(self) -> str: # String to format how does print(params) work
        return f"mass1: {self.m1} | mass2: {self.m2} | spin1: {self.s1x, self.s1y, self.s1z}| spin2: {self.s2x, self.s2y, self.s2z}, | distance: {self.distance}, |inclination: {self.inclination}, |phiRef: {self.phiRef}, |longAscNodes: {self.longAscNodes}, |eccentricity: {self.eccentricity}| meanPerAno: {self.meanPerAno}"
    
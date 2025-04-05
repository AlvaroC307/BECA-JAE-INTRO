from Initial_Values import M_c_and_q_m
import lal

masses = M_c_and_q_m(1.5,100*lal.MSUN_SI )
print(masses[0]/lal.MSUN_SI, masses[1]/lal.MSUN_SI)
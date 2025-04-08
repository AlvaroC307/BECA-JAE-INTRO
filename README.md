# FITTING FACTOR

For the moment, this codes calculates the Fitting Factor of a target Gravitational Wave. The Fitting Factor is calculated using a hierarchical program which follows the following hierarchy:

- Q=m1/m2, M_chirp, Chi_eff -> Chi_2z, Chi_p, Incl -> Theta_prec, LongAscNodes, Pol (Full optimization).

- Q=m1/m2, M_chirp, Chi_eff -> Chi_2z, Chi_p, Theta_prec (Intrinsic optimization).

There are multiple options inside the program that you can use to obtain the Fitting Factor.

### Options:

- The user can either compute the optimization for **every possible** relevant **parameter** in a gravitational wave or compute only the **intrinsic parameters**.

- In the program there are three modes of supplying the target gravitational wave. The first one is as a **single point** in the complete **parameter space**, another mode is simulating **n<sub>points<sub>** completely **random** points in the parameter space and finally, the user can also supply the target GW as a **NRfile** (copy the file into the /Input/NR_files/ folder). In the three modes, the user can **choose the approximant** for the target GW or the approximant used during the optimization (or a list of approximants)<sup>1,2</sup>. 

- The user can choose the **spherical armonics modes** that they want to use (Now, there are only two possibilities: Every mode in the approximant or only the (2,2) and (2,-2)). 

### Packages needed

There are many packages needed to use this program. As may be expected: **numpy, lal, lalsim, pycbc, nlopt and matplotlib** are all mandatory (although matplot lib just creates some plots that may be commented out).

In addition, I used a package to have colours in the prints and make it easier to read while debugging: **termcolor**. It is not esential to the program so the user **can comment out** the prints, but most of the prints that happen during the optimization use it.

# How to Use the code

To use the code you should only change the file Initial_Values.py (unless you know what you are doing with it). Here, you can read the most useful things to set in the code.

To choose a option of the code you can change the lines 44 to 46. The options are set on lines 38-40, so you can add more options if you want. The line 48 are the number of random points in the parameter space to generate (if you chose Target_Form == "Random_Space_Point") and the line 49 are the number of workers (cpus) that you want to use in multiprocessing.

The lines 53 to 56 set the frequency constants of the simulation. Then, in the lines 58-67 you can set the name of the aproximant that you want to use, you should only change line 60 or 64.

Finally, you can set r and Phi_ref in the lines 70 and 71 if you want a specific set of those reference values. Moreover, in Target_Form == "Param_Space_Point" you can set a list of the target values that you want to test, the important thing is that in the end there is a param class ((masses), (spins1), (spins2), r, incl, PhiRef, longascnodes) and a Info_target (Approximant, param class, polarization). Target_Form == "NR_file" is yet to be tested and finalized.

Once everything regarding the target gravitational waves is set, you can execute the code by writing: python main.py. After that, the data is saved in the folder /Data/. 
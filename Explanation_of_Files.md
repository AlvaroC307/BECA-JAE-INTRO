# TO BE UPDATED

### What does each file do/what functions are inside:

- **Initial_Values.py**: In this file the user specify the **choice** between the **different options** in the program and the **approximants** they want to use. Moreover, it reads the file "Input/Input.csv" for more information regarding certain target parameters and/or the NR_files to find the Fitting Factor for an NR GW. Additionally, there are a few functions (**M_c_and_q_m**, **Eff_spin_and_spin1** and **spin1p_mod_and_angle**) which are just the inverted relation of some parameters, we may use them here to choose the target parameters.

- **Target.py**. In this file there are two functions, one for generating the waveform_params depending on the choise of **Spherical Modes** and another one for **computing the GWs**. Moreover, in this file we compute the target GWs to then call them from other parts of the code.
 
- **classes.py**: In this file we define a class called **params** where every parameter (**except polarization**) needed to compute a GW is included. Additionally, it contains two more functions, these functions compute the **chirp mass** and the **effective spin parameter** given the necessary values.

- **match.py**: In this file there is only one function, the one used to **compute the match** between two GWs in the **Time Domain**.

- **optimization_functions.py**: This is one of the most important files. There are many functions there is **opt_match_full**, the function that we want to optimize, i.e. the **simulation of a GW** given the complete list of parameters and the **computation of the match** between itself and a target GW. Then, other functions are **iterations of this one** with different uses, they **limit** the quantity of **parameters** to optimize. Finally, there are the functions which define the **optimizers**. There are **five** of them, two for the only intrinsic case and another three for the full case. The **tolerance** in the optimization **increases** as the programm goes to **higher steps**.

- **global_variables.py**: This file only exists to inizialize two integers that specify the Approximant to use in the optimization or the target Gw in the case that the user wants to find more than one Fitting Factor.<sup>1</sup>

- **main.py**: Finally, the main file. This is divided in **three ways**, first of all, the functions used to save the Data in the csvs. After that, **main_optimization_intrinsic** and **main_optimization_full** are the most important functions in the whole program. The **hierarchical optimization** is embeded in these functions. They execute every step six times for **six** different list of **initial parameters** in every step<sup>2</sup>. To continue, the program keeps the best match and uses those already optimized parameters plus a list of new parameters to compute the following step. Finally, the **last step** just has a **higher tolerance** in the optimization and uses the best obtained value of the previous step. Then, the main function: it sees what choices had the user made and executes the chosen function.

- **indistinguishability.py**: This file has two functions. The first one, **overlap** computes the overlap between the Gw generated using the Target Approximant and the one generated using the Optimization Approximant. Then, the other function, **minimun_SNR**, computes the minimum SNR at which point the biases from the choice of waveform model start to show up. It uses two different formulae to compare them.

- **Analysis_Data.py**: This file only reads the csvs from Data and makes some plots to see results. 

- About the **folders**, Input has a folder for the **NR files** were we can save Numerical Relaticity files to be used as a target. **Data** is the folder were I **save the data obtained** during a compilation of the code and **Graphics** is the folder where I save the plots made.

### Limitations/Bugs/Comments:


- 1: There may be a more straighforward way of doing this, but there was a **problem with nlopt** that I couldn't see how to avoid. The nlopt optimizers can only be fed functions with a specific syntaxis, **a list of parameters** (all of them are optimizable) and **a gradient** (which I don't need because I use derivative free methods). Because of this, I have to fed the target GW and the approximant as a global variable (therefore n_aprox_opt and n_target, which decide the item in the list, **are needed as gloval variables**)

- 2: A **better way** of making the program **more robust** without losing much time (a few seconds) is to create **more initial lists of parameters**, for example 8 or 10 initial points in the parameter space. However, I found out that 6 is a good enough number. In my opinion, the **best moment** to increase the number of initial points is in the **first step** (it is by far the fastest and also the most relevant). 

#### Extra Limitations:

- The **eccentricity** and the **precession of the small black hole** are not considered in the chosen hierarchy and should be added.

- We may fasten the program by changing the **size of the constraints** in different steps. For example, after the first step we may know the Chirp mass with good accuracy but in the second step a few of the evaluations are made far from the known point. We should be able to **limit the size** of the constraint box to avoid these evaluations. However, how much should the box be limited is not clear so we have not included it yet, many many tests should be made to ensure that the box is not too limited.

- There is not an option to use **real life data**, but should not be that difficult to add.

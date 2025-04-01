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

### What does each file do/what functions are inside:

- **Initial_Values.py**: In this file the user specify the **choice** between the **different options** in the program and the **approximants** they want to use. Moreover, it reads the file Input/Input.csv for more information regarding certain target parameters <sup>3</sup> and/or the NR_files to find the Fitting Factor for an NR GW.

- **classes.py**: In this file we create a class called **params** where every parameter (**except polarization**) needed to compute a GW is included. Also, there are two more functions defined here, these functions compute the **chirp mass** and the **effective spin parameter** given the necessary values.

- **Simulations.py**: In this file there are two functions, one for **computing the GW** for the optimization and another one for computing the target GW<sup>4</sup>. Also, here the program compute once the target GW. During the rest of the program it will **call this file to use the target GW**.

- **match.py**: In this file there is only one function, the one used to **compute the match** between two GWs in the **Time Domain**.

- **optimization_functions.py**: This is one of the most important files. There are many functions, a few of them (**M_c_and_q_m**, **Eff_spin_and_spin1** and **spin1p_mod_and_angle**) are just the inverted relation of those parameters with the masses, spins in the z-direction and spin1_xy respectively. Then, there is **opt_match_full**, the function that we want to optimize, i.e. the **simulation of a GW** given the complete list of parameters and the **computation of the match** between itself and a target GW. Finally, the rest of the functions are **iterations of this one** with different uses, they **limit** the quantity of **parameters** to optimize.

- **global_variables.py**: This file only exists to inizialize two integers that specify the Approximant to use in the optimization or the target Gw in the case that the user wants to find more than one Fitting Factor.<sup>5</sup>

- **main.py**: Finally, the main file. This is divided in **three ways**<sup>6</sup>, first of all, there are the functions which define the **optimizers**. There are **five** of them, two for the only intrinsic case and another three for the full case. The **tolerance** in the optimization **increases** as the programm goes to **higher steps**<sup>7</sup>. After that, **main_optimization_intrinsic** and **main_optimization_full** are the most important functions in the whole program. The **hierarchical optimization** is embeded in these functions. They execute every step six times for **six** different list of **initial parameters** in every step<sup>8</sup>. To continue, the program keeps the best match and uses those already optimized parameters plus a list of new parameters to compute the following step. Finally, the **last step** just has a **higher tolerance** in the optimization and uses the best obtained value of the previous step. Then, the main function: it sees what choices had the user made and executes the chosen function.

- About the **folders**, Input has a folder for the **NR files** and a **Target.csv** were we can specify a few target parameters to be used, but should be reworked. **Data** is the folder were I **save the data obtained** during a compilation of the code.

- Most of the stuff in the **other folders** should be **ignored**. The Jupiter notebooks were made to learn/test a few ideas. The Mismatch_test folder is just the code from the Baleares. Graphics save a comparison between the target GW and the optimized one, but it is not updated to work with multiple targets.

### Limitations/Bugs/Comments:

- 1: Choosing a **list of approximants** for the target GW n<sub>points</sub> random points isn't as straight forward as in the other cases and **should be made more intuitively**.

- 2: Moreover, using a list of **approximants for the optimization** GW does **not save** the information and may cause errors. (Important to fix soon)

- 3: Perhaps it would be better to just write the parameters in the Initial_Values.py file directly?.

- 4: I don't remember why I made two different functions, they are the same. (Easy Fix)

- 5: There may be a more straighforward way of doing this, but there was a **problem with nlopt** that I couldn't see how to avoid. The nlopt optimizers can only be fed functions with a specific syntaxis, **a list of parameters** (all of them are optimizable) and **a gradient** (which I don't need because I use derivative free methods). Because of this, I have to fed the target GW and the approximant as a global variable (therefore n_aprox_opt and n_target, which decide the item in the list, **are needed as gloval variables**)

- 6: Maybe I should consider moving some functions to a new file?

- 7: We can make the program **more robust** increasing the **tolerance in the last step**, but it would run slower.

- 8: A **better way** of making the program **more robust** without losing much time (a few seconds) is to create **more initial lists of parameters**, for example 8 or 10 initial points in the parameter space. However, I found out that 6 is a good enough number. In my opinion, the **best moment** to increase the number of initial points is in the **first step** (it is by far the fastest and also the most relevant). 

#### Extra Limitations:

- The **eccentricity** and the **precession of the small black hole** are not considered in the chosen hierarchy and should be added.

- We may fasten the program by changing the **size of the constraints** in different steps. For example, after the first step we may know the Chirp mass with good accuracy but in the second step a few of the evaluations are made far from the known point. We should be able to **limit the size** of the constraint box to avoid these evaluations. However, how much should the box be limited is not clear so we have not included it yet, many many tests should be made to ensure that the box is not too limited.

- There is not an option to use **real life data**, but should not be that difficult to add.

- Something should be changed about how does the program runs many target GW because the program **may not be able to work with multiple processes** (Important to test if true and solve)


# Bug (IMPORTANT):

Solve whatever this is:

    Third Optimization: Theta_precession, LongAscNodes and Polarization
    Traceback (most recent call last):
    File "/mnt/c/Users/USUARIO/Desktop/GitHub/BECA-JAE-INTRO/main.py", line 418, in <module>
        
    File "/mnt/c/Users/USUARIO/Desktop/GitHub/BECA-JAE-INTRO/main.py", line 389, in main
        Fitting_Factor.append(1-results[0])
    File "/mnt/c/Users/USUARIO/Desktop/GitHub/BECA-JAE-INTRO/main.py", line 329, in main_optimization_full
        prms_final[5], LongAscNodes_template[i], pol_template[i]]) # Create the templates
    IndexError: index 3 is out of bounds for axis 0 with size 3
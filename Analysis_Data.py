import csv
import math
import numpy as np
import matplotlib.pyplot as plt

def Read_Testing_Full()->list:
    # Leer el fichero de inputs y de constantes
    csv_Testing = open('./Data/Testing_FUll.csv', 'r')
    Reader_Testing = csv.reader(csv_Testing)

    # Escribir el fichero de inputs en una lista
    list_Testing = []
    for row in Reader_Testing:
        if row[0]=="1-FF":
            print("Ignore this row")
        else:
            list_Testing.append([float(x) for x in row])
    csv_Testing.close()
    
    return np.array(list_Testing)


def Read_min_SNR()->list:
    # Leer el fichero de inputs y de constantes
    csv_min_SNR = open('./Data/Min_SNR.csv', 'r')
    Reader_min_SNR = csv.reader(csv_min_SNR)

    # Escribir el fichero de inputs en una lista
    list_min_SNR = []
    for row in Reader_min_SNR:
        if row[0]=="1-FF":
            print("Ignore Header")
            i=0
        elif row[2]=="FF<Overlap":
            i+=1
            print(i)
        else:
            list_min_SNR.append([float(x) for x in row])
    csv_min_SNR.close()
    
    return np.array(list_min_SNR)

list_Testing = Read_Testing_Full()
list_min_SNR = Read_min_SNR()


def Plot_old_ind(n_p, SNR):

    return n_p/(2*SNR**2)

def Plot_new_ind(n_p, SNR, mis_FF):

    return (n_p*(1-(2/(9*n_p))+1.3*math.sqrt(2/(9*n_p)))**3)/(2*SNR**2) + (mis_FF)

def Plot_mis_FF(SNR, mis_FF):
    return mis_FF*SNR/SNR

def Plot_overlap(SNR, overlap):
    return overlap*SNR/SNR

n_p = 9
# Generate x values
SNR = np.linspace(10, 10**3, 100)

old_ind_crit = Plot_old_ind(n_p, SNR)

new_ind_crit = []
mis_FF = []
overlap = []
for i in range(len(list_Testing)):
    new_ind_crit.append(Plot_new_ind(n_p, SNR, mis_FF = list_Testing[i][0]))
    mis_FF.append(Plot_mis_FF(SNR, list_Testing[i][0]))
    overlap.append(Plot_overlap(SNR, list_min_SNR[i][1]))

name_list = ["GW150914", "q=3", "High SNR", "Simp_pol_0", "Simp_pol_pi4"]

for i, name in enumerate(name_list):
    plt.loglog(figsize=(12, 5))
    plt.plot(SNR, new_ind_crit[i], label = f'New SNR.')
    plt.plot(SNR, old_ind_crit, label = f'Old SNR')
    plt.plot(SNR, mis_FF[i], label = f'Fitting Factor.', linestyle='dashed')
    plt.plot(SNR, overlap[i], label = f'Overlap.', linestyle='dashed')
    plt.title(f'Mismatch vs SNR. {name}. IMRPhenomTPHM vs IMRPhenomPv2. All Modes')
    plt.xlabel('SNR')
    plt.ylabel('Mismatch')
    plt.legend()
    plt.savefig(f'./Graphics/{name}_IMRTPHM_vs_IMRv2_modes_all.png', bbox_inches='tight') 
    plt.savefig(f'./Graphics/{name}_IMRTPHM_vs_IMRv2_modes_all.pdf', bbox_inches='tight')
    plt.show() 

    plt.loglog(figsize=(12, 5))
    plt.plot(SNR, new_ind_crit[len(name_list)+i], label = f'New SNR.')
    plt.plot(SNR, old_ind_crit, label = f'Old SNR')
    plt.plot(SNR, mis_FF[len(name_list)+i], label = f'Fitting Factor.', linestyle='dashed')
    plt.plot(SNR, overlap[len(name_list)+i], label = f'Overlap.', linestyle='dashed')
    plt.title(f'Mismatch vs SNR. {name}. IMRPhenomTPHM vs IMRPhenomPv2. Only 2,2 mode')
    plt.xlabel('SNR')
    plt.ylabel('Mismatch')
    plt.legend()
    plt.savefig(f'./Graphics/{name}_IMRTPHM_vs_IMRPv2_modes_2.png', bbox_inches='tight') 
    plt.savefig(f'./Graphics/{name}_IMRTPHM_vs_IMRPv2_modes_2.pdf', bbox_inches='tight') 
    plt.show()



""" # Plot of the old vs new SNR evolution
plt.figure(figsize=(12, 5))
#plt.plot(list_min_SNR[:,5], list_min_SNR[:,2], label = f'New SNR')
#plt.plot(list_min_SNR[:,5], list_min_SNR[:,3], label = f'Old SNR.', linestyle='dashed')
plt.scatter(list_min_SNR[:,5], list_min_SNR[:,2], label = f'New SNR', c="blue")
plt.scatter(list_min_SNR[:,5], list_min_SNR[:,3], label = f'Old SNR.', c="orange")
plt.title(f'Indistinguishability SNR.')
plt.xlabel('Chirp Mass')
plt.ylabel('SNR')
plt.legend()
plt.savefig('./Graphics/SNR.png', bbox_inches='tight') 
plt.savefig('./Graphics/SNR.pdf', bbox_inches='tight') 
plt.show() 

# Plot of the FF vs overlap evolution
plt.figure(figsize=(12, 5))
#plt.plot(list_min_SNR[:,5], list_min_SNR[:,0], label = f'1-FF')
#plt.plot(list_min_SNR[:,5], list_min_SNR[:,1], label = f'1-Overlap.', linestyle='dashed')
plt.scatter(list_min_SNR[:,5], list_min_SNR[:,0], label = f'1-FF', c="blue")
plt.scatter(list_min_SNR[:,5], list_min_SNR[:,1], label = f'1-Overlap.', c="orange")
plt.title(f'1-FF and 1-Overlap.')
plt.xlabel('Chirp Mass')
plt.ylabel('1-FF or 1-Overlap')
plt.legend()
plt.savefig('./Graphics/FF.png', bbox_inches='tight') 
plt.savefig('./Graphics/FF.pdf', bbox_inches='tight') 
plt.show()  """
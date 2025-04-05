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
            print("Ignore this row")
        else:
            list_min_SNR.append([float(x) for x in row])
    csv_min_SNR.close()
    
    return np.array(list_min_SNR)

list_Testing = Read_Testing_Full()

average_mismatch = np.sum(list_Testing[:,0])/len(list_Testing)
average_Comp_Time = np.sum(list_Testing[:,1])/len(list_Testing)


Best_FF = max(list_Testing[:,0])
Worst_FF = min(list_Testing[:,0])
Best_Comp_Time = max(list_Testing[:,1])
Worst_Comp_Time = min(list_Testing[:,1])

print(f"The average value for 1-FF = {average_mismatch}. The average value for Commputing Time = {average_Comp_Time}.")
print(f"The best 1-FF = {Best_FF}, while the worst 1-FF = {Worst_FF}")
print(f"The best Comp_Time = {Best_Comp_Time}, while the worst Comp_Time = {Worst_Comp_Time}")

list_min_SNR = Read_min_SNR()

average_SNR = np.sum(list_min_SNR[:,2])/len(list_min_SNR)
High_SNR = max(list_min_SNR[:,2])
Low_SNR = min(list_min_SNR[:,2])

print(f"The average value for SNR = {average_SNR}. The highest value = {High_SNR}, while the lowest value = {Low_SNR}.")


# Plot of the old vs new SNR evolution
plt.figure(figsize=(12, 5))
plt.plot(list_min_SNR[:,5], list_min_SNR[:,2], label = f'New SNR')
plt.plot(list_min_SNR[:,5], list_min_SNR[:,3], label = f'Old SNR.', linestyle='dashed')
plt.scatter(list_min_SNR[:,5], list_min_SNR[:,2], label = f'New SNR', c="blue")
plt.scatter(list_min_SNR[:,5], list_min_SNR[:,3], label = f'Old SNR.', c="orange")
plt.title(f'Indistinguishability SNR. Q=1.5, s1x=s1y=0.7, s2z=0.7.')
plt.xlabel('Chirp Mass')
plt.ylabel('SNR')
plt.legend()
plt.savefig('./Graphics/SNR_Chirp_mass.png', bbox_inches='tight') 
plt.savefig('./Graphics/SNR_Chirp_mass.pdf', bbox_inches='tight') 
plt.show() 

# Plot of the FF vs overlap evolution
plt.figure(figsize=(12, 5))
plt.plot(list_min_SNR[:,5], list_min_SNR[:,0], label = f'1-FF')
plt.plot(list_min_SNR[:,5], list_min_SNR[:,1], label = f'1-Overlap.', linestyle='dashed')
plt.scatter(list_min_SNR[:,5], list_min_SNR[:,0], label = f'1-FF', c="blue")
plt.scatter(list_min_SNR[:,5], list_min_SNR[:,1], label = f'1-Overlap.', c="orange")
plt.title(f'1-FF and 1-Overlap. Q=1.5, s1x=s1y=0.7, s2z=0.7.')
plt.xlabel('Chirp Mass')
plt.ylabel('1-FF or 1-Overlap')
plt.legend()
plt.savefig('./Graphics/FF_Chirp_mass.png', bbox_inches='tight') 
plt.savefig('./Graphics/FF_Chirp_mass.pdf', bbox_inches='tight') 
plt.show() 

# Plot of the Time evolution
plt.figure(figsize=(12, 5))
plt.plot(list_min_SNR[:,5], list_Testing[:,1], label = f'Time')
plt.scatter(list_min_SNR[:,5], list_Testing[:,1], label = f'Time')
plt.title(f'1-FF and 1-Overlap. Q=1.5, s1x=s1y=0.7, s2z=0.7.')
plt.xlabel('Chirp Mass')
plt.ylabel('Time (s)')
plt.legend()
plt.savefig('./Graphics/Time_Chirp_mass.png', bbox_inches='tight') 
plt.savefig('./Graphics/Time_Chirp_mass.pdf', bbox_inches='tight') 
plt.show() 
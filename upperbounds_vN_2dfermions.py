#!/usr/bin/env python3
#SBATCH --job-name=1dfermions      # Name of your job
#SBATCH --output=1dfermions.out    # Output file
#SBATCH --error=1dfermions.err     # Error file
#SBATCH --qos=short               # Use the short QoS
#SBATCH --time=24:00:00           # Time limit (4 hours)

import numpy as np
from numpy.random import random
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.linalg import eig, expm
import argparse
from scipy.optimize import root_scalar

parser = argparse.ArgumentParser(description='the parser')
parser.add_argument("-gap", dest="gap", type=np.float64, nargs='?', const=1, help="gap")
parser.add_argument("-alpha", dest="alpha", type=np.float64, nargs='?', const=1, help="exponent")
parser.add_argument("-pref", dest="pref", type=np.float64, nargs='?', const=1, help="prefactor")
args = parser.parse_args()
print(args)

gap = args.gap
alpha = args.alpha
prefactor = args.pref
Ls = np.array([15,19,25,29,35,39,45,49,55,59])#9,15,19,25,29,35,39,45,49np.array([np.arange(10,71,6)
Ts = np.zeros((Ls.shape[0]))
Emax = prefactor*(Ls**alpha) # energy cutoff
vN_upperbounds = np.zeros(Ls.shape)

for l in range(Ls.shape[0]):
    L = Ls[l]
    N = L**2

    h = np.zeros((N,N))
    for x in range(L-1):
        for y in range(L-1):
            index = L*y+x
            neighbors = [L*y+(x+1),L*(y+1)+x]
            for neighbor in neighbors:
                h[index,neighbor] = -1
                h[neighbor,index] = -1
    for x in range(L):
        for y in range(L):
            index = L*y+x
            if (x+y)%2 == 0:
                h[index,index] = 0.5*gap
            elif (x+y)%2 == 1:
                h[index,index] = -0.5*gap

    hvals,hvecs = np.linalg.eigh(h)

    # ground state properties
    occupations = hvals<0
    E0 = np.sum(occupations*hvals)

    # construct the upper bound
    # hAB and hBC are identical here
    hAB = np.zeros((L*(int(L/2)+1),L*(int(L/2)+1)))
    for x in range(L):
        for y in range(int(L/2)+1):
            index = L*y+x
            if x != L-1 and y != int(L/2):
                neighbors = [L*y+(x+1),L*(y+1)+x]
                for neighbor in neighbors:
                    #plt.plot([x,neighbor%L],[y,(neighbor-neighbor%L)/L])
                    hAB[index,neighbor] = -1
                    hAB[neighbor,index] = -1
            elif x == L-1 and y != int(L/2):
                neighbor = L*(y+1)+x
                #plt.plot([x,neighbor%L],[y,(neighbor-neighbor%L)/L])
                hAB[index,neighbor] = -1
                hAB[neighbor,index] = -1
            elif x != L-1 and y == int(L/2):
                neighbor = L*y+(x+1)
                #plt.plot([x,neighbor%L],[y,(neighbor-neighbor%L)/L])
                hAB[index,neighbor] = -0.5
                hAB[neighbor,index] = -0.5
    #plt.show()


    for x in range(L):
        for y in range(int(L/2)):
            index = L*y+x
            if (x+y)%2 == 0:
                hAB[index,index] = 0.5*gap
            elif (x+y)%2 == 1:
                hAB[index,index] = -0.5*gap
        # half the potential in the final row
        y = int(L/2)
        index = L*y+x
        if (x+y)%2 == 0:
            hAB[index,index] = 0.25*gap
        elif (x+y)%2 == 1:
            hAB[index,index] = -0.25*gap

    hABvals = np.linalg.eigvalsh(hAB)

    E0AB = np.sum((hABvals<-1e-7)*hABvals)
    print(2*E0AB-E0)

    # find temperature T such that hAB has half the energy E0+Emax
    f = lambda T: np.sum(hABvals/(np.exp(hABvals/T)+1)) - 0.5*(E0+Emax[l])
    fprime = lambda T: 0.5*np.sum((hABvals**2)/(1+np.cosh(hABvals/T)))/(T**2)
    sol = root_scalar(f,bracket=[1e-2,10],fprime=fprime,method='brentq',rtol=1e-8)
    T = sol.root
    Ts[l] = T

    # calculate thermal entropy of hAB system at temperature T
    Fermi_AB = 1/(np.exp(hABvals/T)+1)
#    Fermi_AB = 0.5+(1-1e-9)*(Fermi_AB-0.5)
    thermalvN_AB = -np.sum(Fermi_AB*np.log(Fermi_AB))-np.sum((1-Fermi_AB)*np.log(1-Fermi_AB))

    vN_upperbounds[l] = thermalvN_AB

# plt.scatter(Ls,Ts)
# plt.plot(Ls,1/np.log(Ls))
# plt.show()
#
# plt.scatter(Ls,vN_upperbounds/Ls)
# plt.plot(Ls,Ls**0.5)
# plt.xscale('log')
# plt.yscale('log')
# plt.show()

D = np.zeros((2,Ls.shape[0]))
D[0],D[1] = Ls,vN_upperbounds
#filename = 'vN_upperbound_data/20260112_2dfermions_gap'+str(gap)+'_alpha'+str(alpha)+'_pref'+str(prefactor)+'.npy'
filename = 'vN_upperbound_data/20260828_2dfermions_gap'+str(gap)+'_alpha'+str(alpha)+'_pref'+str(prefactor)+'.npy'
np.save(filename,D)

# for gap in [0.0,1.0,2.0,3.0]:
#     filename = 'vN_upperbound_data/20260102_2dfermions_gap'+str(gap)+'_alpha'+str(alpha)+'_pref'+str(prefactor)+'.npy'
#     Ls,vN_upperbounds = np.load(filename)
#     plt.plot(Ls,vN_upperbounds/Ls,label=gap)
# plt.plot(Ls,1.1*np.log(Ls),color='k')
# plt.xscale('log')
# plt.legend()
# plt.show()

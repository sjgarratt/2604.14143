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
parser.add_argument("-L", dest="L", type=int, nargs='?', const=1, help="system size")
parser.add_argument("-gap", dest="gap", type=np.float64, nargs='?', const=1, help="gap")
parser.add_argument("-alpha", dest="alpha", type=np.float64, nargs='?', const=1, help="exponent")
parser.add_argument("-pref", dest="pref", type=np.float64, nargs='?', const=1, help="prefactor")
# parser.add_argument("-mu", dest="mu", type=np.float64, nargs='?', const=1, help="chemical potential")
# parser.add_argument("-ind", dest="ind", type=int, nargs='?', const=1)
args = parser.parse_args()
print(args)
print('rev')

L = args.L
gap = args.gap
N = L**2
prefactor = args.pref
alpha = args.alpha
Emax = prefactor*(L**alpha) # energy cutoff
print('Emax',Emax)
NMC1 = 1000
NMC2 = 100
Asites = np.arange(L*int(L/2))

h = np.zeros((N,N))
for x in range(L):
    for y in range(L):
        index = L*y+x
        neighbors = [L*y+(x+1),L*(y+1)+x]
        if x == L-1 and y != L-1:
            neighbors = [L*(y+1)+x]
        elif x != L-1 and y == L-1:
            neighbors = [L*y+(x+1)]
        elif x == L-1 and y == L-1:
            neighbors = []
        for neighbor in neighbors:
            # plt.plot([x,neighbor%L],[y,(neighbor-neighbor%L)/L])
            h[index,neighbor] = -1
            h[neighbor,index] = -1
    #plt.show()
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
print(-2*E0)

# initalize energy
E = E0*1
Es,vNAs = [],[]
for n1 in range(NMC1):
    if n1 % 100 == 0:
        print(n1)
    # change occupations
    n2 = 0
    while n2 < NMC2: # require at least NMC2 updates, only stop if E < E0+Emax
        # one mode move
        mode = np.random.randint(N)
        dE = (1-2*occupations[mode])*hvals[mode]
        if E + dE < E0 + Emax:
            occupations[mode] = (occupations[mode]+1)%2
            E += dE
            n2 += 1
        # two mode move
        mode1,mode2 = np.random.randint(N),np.random.randint(N)
        if mode1 != mode2:
            dE1 = (1-2*occupations[mode1])*hvals[mode1]
            dE2 = (1-2*occupations[mode2])*hvals[mode2]
            if E + dE1 + dE2 < E0 + Emax:
                occupations[mode1] = (occupations[mode1]+1)%2
                occupations[mode2] = (occupations[mode2]+1)%2
                E += dE1+dE2
                n2 += 1
    Es.append(E-E0)
    # construct the correlation matrix
    occupied = np.argwhere(occupations==1).flatten()
    phi = ((hvecs.T)[occupied])[:,Asites]
    CA = np.matmul(np.conj(phi).T,phi)
    CAvals = np.linalg.eigvalsh(CA)
    CAvals = 0.5+(1-1e-9)*(CAvals-0.5)
    thisvNA = -np.sum(CAvals*np.log(CAvals))-np.sum((1-CAvals)*np.log(1-CAvals))
    # print(thisvNA)
    vNAs.append(thisvNA)
print('vNmax',np.max(vNAs))

# plt.plot(np.sort(vNAs),np.linspace(0,1,NMC1))
# plt.show()

D = np.zeros((2,NMC1))
D[0],D[1] = Es,vNAs
#filename = 'randomeig_2d_data/20260601_randomeig_2dfermions_gap'+str(gap)+'_L'+str(L)+'_alpha'+str(alpha)+'_pref'+str(prefactor)+'.npy'
filename = 'randomeig_2d_data/20260828_randomeig_2dfermions_gap'+str(gap)+'_L'+str(L)+'_alpha'+str(alpha)+'_pref'+str(prefactor)+'.npy'
np.save(filename,D)

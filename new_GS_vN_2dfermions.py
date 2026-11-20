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

gap = 0.0
Ls = np.array([15,19,25,29,35,39,45,49,55,59])
vNAs = np.zeros(Ls.shape)

for l in range(Ls.shape[0]):

    L = Ls[l]
    N = L**2
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

    occupied = np.argwhere(occupations==1).flatten()
    phi = ((hvecs.T)[occupied])[:,Asites]
    CA = np.matmul(np.conj(phi).T,phi)
    CAvals = np.linalg.eigvalsh(CA)
    CAvals = 0.5+(1-1e-9)*(CAvals-0.5)
    vNAs[l] = -np.sum(CAvals*np.log(CAvals))-np.sum((1-CAvals)*np.log(1-CAvals))

plt.plot(Ls,vNAs/Ls)
plt.show()

D = np.zeros((2,Ls.shape[0]))
D[0],D[1] = Ls,vNAs
filename = 'randomeig_2d_data/20260828_quickGSvN_2dfermions_gap'+str(gap)+'.npy'
np.save(filename,D)

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
Ls = np.array([65,129,257,513,1025,2049,4097])
Ts = np.zeros((Ls.shape[0]))
Emax = prefactor*(Ls**alpha) # energy cutoff
vN_upperbounds = np.zeros(Ls.shape)

for l in range(Ls.shape[0]):
    L = Ls[l]

    h = np.zeros((L,L))
    for i in range(L-1):
        h[i,i+1] = -1
        h[i+1,i] = -1
    for i in range(L):
        h[i,i] = 0.5*gap*(2*(i%2)-1)
    hvals,hvecs = np.linalg.eigh(h)
    # plt.plot(np.sort(hvals))
    # plt.show()

    # ground state properties
    occupations = hvals<0
    E0 = np.sum(occupations*hvals)

    # construct the upper bound
    # hAB and hBC are identical here
    hAB = np.zeros((int(L/2)+1,int(L/2)+1))
    for i in range(int(L/2)):
        hAB[i,i+1] = -1
        hAB[i+1,i] = -1
    for i in range(int(L/2)):
        hAB[i,i] = 0.5*gap*(2*(i%2)-1)
    hAB[int(L/2),int(L/2)] = 0.25*gap*(2*(int(L/2)%2)-1)
    hABvals = np.linalg.eigvalsh(hAB)

    # find temperature T such that hAB has half the energy E0+Emax
    f = lambda T: np.sum(hABvals/(np.exp(hABvals/T)+1)) - 0.5*(E0+Emax[l])
    fprime = lambda T: 0.5*np.sum((hABvals**2)/(1+np.cosh(hABvals/T)))/(T**2)
    sol = root_scalar(f,bracket=[1e-2,2],fprime=fprime,method='brentq',rtol=1e-6)
    T = sol.root
    print('T',T,'f(T)',f(T))

    # calculate thermal entropy of hAB system at temperature T
    Fermi_AB = 1/(np.exp(hABvals/T)+1)
    Fermi_AB = 0.5+(1-1e-9)*(Fermi_AB-0.5)

    thermalvN_AB = -np.sum(Fermi_AB*np.log(Fermi_AB))-np.sum((1-Fermi_AB)*np.log(1-Fermi_AB))

    vN_upperbounds[l] = thermalvN_AB

D = np.zeros((2,Ls.shape[0]))
D[0],D[1] = Ls,vN_upperbounds
filename = 'vN_upperbound_data/20260112_1dfermions_gap'+str(gap)+'_alpha'+str(alpha)+'_pref'+str(prefactor)+'.npy'
np.save(filename,D)

# for gap in [1.0,2.0,3.0,4.0,5.0]: #,10.0]:#3.0,4.0,
#     filename = 'vN_upperbound_data/20260112_1dfermions_gap'+str(gap)+'_alpha'+str(alpha)+'_pref'+str(prefactor)+'.npy'
#
#     Ls,vN_upperbounds = np.load(filename)
#     plt.scatter(Ls,vN_upperbounds*gap,label=str(gap))
#
#     linfit = lambda x,offset: (prefactor/gap)*x + offset
#     popt,pcov = curve_fit(linfit, np.log(Ls), vN_upperbounds)
#
#     a,b = np.polyfit(np.log(Ls),vN_upperbounds,deg=1)
#     plt.plot(Ls,(a*np.log(Ls)+b)*gap)
#
#         #plt.plot(Ls,1.5*(Emax/gap)*(1-alpha)*np.log(Ls),color='k')
#         # plt.plot(Ls,np.sqrt(Ls))
#     plt.xscale('log')
#     # plt.plot(Ls,(prefactor/gap)*np.log(Ls/2))
#     # plt.yscale('log')
# plt.legend(frameon=False)
# plt.xlabel(r'$L$')
# plt.ylabel(r'$S^{\mathcal{H}_{AB}}(T^*) \times \Delta$')
# plt.show()

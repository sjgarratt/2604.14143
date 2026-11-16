import numpy as np
from numpy.random import random
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.linalg import eig, expm
import argparse
from matplotlib import rc,cm
from scipy.optimize import root_scalar
plt.rcParams["text.usetex"] = True

fig,axs = plt.subplots()

gap = 0.0
Ls = np.array([15,19,25,29,35,39,45,49,55,59])
GS_vNs = np.load('randomeig_2d_data/20260828_quickGSvN_2dfermions_gap'+str(gap)+'.npy')[1]

Nstates = 5

alpha = 0.0
prefactors = np.array([5.0,3.0,1.0])
colors = ['b','r','g']

for k in range(prefactors.shape[0]):
    prefactor = prefactors[k]
    max_vNs = np.zeros((Nstates,Ls.shape[0]))
    for l in range(Ls.shape[0]):
        L = Ls[l]
        filename = 'randomeig_2d_data/20260828_randomeig_2dfermions_gap'+str(gap)+'_L'+str(L)+'_alpha'+str(alpha)+'_pref'+str(prefactor)+'.npy'
        these_vNs = np.sort(np.load(filename)[1])
        max_vNs[:,l] = np.flip(these_vNs[-Nstates:])

    for n in range(Nstates):
        if n == 0:
            axs.plot(Ls,(max_vNs[n]-GS_vNs)/Ls,alpha=1-n/Nstates,linestyle='None',marker='o',mfc='None',mec=colors[k],mew=1.5,label=r'$\leq$'+str(prefactor))
        elif n != 0:
            axs.plot(Ls,(max_vNs[n]-GS_vNs)/Ls,alpha=1-n/Nstates,linestyle='None',marker='o',mfc='None',mec=colors[k],mew=1.5)
axs.plot(Ls,GS_vNs/Ls,linestyle='None',marker='o',mfc='None',mec='k',mew=1.5)
axs.set_xscale('log')
axs.set_xlim(13,65)
axs.set_ylim(0,1)
axs.minorticks_off()
axs.set_xticks([16,32,64])
axs.set_xticklabels([16,32,64],fontsize=14)
axs.set_yticks([0,0.5,1.0])
axs.set_yticklabels([0,0.5,1.0],fontsize=14)
axs.text(45,0.9,r'$S_A(\psi^H_0)/L$',fontsize=14)
axs.text(34.5,0.04,r'$[S_A(\psi)-S_A(\psi^H_0)]/L$',fontsize=14)
axs.set_xlabel(r'$L$',fontsize=16)
axs.set_ylabel(r'$[S_A(\psi)-S_A(\psi^H_0)]/L \quad \quad S_A(\psi^H_0)/L$',fontsize=14)
axs.legend(title=r'$E-E^H_0$',fontsize=12,title_fontsize=12,bbox_to_anchor=[0.97, 0.6], loc='center right')

plt.savefig('20260828_2dfermions_excitation_contribution.pdf',dpi=600)
plt.show()

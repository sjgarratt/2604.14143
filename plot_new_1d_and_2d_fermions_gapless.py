import argparse

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from numpy.random import random
from scipy.linalg import eig, expm
from scipy.optimize import curve_fit, root_scalar

plt.rcParams["text.usetex"] = True

fig, axs = plt.subplots(2, 1)
fig.set_size_inches(6, 8)

# 1d fermions

Ls = np.array([65, 129, 257, 513, 1025, 2049, 4097])

alpha = 0.0
gap = 0.0
prefactors = np.array([5.0])
colors = ["r"]
max_vNAs = np.zeros((prefactors.shape[0], Ls.shape[0]))

linfit = lambda x, a: 0.5 * x + a

# 1d upper bounds
for i in range(prefactors.shape[0]):
    prefactor = prefactors[i]

    filename = (
        "vN_upperbound_data/20260112_1dfermions_gap"
        + str(gap)
        + "_alpha"
        + str(alpha)
        + "_pref"
        + str(prefactor)
        + ".npy"
    )
    print(prefactor)
    L2s, vN_upperbounds = np.load(filename)
    axs[0].plot(
        L2s,
        vN_upperbounds,
        linestyle="None",
        marker="^",
        mec=colors[i],
        mew=1.5,
        mfc="None",
        markersize=7,
    )

    # linfit = lambda x,offset: (prefactor/gap)*x + offset
    # popt,pcov = curve_fit(linfit, np.log(L2s), vN_upperbounds)

    # a,b = np.polyfit(np.log(Ls)[1:],np.log(max_vNAs[i])[1:],deg=1)
    # axs.plot(Ls,np.exp(a*np.log(Ls)+b),color=colors[i])

    # a,b = np.polyfit(np.log(L2s),vN_upperbounds,deg=1)
    # axs[0].plot(L2s,(a*np.log(L2s)+b)*gap)

    for j in range(Ls.shape[0]):
        # 1d random eigenstates
        L = int(Ls[j])
        print(prefactor)
        filename = (
            "randomeig_1d_data/20260112_randomeig_1dfermions_gap"
            + str(gap)
            + "_L"
            + str(L)
            + "_alpha"
            + str(alpha)
            + "_pref"
            + str(prefactor)
            + ".npy"
        )
        Es, vNAs = np.load(filename)
        max_vNAs[i, j] = np.max(vNAs)

    # a,b = np.polyfit(np.log(Ls)[2:],np.log(max_vNAs[i])[2:],deg=1)
    # axs.plot(Ls,np.exp(a*np.log(Ls)+b),color=colors[i])
    # print(prefactor,a)

    popt, pcov = curve_fit(linfit, np.log(Ls)[2:], np.log(max_vNAs[i])[2:])
    axs[0].plot(Ls, np.exp(0.5 * np.log(Ls) + popt[0]), color="k", zorder=-1)

    axs[0].plot(
        Ls,
        max_vNAs[i],
        linestyle="None",
        marker="o",
        mec=colors[i],
        mew=1.5,
        mfc="None",
        markersize=7,
    )
# 2d fermions

Ls = np.array([15, 19, 25, 29, 35, 39, 45, 49, 55, 59])  # ,4097])
alpha = 1.0
gap = 0.0

linfit = lambda x, a: 1.5 * x + a

# prefactors = np.array([0.2,0.5,1.0,2.0,5.0]) # also have 0.1
prefactors = np.array([1.0])
colors = ["b", "r"]
max_vNAs = np.zeros((prefactors.shape[0], Ls.shape[0]))


# 1d upper bounds
for i in range(prefactors.shape[0]):
    prefactor = prefactors[i]

    filename = (
        "vN_upperbound_data/20260828_2dfermions_gap"
        + str(gap)
        + "_alpha"
        + str(alpha)
        + "_pref"
        + str(prefactor)
        + ".npy"
    )

    L2s, vN_upperbounds = np.load(filename)
    axs[1].plot(
        L2s,
        vN_upperbounds,
        linestyle="None",
        marker="^",
        mec=colors[i],
        mew=1.5,
        mfc="None",
        markersize=7,
    )

    for j in range(Ls.shape[0]):
        # 1d random eigenstates
        L = int(Ls[j])

        filename = (
            "randomeig_2d_data/20260828_randomeig_2dfermions_gap"
            + str(gap)
            + "_L"
            + str(L)
            + "_alpha"
            + str(alpha)
            + "_pref"
            + str(prefactor)
            + ".npy"
        )
        Es, vNAs = np.load(filename)
        max_vNAs[i, j] = np.max(vNAs)

    popt, pcov = curve_fit(linfit, np.log(Ls)[2:], np.log(max_vNAs[i])[2:])
    axs[1].plot(Ls, np.exp(1.5 * np.log(Ls) + popt[0]), color="k", zorder=-1)

    axs[1].plot(
        Ls,
        max_vNAs[i],
        linestyle="None",
        marker="o",
        mec=colors[i],
        mew=1.5,
        mfc="None",
        markersize=7,
    )

axs[0].set_xscale("log")
axs[0].set_yscale("log")
# axs.plot(Ls,0.2*Ls**0.5,color='k')
# axs.legend(frameon=False)
axs[0].set_xlabel(r"$L$", fontsize=16)
axs[0].set_ylabel(r"$S_A(\psi)$", fontsize=16)

axs[0].set_ylim(3.8, 90)
axs[0].set_xlim(46, 5300)
# axs[0].set_xticks([50, 500, 5000])
axs[0].set_xticks([64, 512, 4096])
axs[0].set_xticklabels(["64", "512", "4096"], fontsize=14)
#
axs[0].set_yticks([4, 8, 16, 32, 64])
axs[0].set_yticklabels(["4", "8", "16", "32", "64"], fontsize=14)

axs[1].set_xscale("log")
axs[1].set_yscale("log")
axs[1].set_ylim(15, 430)
axs[1].set_xlim(14, 66)
# axs.plot(Ls,Ls**1.5,color='k')
# axs.legend(frameon=False)
axs[1].set_xlabel(r"$L$", fontsize=16)
axs[1].set_ylabel(r"$S_A(\psi)$", fontsize=16)

axs[1].set_xticks([16, 32, 64])
axs[1].set_xticklabels(["16", "32", "64"], fontsize=14)

axs[1].set_yticks([16, 32, 64, 128, 256])
axs[1].set_yticklabels(["16", "32", "64", "128", "256"], fontsize=14)

# extra labels
axs[0].text(58, 61, r"$d=1$", fontsize=16, color="r")
axs[1].text(15.1, 285, r"$d=2$", fontsize=16, color="b")

axs[0].text(1200, 18, r"$\sim L^{1 / 2}$", fontsize=16)
axs[1].text(29.8, 43, r"$\sim L^{3 / 2}$", fontsize=16)

axs[0].plot(
    [],
    [],
    color="k",
    linestyle="None",
    marker="^",
    mec="k",
    mew=1.5,
    mfc="None",
    label="Upper bound",
)
axs[0].plot(
    [],
    [],
    color="k",
    linestyle="None",
    marker="o",
    mec="k",
    mew=1.5,
    mfc="None",
    label="Eigenstate",
)
axs[0].legend(frameon=False, fontsize=14, loc="lower right")

axs[0].minorticks_off()
axs[1].minorticks_off()

plt.tight_layout()
plt.savefig("20260828_1d2dfermions_gapless.pdf", dpi=600)
plt.show()

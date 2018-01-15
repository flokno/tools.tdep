#! /usr/bin/env python3.6

import os
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser as argpars

parser = argpars(description='perform 2D EoS fit (BM)')
parser.add_argument('infiles', type=str, nargs = '+', help='input files')
parser.add_argument('-p', '--plot', type=str, nargs='?', help='plot the distribution of a/c',
                    const = True, default = None)
parser.add_argument('-o', '--output', type=str, default='infile.eosfit', help='output file')
args = parser.parse_args()

def eosfit_2D(As, Cs, Es):
    import subprocess as sp
    As = np.array(As)
    Cs = np.array(Cs)
    Es = np.array(Es)
    
    Vs   = np.sqrt(3) / 2 *As**2 * Cs
    # eta = a/c
    #   a = eta * c
    #   V = \sqrt(3)/2 * eta**2 * c**3
    etas =  As / Cs
    
    with open(args.output, 'w') as f:
        f.write(f'{len(Es):d}')
        for ii, EE in enumerate(Es):
            f.write(f'\n{Vs[ii]:15.8f}  {EE:15.8f} {etas[ii]:15.8f}')
    
    # execute eosfit
    outp = sp.Popen(['eosfit', '--dim', '2'], encoding='utf-8',
            stdout=sp.PIPE).communicate()[0].split('\n')
    
    # parse output
    for ii, line in enumerate(outp):
        if 'Birch-Murnaghan' in line:
            vnames = [ s  for s in outp[ii+1].split()]
            E0, V0, eta0, B0, B0p, C0, C1, C2, C3 = [float(s) for s in outp[ii+2].split()]
        if 'Total RMS' in line:
            rms = line
    
    # restore a/c from V/eta:
    c0 = (2 * V0 / np.sqrt(3) / eta0**2)**(1/3)
    a0 = eta0 * c0
    
    return a0, c0, V0, eta0, E0, rms


# Parse the data
# Prepare collections of data:
coll_a = []
coll_c = []
coll_T = [] 
coll_E = [] 

for infile in args.infiles:
    # remove hashes
    with open(infile, 'r') as f:
        n_lines_1 = [  int(s) for s in f.readline().split()[:2]]
        a, c      = [float(s) for s in f.readline().split()[:2]]
        n_lines_2 = [  int(s) for s in f.readline().split()[:2]]
        nT        = n_lines_2[0]
        Ts        = np.zeros(nT)
        Es        = np.zeros(nT)
        for ii in range(nT):
            Ts[ii], Es[ii] = [float(s) for s in f.readline().split()[:2]]
        #
        # Append everything to the collections
        coll_a.append(a)
        coll_c.append(c)
        coll_T.append(Ts)
        coll_E.append(Es)
#
#
# make numpy arrays
As = np.array(coll_a)
Cs = np.array(coll_c)
Ts = np.array(coll_T)
Es = np.array(coll_E)


# temperature dependent stuff:
aT   = np.empty(nT)
cT   = np.empty(nT)
etaT = np.empty(nT)
VT   = np.empty(nT)
ET   = np.empty(nT)
rmsT = np.empty(nT, dtype=str)

print(f'Fit for {nT} temperatures')
for ii in range(nT):
    aT[ii], cT[ii], VT[ii], etaT[ii], ET[ii], rmsT[ii] = eosfit_2D(As, Cs, Es[:, ii])
print('..done')

with open('eosfit.dat', 'w') as f:
    f.write(f'#   {"T":6s} {"a(T)":15s} {"c(T)":15s} {"V(T)":15s} {"eta(T)":15s} {"E(T)":15s}')
    for ii in range(nT):
        f.write(f'\n {Ts[0, ii]:6.1f} {aT[ii]:15.7e} {cT[ii]:15.7e}')
        f.write(f' {VT[ii]:15.7e} {etaT[ii]:15.7e} {ET[ii]:15.7e}')


# print(f'Output values:')
# print(f'  V0:   {V0}')
# print(f'  eta0: {eta0}')
# print(f'  a0:   {a0}')
# print(f'  c0:   {c0}')
# print(f'{rms}')

if args.plot:
    fig, ax = plt.subplots(1)
    ax.plot(As, Cs, 'k*')
    ax.set_xlabel('a')
    ax.set_ylabel(r'c')
    plt.savefig(f'a_c.pdf')

    fig, axs = plt.subplots(4, sharex='col')
    axs[0].plot(Ts[0, :], aT)
    axs[1].plot(Ts[0, :], cT)
    axs[2].plot(Ts[0, :], etaT)
#    axs[3].plot(Ts[0, :], VT)
    axs[3].plot(Ts[0, :], (VT - VT[0]) / VT[0])
    # ax.plot(Ts[0, :], cT)
    # ax.plot(Ts[0, :], VT)
    axs[3].set_xlabel('T [K]')
    axs[0].set_ylabel(r'a')
    axs[1].set_ylabel(r'c')
    axs[2].set_ylabel(r'eta')
    axs[3].set_ylabel(r'V(T) / V(0) - 1')
    plt.savefig(f'overview.pdf')
    #plt.show()

    fig, ax = plt.subplots(1)
    ax.plot(Ts[0, :], (VT - VT[0]) / VT[0])
    ax.set_xlabel('T [K]')
    ax.set_ylabel(r'V(T) / V(0) - 1')
    plt.savefig(f'V_T.pdf')

    fig, ax = plt.subplots(1)
    ax.plot(aT, cT, 'k')
    plt.savefig(f'ac_T.pdf')


    # plt.show()
    # ax.legend()
    # ax.set_title(f'')


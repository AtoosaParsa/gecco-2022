## plot everything after loading data from the pickled files for MOO

import time, array, random, copy, math
import numpy as np
from deap import algorithms, base, benchmarks, tools, creator
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import random
import pickle
from os.path import exists
import os

from ConfigPlot import ConfigPlot_DiffStiffness2, ConfigPlot_DiffStiffness3
from MD_functions import MD_VibrSP_ConstV_Yfixed_DiffK, FIRE_YFixed_ConstV_DiffK, MD_VibrSP_ConstV_Yfixed_DiffK2
from MD_functions import MD_VibrSP_ConstV_Yfixed_DiffK_Freqs,  MD_VibrSP_ConstV_Yfixed_DiffK2_Freqs
from DynamicalMatrix import DM_mass_DiffK_Yfixed

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def evaluate(indices):       
    #%% Initial Configuration
    m1 = 1
    m2 = 10
    k1 = 1.
    k2 = 10. 
    
    n_col = 6
    n_row = 5
    N = n_col*n_row
    
    Nt_fire = 1e6
    
    dt_ratio = 40
    Nt_SD = 1e5
    Nt_MD = 1e5
    
    
    dphi_index = -1
    dphi = 10**dphi_index
    d0 = 0.1
    d_ratio = 1.1
    Lx = d0*n_col
    Ly = (n_row-1)*np.sqrt(3)/2*d0+d0
    
    x0 = np.zeros(N)
    y0 = np.zeros(N)
    
    phi0 = N*np.pi*d0**2/4/(Lx*Ly)
    d_ini = d0*np.sqrt(1+dphi/phi0)
    D = np.zeros(N)+d_ini
    #D = np.zeros(N)+d0 
    
    x0 = np.zeros(N)
    y0 = np.zeros(N)
    for i_row in range(1, n_row+1):
        for i_col in range(1, n_col+1):
            ind = (i_row-1)*n_col+i_col-1
            if i_row%2 == 1:
                x0[ind] = (i_col-1)*d0
            else:
                x0[ind] = (i_col-1)*d0+0.5*d0
            y0[ind] = (i_row-1)*np.sqrt(3)/2*d0
    y0 = y0+0.5*d0
    
    mass = np.zeros(N) + 1
    k_list = np.array([k1, k2, k1 * k2 / (k1 + k2)])
    k_type = indices #np.zeros(N, dtype=np.int8)
    #k_type[indices] = 1
    
    # Steepest Descent to get energy minimum      
    #x_ini, y_ini, p_now = MD_YFixed_ConstV_SP_SD_DiffK(Nt_SD, N, x0, y0, D, mass, Lx, Ly, k_list, k_type)
    x_ini, y_ini, p_now = FIRE_YFixed_ConstV_DiffK(Nt_fire, N, x0, y0, D, mass, Lx, Ly, k_list, k_type)
    # skip the steepest descent for now to save time
    #x_ini = x0
    #y_ini = y0

    # calculating the bandgap - no need to do this in this problem
    #w, v = DM_mass_DiffK_Yfixed(N, x_ini, y_ini, D, mass, Lx, 0.0, Ly, k_list, k_type)
    #w = np.real(w)
    #v = np.real(v)
    #freq = np.sqrt(np.absolute(w))
    #ind_sort = np.argsort(freq)
    #freq = freq[ind_sort]
    #v = v[:, ind_sort]
    #ind = freq > 1e-4
    #eigen_freq = freq[ind]
    #eigen_mode = v[:, ind]
    #w_delta = eigen_freq[1:] - eigen_freq[0:-1]
    #index = np.argmax(w_delta)
    #F_low_exp = eigen_freq[index]
    #F_high_exp = eigen_freq[index+1]

    #print("specs:")

    #print(F_low_exp)
    #print(F_high_exp)
    #print(max(w_delta))


    # specify the input ports and the output port
    SP_scheme = 0
    digit_in = SP_scheme//2
    digit_out = SP_scheme-2*digit_in 

    ind_in1 = int((n_col+1)/2)+digit_in - 1
    ind_in2 = ind_in1 + 2
    ind_out = int(N-int((n_col+1)/2)+digit_out)
    ind_fix = int((n_row+1)/2)*n_col-int((n_col+1)/2)

    B = 1
    Nt = 1e4 # it was 1e5 before, I reduced it to run faster

    # we are designing an and gait at this frequency
    Freq_Vibr = 7

    # case 1, input [1, 1]
    Amp_Vibr1 = 1e-2
    Amp_Vibr2 = 1e-2

    # changed the resonator to one in MD_functions file and vibrations in x direction
    freq_fft, fft_in1, fft_in2, fft_x_out, fft_y_out, mean_cont, nt_rec, Ek_now, Ep_now, cont_now = MD_VibrSP_ConstV_Yfixed_DiffK(k_list, k_type, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

    ind = np.where(freq_fft>Freq_Vibr)
    index_=ind[0][0]
    # fft of the output port at the driving frequency
    out1 = fft_x_out[index_-1] + (fft_x_out[index_]-fft_x_out[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    # fft of input1 at driving frequency
    inp1 = fft_in1[index_-1] + (fft_in1[index_]-fft_in1[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    # fft of input2 at driving frequency
    inp2 = fft_in2[index_-1] + (fft_in2[index_]-fft_in2[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    gain1 = out1/(inp1+inp2)

    # case 2, input [1, 0]
    Amp_Vibr1 = 1e-2
    Amp_Vibr2 = 0

    # changed the resonator to one in MD_functions file and vibrations in x direction
    freq_fft, fft_in1, fft_in2, fft_x_out, fft_y_out, mean_cont, nt_rec, Ek_now, Ep_now, cont_now = MD_VibrSP_ConstV_Yfixed_DiffK(k_list, k_type, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

    ind = np.where(freq_fft>Freq_Vibr)
    index_=ind[0][0]
    # fft of the output port at the driving frequency
    out2 = fft_x_out[index_-1] + (fft_x_out[index_]-fft_x_out[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    # fft of input1 at driving frequency
    inp1 = fft_in1[index_-1] + (fft_in1[index_]-fft_in1[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    # fft of input2 at driving frequency
    inp2 = fft_in2[index_-1] + (fft_in2[index_]-fft_in2[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    gain2 = out2/(inp1+inp2)

    # case 3, input [0, 1]
    Amp_Vibr1 = 0
    Amp_Vibr2 = 1e-2

    # changed the resonator to one in MD_functions file and vibrations in x direction
    freq_fft, fft_in1, fft_in2, fft_x_out, fft_y_out, mean_cont, nt_rec, Ek_now, Ep_now, cont_now = MD_VibrSP_ConstV_Yfixed_DiffK(k_list, k_type, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

    ind = np.where(freq_fft>Freq_Vibr)
    index_=ind[0][0]
    # fft of the output port at the driving frequency
    out3 = fft_x_out[index_-1] + (fft_x_out[index_]-fft_x_out[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    # fft of input1 at driving frequency
    inp1 = fft_in1[index_-1] + (fft_in1[index_]-fft_in1[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    # fft of input2 at driving frequency
    inp2 = fft_in2[index_-1] + (fft_in2[index_]-fft_in2[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    gain3 = out3/(inp1+inp2)
    
    andness = 2*gain1/(gain2+gain3)

    
    
    
    # we are designing an and gait at this frequency
    Freq_Vibr = 10

    # case 1, input [1, 1]
    Amp_Vibr1 = 1e-2
    Amp_Vibr2 = 1e-2

    # changed the resonator to one in MD_functions file and vibrations in x direction
    freq_fft, fft_in1, fft_in2, fft_x_out, fft_y_out, mean_cont, nt_rec, Ek_now, Ep_now, cont_now = MD_VibrSP_ConstV_Yfixed_DiffK(k_list, k_type, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

    ind = np.where(freq_fft>Freq_Vibr)
    index_=ind[0][0]
    # fft of the output port at the driving frequency
    out1 = fft_x_out[index_-1] + (fft_x_out[index_]-fft_x_out[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    # fft of input1 at driving frequency
    inp1 = fft_in1[index_-1] + (fft_in1[index_]-fft_in1[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    # fft of input2 at driving frequency
    inp2 = fft_in2[index_-1] + (fft_in2[index_]-fft_in2[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    gain1 = out1/(inp1+inp2)

    # case 2, input [1, 0]
    Amp_Vibr1 = 1e-2
    Amp_Vibr2 = 0

    # changed the resonator to one in MD_functions file and vibrations in x direction
    freq_fft, fft_in1, fft_in2, fft_x_out, fft_y_out, mean_cont, nt_rec, Ek_now, Ep_now, cont_now = MD_VibrSP_ConstV_Yfixed_DiffK(k_list, k_type, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

    ind = np.where(freq_fft>Freq_Vibr)
    index_=ind[0][0]
    # fft of the output port at the driving frequency
    out2 = fft_x_out[index_-1] + (fft_x_out[index_]-fft_x_out[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    # fft of input1 at driving frequency
    inp1 = fft_in1[index_-1] + (fft_in1[index_]-fft_in1[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    # fft of input2 at driving frequency
    inp2 = fft_in2[index_-1] + (fft_in2[index_]-fft_in2[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    gain2 = out2/(inp1+inp2)

    # case 3, input [0, 1]
    Amp_Vibr1 = 0
    Amp_Vibr2 = 1e-2

    # changed the resonator to one in MD_functions file and vibrations in x direction
    freq_fft, fft_in1, fft_in2, fft_x_out, fft_y_out, mean_cont, nt_rec, Ek_now, Ep_now, cont_now = MD_VibrSP_ConstV_Yfixed_DiffK(k_list, k_type, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

    ind = np.where(freq_fft>Freq_Vibr)
    index_=ind[0][0]
    # fft of the output port at the driving frequency
    out3 = fft_x_out[index_-1] + (fft_x_out[index_]-fft_x_out[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    # fft of input1 at driving frequency
    inp1 = fft_in1[index_-1] + (fft_in1[index_]-fft_in1[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    # fft of input2 at driving frequency
    inp2 = fft_in2[index_-1] + (fft_in2[index_]-fft_in2[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    gain3 = out3/(inp1+inp2)
    
    XOR = (gain2+gain3)/(2*gain1)

    print("done eval", flush=True)

    return andness, XOR

def showPacking(indices):
    k1 = 1.
    k2 = 10.

    n_col = 6
    n_row = 5
    N = n_col*n_row

    m1=1
    m2=10
    
    dphi_index = -1
    dphi = 10**dphi_index
    d0 = 0.1
    Lx = d0*n_col
    Ly = (n_row-1)*np.sqrt(3)/2*d0+d0
    
    x0 = np.zeros(N)
    y0 = np.zeros(N)
    
    phi0 = N*np.pi*d0**2/4/(Lx*Ly)
    d_ini = d0*np.sqrt(1+dphi/phi0)
    D = np.zeros(N)+d_ini
    #D = np.zeros(N)+d0 
    
    x0 = np.zeros(N)
    y0 = np.zeros(N)
    for i_row in range(1, n_row+1):
        for i_col in range(1, n_col+1):
            ind = (i_row-1)*n_col+i_col-1
            if i_row%2 == 1:
                x0[ind] = (i_col-1)*d0
            else:
                x0[ind] = (i_col-1)*d0+0.5*d0
            y0[ind] = (i_row-1)*np.sqrt(3)/2*d0
    y0 = y0+0.5*d0
    
    mass = np.zeros(N) + 1
    k_list = np.array([k1, k2, k1 * k2 / (k1 + k2)])
    k_type = indices #np.zeros(N, dtype=np.int8)
    #k_type[indices] = 1

    # specify the input ports and the output port
    SP_scheme = 0
    digit_in = SP_scheme//2
    digit_out = SP_scheme-2*digit_in 

    ind_in1 = int((n_col+1)/2)+digit_in - 1
    ind_in2 = ind_in1 + 2
    ind_out = int(N-int((n_col+1)/2)+digit_out)
    ind_fix = int((n_row+1)/2)*n_col-int((n_col+1)/2)

    # show packing
    ConfigPlot_DiffStiffness3(N, x0, y0, D, [Lx,Ly], k_type, 0, '/Users/atoosa/Desktop/results/packing.pdf', ind_in1, ind_in2, ind_out)

def plotInOut_and(indices):

    #%% Initial Configuration
    k1 = 1.
    k2 = 10. 
    m1 = 1
    m2 = 10
    
    n_col = 6
    n_row = 5
    N = n_col*n_row
    
    Nt_fire = 1e6
    
    dt_ratio = 40
    Nt_SD = 1e5
    Nt_MD = 1e5
    
    
    dphi_index = -1
    dphi = 10**dphi_index
    d0 = 0.1
    d_ratio = 1.1
    Lx = d0*n_col
    Ly = (n_row-1)*np.sqrt(3)/2*d0+d0
    
    x0 = np.zeros(N)
    y0 = np.zeros(N)
    
    phi0 = N*np.pi*d0**2/4/(Lx*Ly)
    d_ini = d0*np.sqrt(1+dphi/phi0)
    D = np.zeros(N)+d_ini
    #D = np.zeros(N)+d0 
    
    x0 = np.zeros(N)
    y0 = np.zeros(N)
    for i_row in range(1, n_row+1):
        for i_col in range(1, n_col+1):
            ind = (i_row-1)*n_col+i_col-1
            if i_row%2 == 1:
                x0[ind] = (i_col-1)*d0
            else:
                x0[ind] = (i_col-1)*d0+0.5*d0
            y0[ind] = (i_row-1)*np.sqrt(3)/2*d0
    y0 = y0+0.5*d0
    
    mass = np.zeros(N) + 1
    k_list = np.array([k1, k2, k1 * k2 / (k1 + k2)])
    k_type = indices #np.zeros(N, dtype=np.int8)
    #k_type[indices] = 1
    
    # Steepest Descent to get energy minimum      
    #x_ini, y_ini, p_now = MD_YFixed_ConstV_SP_SD_DiffK(Nt_SD, N, x0, y0, D, mass, Lx, Ly, k_list, k_type)
    x_ini, y_ini, p_now = FIRE_YFixed_ConstV_DiffK(Nt_fire, N, x0, y0, D, mass, Lx, Ly, k_list, k_type)

    # skip the steepest descent for now to save time
    #x_ini = x0
    #y_ini = y0

    # calculating the bandgap - no need to do this in this problem
    w, v = DM_mass_DiffK_Yfixed(N, x_ini, y_ini, D, mass, Lx, 0.0, Ly, k_list, k_type)
    w = np.real(w)
    v = np.real(v)
    freq = np.sqrt(np.absolute(w))
    ind_sort = np.argsort(freq)
    freq = freq[ind_sort]
    v = v[:, ind_sort]
    ind = freq > 1e-4
    eigen_freq = freq[ind]
    eigen_mode = v[:, ind]
    w_delta = eigen_freq[1:] - eigen_freq[0:-1]
    index = np.argmax(w_delta)
    F_low_exp = eigen_freq[index]
    F_high_exp = eigen_freq[index+1]

    plt.figure(figsize=(6.4,4.8))
    ax = plt.axes()
    plt.scatter(np.arange(0, len(eigen_freq)), eigen_freq, marker='x', color='blue')
    plt.xlabel(r"Index $(k)$", fontsize=16)
    plt.ylabel(r"Frequency $(\omega)$", fontsize=16)
    plt.title("Frequency Spectrum", fontsize=16, fontweight="bold")
    plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
    props = dict(facecolor='green', alpha=0.1)
    myText = r'$\omega_{low}=$'+"{:.2f}".format(F_low_exp)+"\n"+r'$\omega_{high}=$'+"{:.2f}".format(F_high_exp)+"\n"+r'$\Delta \omega=$'+"{:.2f}".format(max(w_delta))
    #plt.text(0.78, 0.15, myText, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16, bbox=props)
    plt.text(0.2, 0.8, myText, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16, bbox=props)
    plt.hlines(y=7, xmin=0, xmax=50, linewidth=1, linestyle='dashdot', color='limegreen', alpha=0.9)
    plt.hlines(y=10, xmin=0, xmax=50, linewidth=1, linestyle='dotted', color='brown', alpha=0.9)
    plt.text(51, 5, '$\omega=7$', fontsize=12, color='limegreen', alpha=0.9)
    plt.text(51, 12, '$\omega=10$', fontsize=12, color='brown', alpha=0.9)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()

    print("specs:")

    print(F_low_exp)
    print(F_high_exp)
    print(max(w_delta))


    # specify the input ports and the output port
    SP_scheme = 0
    digit_in = SP_scheme//2
    digit_out = SP_scheme-2*digit_in 

    ind_in1 = int((n_col+1)/2)+digit_in - 1
    ind_in2 = ind_in1 + 2
    ind_out = int(N-int((n_col+1)/2)+digit_out)
    ind_fix = int((n_row+1)/2)*n_col-int((n_col+1)/2)

    B = 1
    Nt = 1e4 # it was 1e5 before, i reduced it to run faster

    # we are designing an and gait at this frequency
    Freq_Vibr = 7

    # case 0, input [0, 0]
    fig = plt.figure(figsize=(6.4,4.8))
    ax = plt.axes()
    plt.hlines(y=0, xmin=0, xmax=30, color='green', label='Input1', linestyle='dotted')
    plt.hlines(y=0, xmin=0, xmax=30, color='blue', label='Input2', linestyle=(0, (3, 5, 1, 5)))
    plt.hlines(y=0, xmin=0, xmax=30, color='red', label='Output', linestyle='dashed')
    plt.xlabel("Frequency", fontsize=16)
    plt.ylabel("Amplitude of FFT", fontsize=16)
    plt.title("Logic Gate Response - input = 00", fontsize=16)
    plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
    plt.legend(loc='upper right', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0, 0.005)
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(6.4,4.8))
    ax = plt.axes()
    plt.hlines(y=0, xmin=0, xmax=10000, color='green', label='Input1', linestyle='solid')
    plt.hlines(y=0, xmin=0, xmax=10000, color='blue', label='Input2', linestyle='dotted')
    plt.hlines(y=0, xmin=0, xmax=10000, color='red', label='Output', linestyle='solid')
    plt.xlabel("Time Steps", fontsize=16)
    plt.ylabel("Displacement", fontsize=16)
    plt.title("Logic Gate Response - input = 00", fontsize=16)
    plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
    plt.legend(loc='upper right', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(-0.0100, 0.0100)
    plt.tight_layout()
    plt.show()


    # case 1, input [1, 1]
    Amp_Vibr1 = 1e-2
    Amp_Vibr2 = 1e-2

    # changed the resonator to one in MD_functions file and vibrations in x direction
    freq_fft, fft_in1, fft_in2, fft_x_out, fft_y_out, mean_cont, nt_rec, Ek_now, Ep_now, cont_now = MD_VibrSP_ConstV_Yfixed_DiffK(k_list, k_type, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

    ind = np.where(freq_fft>Freq_Vibr)
    index_=ind[0][0]
    # fft of the output port at the driving frequency
    out1 = fft_x_out[index_-1] + (fft_x_out[index_]-fft_x_out[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    # fft of input1 at driving frequency
    inp1 = fft_in1[index_-1] + (fft_in1[index_]-fft_in1[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    # fft of input2 at driving frequency
    inp2 = fft_in2[index_-1] + (fft_in2[index_]-fft_in2[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    gain1 = out1/(inp1+inp2)
    

    fig = plt.figure(figsize=(6.4,4.8))
    ax = plt.axes()
    plt.plot(freq_fft, fft_in1, color='green', label='Input1', linestyle='dotted')
    plt.plot(freq_fft, fft_in2, color='blue', label='Input2', linestyle=(0, (3, 5, 1, 5)))
    plt.plot(freq_fft, fft_x_out, color='red', label='Output', linestyle='dashed')
    plt.xlabel("Frequency", fontsize=16)
    plt.ylabel("Amplitude of FFT", fontsize=16)
    plt.title("Logic Gate Response - input = 11", fontsize=16)
    plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
    plt.legend(loc='upper right', fontsize=16)
    #plt.axvline(x=Freq_Vibr, color='purple', linestyle='solid', alpha=0.5)
    myText = 'Gain='+"{:.3f}".format(gain1)
    plt.text(0.5, 0.9, myText, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

    x_in1, x_in2, x_out = MD_VibrSP_ConstV_Yfixed_DiffK2(k_list, k_type, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

    fig = plt.figure(figsize=(6.4,4.8))
    ax = plt.axes()
    plt.plot(x_in1, color='green', label='Input1', linestyle='solid')
    plt.plot(x_in2, color='blue', label='Input2', linestyle='dotted')
    plt.plot(x_out-np.mean(x_out, axis=0), color='red', label='Output', linestyle='solid')
    plt.xlabel("Time Steps", fontsize=16)
    plt.ylabel("Displacement", fontsize=16)
    plt.title("Logic Gate Response - input = 11", fontsize=16)
    plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
    plt.legend(loc='upper right', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

    # case 2, input [1, 0]
    Amp_Vibr1 = 1e-2
    Amp_Vibr2 = 0

    # changed the resonator to one in MD_functions file and vibrations in x direction
    freq_fft, fft_in1, fft_in2, fft_x_out, fft_y_out, mean_cont, nt_rec, Ek_now, Ep_now, cont_now = MD_VibrSP_ConstV_Yfixed_DiffK(k_list, k_type, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

    ind = np.where(freq_fft>Freq_Vibr)
    index_=ind[0][0]
    # fft of the output port at the driving frequency
    out2 = fft_x_out[index_-1] + (fft_x_out[index_]-fft_x_out[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    # fft of input1 at driving frequency
    inp1 = fft_in1[index_-1] + (fft_in1[index_]-fft_in1[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    # fft of input2 at driving frequency
    inp2 = fft_in2[index_-1] + (fft_in2[index_]-fft_in2[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    gain2 = out2/(inp1+inp2)

    fig = plt.figure(figsize=(6.4,4.8))
    ax = plt.axes()
    plt.plot(freq_fft, fft_in1, color='green', label='Input1', linestyle='dotted')
    plt.plot(freq_fft, fft_in2, color='blue', label='Input2', linestyle=(0, (3, 5, 1, 5)))
    plt.plot(freq_fft, fft_x_out, color='red', label='Output', linestyle='dashed')
    plt.xlabel("Frequency", fontsize=16)
    plt.ylabel("Amplitude of FFT", fontsize=16)
    plt.title("Logic Gate Response - input = 10", fontsize=16)
    plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
    plt.legend(loc='upper right', fontsize=16)
    #plt.axvline(x=Freq_Vibr, color='purple', linestyle='solid', alpha=0.5)
    myText = 'Gain='+"{:.3f}".format(gain2)
    plt.text(0.5, 0.9, myText, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

    x_in1, x_in2, x_out = MD_VibrSP_ConstV_Yfixed_DiffK2(k_list, k_type, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)
    
    fig = plt.figure(figsize=(6.4,4.8))
    ax = plt.axes()
    plt.plot(x_in1, color='green', label='Input1', linestyle='solid')
    plt.plot(x_in2, color='blue', label='Input2', linestyle='dotted')
    plt.plot(x_out-np.mean(x_out, axis=0), color='red', label='Output', linestyle='solid')
    plt.xlabel("Time Steps", fontsize=16)
    plt.ylabel("Displacement", fontsize=16)
    plt.title("Logic Gate Response - input = 10", fontsize=16)
    plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
    plt.legend(loc='upper right', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

    # case 3, input [0, 1]
    Amp_Vibr1 = 0
    Amp_Vibr2 = 1e-2

    # changed the resonator to one in MD_functions file and vibrations in x direction
    freq_fft, fft_in1, fft_in2, fft_x_out, fft_y_out, mean_cont, nt_rec, Ek_now, Ep_now, cont_now = MD_VibrSP_ConstV_Yfixed_DiffK(k_list, k_type, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

    ind = np.where(freq_fft>Freq_Vibr)
    index_=ind[0][0]
    # fft of the output port at the driving frequency
    out3 = fft_x_out[index_-1] + (fft_x_out[index_]-fft_x_out[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    # fft of input1 at driving frequency
    inp1 = fft_in1[index_-1] + (fft_in1[index_]-fft_in1[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    # fft of input2 at driving frequency
    inp2 = fft_in2[index_-1] + (fft_in2[index_]-fft_in2[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    gain3 = out3/(inp1+inp2)

    fig = plt.figure(figsize=(6.4,4.8))
    ax = plt.axes()
    plt.plot(freq_fft, fft_in1, color='green', label='Input1', linestyle='dotted')
    plt.plot(freq_fft, fft_in2, color='blue', label='Input2', linestyle=(0, (3, 5, 1, 5)))
    plt.plot(freq_fft, fft_x_out, color='red', label='Output', linestyle='dashed')
    plt.xlabel("Frequency", fontsize=16)
    plt.ylabel("Amplitude of FFT", fontsize=16)
    plt.title("Logic Gate Response - input = 01", fontsize=16)
    plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
    plt.legend(loc='upper right', fontsize=16)
    #plt.axvline(x=Freq_Vibr, color='purple', linestyle='solid', alpha=0.5)
    myText = 'Gain='+"{:.3f}".format(gain3)
    plt.text(0.5, 0.9, myText, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

    x_in1, x_in2, x_out = MD_VibrSP_ConstV_Yfixed_DiffK2(k_list, k_type, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

    fig = plt.figure(figsize=(6.4,4.8))
    ax = plt.axes()
    plt.plot(x_in1, color='green', label='Input1', linestyle='solid')
    plt.plot(x_in2, color='blue', label='Input2', linestyle='dotted')
    plt.plot(x_out-np.mean(x_out, axis=0), color='red', label='Output', linestyle='solid')
    plt.xlabel("Time Steps", fontsize=16)
    plt.ylabel("Displacement", fontsize=16)
    plt.title("Logic Gate Response - input = 01", fontsize=16)
    plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
    plt.legend(loc='upper right', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

    print("gain1:")
    print(gain1)
    print("gain2:")
    print(gain2)
    print("gain3:")
    print(gain3)

    andness = 2*gain1/(gain2+gain3)
    
    return andness

def plotInOut_xor(indices):

    #%% Initial Configuration
    k1 = 1.
    k2 = 10. 
    m1 = 1
    m2 = 10
    
    n_col = 6
    n_row = 5
    N = n_col*n_row
    
    Nt_fire = 1e6
    
    dt_ratio = 40
    Nt_SD = 1e5
    Nt_MD = 1e5
    
    
    dphi_index = -1
    dphi = 10**dphi_index
    d0 = 0.1
    d_ratio = 1.1
    Lx = d0*n_col
    Ly = (n_row-1)*np.sqrt(3)/2*d0+d0
    
    x0 = np.zeros(N)
    y0 = np.zeros(N)
    
    phi0 = N*np.pi*d0**2/4/(Lx*Ly)
    d_ini = d0*np.sqrt(1+dphi/phi0)
    D = np.zeros(N)+d_ini
    #D = np.zeros(N)+d0 
    
    x0 = np.zeros(N)
    y0 = np.zeros(N)
    for i_row in range(1, n_row+1):
        for i_col in range(1, n_col+1):
            ind = (i_row-1)*n_col+i_col-1
            if i_row%2 == 1:
                x0[ind] = (i_col-1)*d0
            else:
                x0[ind] = (i_col-1)*d0+0.5*d0
            y0[ind] = (i_row-1)*np.sqrt(3)/2*d0
    y0 = y0+0.5*d0
    
    mass = np.zeros(N) + 1
    k_list = np.array([k1, k2, k1 * k2 / (k1 + k2)])
    k_type = indices #np.zeros(N, dtype=np.int8)
    #k_type[indices] = 1
    
    # Steepest Descent to get energy minimum      
    #x_ini, y_ini, p_now = MD_YFixed_ConstV_SP_SD_DiffK(Nt_SD, N, x0, y0, D, mass, Lx, Ly, k_list, k_type)
    x_ini, y_ini, p_now = FIRE_YFixed_ConstV_DiffK(Nt_fire, N, x0, y0, D, mass, Lx, Ly, k_list, k_type)

    # skip the steepest descent for now to save time
    #x_ini = x0
    #y_ini = y0

    # calculating the bandgap - no need to do this in this problem
    w, v = DM_mass_DiffK_Yfixed(N, x_ini, y_ini, D, mass, Lx, 0.0, Ly, k_list, k_type)
    w = np.real(w)
    v = np.real(v)
    freq = np.sqrt(np.absolute(w))
    ind_sort = np.argsort(freq)
    freq = freq[ind_sort]
    v = v[:, ind_sort]
    ind = freq > 1e-4
    eigen_freq = freq[ind]
    eigen_mode = v[:, ind]
    w_delta = eigen_freq[1:] - eigen_freq[0:-1]
    index = np.argmax(w_delta)
    F_low_exp = eigen_freq[index]
    F_high_exp = eigen_freq[index+1]

    plt.figure(figsize=(6.4,4.8))
    ax = plt.axes()
    plt.scatter(np.arange(0, len(eigen_freq)), eigen_freq, marker='x', color='blue')
    plt.xlabel(r"Index $(k)$", fontsize=16)
    plt.ylabel(r"Frequency $(\omega)$", fontsize=16)
    plt.title("Frequency Spectrum", fontsize=16, fontweight="bold")
    plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
    props = dict(facecolor='green', alpha=0.1)
    myText = r'$\omega_{low}=$'+"{:.2f}".format(F_low_exp)+"\n"+r'$\omega_{high}=$'+"{:.2f}".format(F_high_exp)+"\n"+r'$\Delta \omega=$'+"{:.2f}".format(max(w_delta))
    plt.text(0.78, 0.15, myText, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16, bbox=props)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()

    print("specs:")

    print(F_low_exp)
    print(F_high_exp)
    print(max(w_delta))


    # specify the input ports and the output port
    SP_scheme = 0
    digit_in = SP_scheme//2
    digit_out = SP_scheme-2*digit_in 

    ind_in1 = int((n_col+1)/2)+digit_in - 1
    ind_in2 = ind_in1 + 2
    ind_out = int(N-int((n_col+1)/2)+digit_out)
    ind_fix = int((n_row+1)/2)*n_col-int((n_col+1)/2)

    B = 1
    Nt = 1e4 # it was 1e5 before, i reduced it to run faster

    # we are designing an and gait at this frequency
    Freq_Vibr = 10

    # case 1, input [1, 1]
    Amp_Vibr1 = 1e-2
    Amp_Vibr2 = 1e-2

    # changed the resonator to one in MD_functions file and vibrations in x direction
    freq_fft, fft_in1, fft_in2, fft_x_out, fft_y_out, mean_cont, nt_rec, Ek_now, Ep_now, cont_now = MD_VibrSP_ConstV_Yfixed_DiffK(k_list, k_type, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

    ind = np.where(freq_fft>Freq_Vibr)
    index_=ind[0][0]
    # fft of the output port at the driving frequency
    out1 = fft_x_out[index_-1] + (fft_x_out[index_]-fft_x_out[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    # fft of input1 at driving frequency
    inp1 = fft_in1[index_-1] + (fft_in1[index_]-fft_in1[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    # fft of input2 at driving frequency
    inp2 = fft_in2[index_-1] + (fft_in2[index_]-fft_in2[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    gain1 = out1/(inp1+inp2)
    

    fig = plt.figure(figsize=(6.4,4.8))
    ax = plt.axes()
    plt.plot(freq_fft, fft_in1, color='green', label='Input1', linestyle='dotted')
    plt.plot(freq_fft, fft_in2, color='blue', label='Input2', linestyle=(0, (3, 5, 1, 5)))
    plt.plot(freq_fft, fft_x_out, color='red', label='Output', linestyle='dashed')
    plt.xlabel("Frequency", fontsize=16)
    plt.ylabel("Amplitude of FFT", fontsize=16)
    plt.title("Logic Gate Response - input = 11", fontsize=16)
    plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
    plt.legend(loc='upper right', fontsize=16)
    #plt.axvline(x=Freq_Vibr, color='purple', linestyle='solid', alpha=0.5)
    myText = 'Gain='+"{:.3f}".format(gain1)
    plt.text(0.5, 0.9, myText, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

    x_in1, x_in2, x_out = MD_VibrSP_ConstV_Yfixed_DiffK2(k_list, k_type, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

    fig = plt.figure(figsize=(6.4,4.8))
    ax = plt.axes()
    plt.plot(x_in1, color='green', label='Input1', linestyle='solid')
    plt.plot(x_in2, color='blue', label='Input2', linestyle='dotted')
    plt.plot(x_out-np.mean(x_out, axis=0), color='red', label='Output', linestyle='solid')
    plt.xlabel("Time Steps", fontsize=16)
    plt.ylabel("Displacement", fontsize=16)
    plt.title("Logic Gate Response - input = 11", fontsize=16)
    plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
    plt.legend(loc='upper right', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

    # case 2, input [1, 0]
    Amp_Vibr1 = 1e-2
    Amp_Vibr2 = 0

    # changed the resonator to one in MD_functions file and vibrations in x direction
    freq_fft, fft_in1, fft_in2, fft_x_out, fft_y_out, mean_cont, nt_rec, Ek_now, Ep_now, cont_now = MD_VibrSP_ConstV_Yfixed_DiffK(k_list, k_type, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

    ind = np.where(freq_fft>Freq_Vibr)
    index_=ind[0][0]
    # fft of the output port at the driving frequency
    out2 = fft_x_out[index_-1] + (fft_x_out[index_]-fft_x_out[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    # fft of input1 at driving frequency
    inp1 = fft_in1[index_-1] + (fft_in1[index_]-fft_in1[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    # fft of input2 at driving frequency
    inp2 = fft_in2[index_-1] + (fft_in2[index_]-fft_in2[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    gain2 = out2/(inp1+inp2)

    fig = plt.figure(figsize=(6.4,4.8))
    ax = plt.axes()
    plt.plot(freq_fft, fft_in1, color='green', label='Input1', linestyle='dotted')
    plt.plot(freq_fft, fft_in2, color='blue', label='Input2', linestyle=(0, (3, 5, 1, 5)))
    plt.plot(freq_fft, fft_x_out, color='red', label='Output', linestyle='dashed')
    plt.xlabel("Frequency", fontsize=16)
    plt.ylabel("Amplitude of FFT", fontsize=16)
    plt.title("Logic Gate Response - input = 10", fontsize=16)
    plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
    plt.legend(loc='upper right', fontsize=16)
    #plt.axvline(x=Freq_Vibr, color='purple', linestyle='solid', alpha=0.5)
    myText = 'Gain='+"{:.3f}".format(gain2)
    plt.text(0.5, 0.9, myText, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

    x_in1, x_in2, x_out = MD_VibrSP_ConstV_Yfixed_DiffK2(k_list, k_type, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)
    
    fig = plt.figure(figsize=(6.4,4.8))
    ax = plt.axes()
    plt.plot(x_in1, color='green', label='Input1', linestyle='solid')
    plt.plot(x_in2, color='blue', label='Input2', linestyle='dotted')
    plt.plot(x_out-np.mean(x_out, axis=0), color='red', label='Output', linestyle='solid')
    plt.xlabel("Time Steps", fontsize=16)
    plt.ylabel("Displacement", fontsize=16)
    plt.title("Logic Gate Response - input = 10", fontsize=16)
    plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
    plt.legend(loc='upper right', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

    # case 3, input [0, 1]
    Amp_Vibr1 = 0
    Amp_Vibr2 = 1e-2

    # changed the resonator to one in MD_functions file and vibrations in x direction
    freq_fft, fft_in1, fft_in2, fft_x_out, fft_y_out, mean_cont, nt_rec, Ek_now, Ep_now, cont_now = MD_VibrSP_ConstV_Yfixed_DiffK(k_list, k_type, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

    ind = np.where(freq_fft>Freq_Vibr)
    index_=ind[0][0]
    # fft of the output port at the driving frequency
    out3 = fft_x_out[index_-1] + (fft_x_out[index_]-fft_x_out[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    # fft of input1 at driving frequency
    inp1 = fft_in1[index_-1] + (fft_in1[index_]-fft_in1[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    # fft of input2 at driving frequency
    inp2 = fft_in2[index_-1] + (fft_in2[index_]-fft_in2[index_-1])*((Freq_Vibr-freq_fft[index_-1])/(freq_fft[index_]-freq_fft[index_-1]))

    gain3 = out3/(inp1+inp2)

    fig = plt.figure(figsize=(6.4,4.8))
    ax = plt.axes()
    plt.plot(freq_fft, fft_in1, color='green', label='Input1', linestyle='dotted')
    plt.plot(freq_fft, fft_in2, color='blue', label='Input2', linestyle=(0, (3, 5, 1, 5)))
    plt.plot(freq_fft, fft_x_out, color='red', label='Output', linestyle='dashed')
    plt.xlabel("Frequency", fontsize=16)
    plt.ylabel("Amplitude of FFT", fontsize=16)
    plt.title("Logic Gate Response - input = 01", fontsize=16)
    plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
    plt.legend(loc='upper right', fontsize=16)
    #plt.axvline(x=Freq_Vibr, color='purple', linestyle='solid', alpha=0.5)
    myText = 'Gain='+"{:.3f}".format(gain3)
    plt.text(0.5, 0.9, myText, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

    x_in1, x_in2, x_out = MD_VibrSP_ConstV_Yfixed_DiffK2(k_list, k_type, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

    fig = plt.figure(figsize=(6.4,4.8))
    ax = plt.axes()
    plt.plot(x_in1, color='green', label='Input1', linestyle='solid')
    plt.plot(x_in2, color='blue', label='Input2', linestyle='dotted')
    plt.plot(x_out-np.mean(x_out, axis=0), color='red', label='Output', linestyle='solid')
    plt.xlabel("Time Steps", fontsize=16)
    plt.ylabel("Displacement", fontsize=16)
    plt.title("Logic Gate Response - input = 01", fontsize=16)
    plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
    plt.legend(loc='upper right', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

    print("gain1:")
    print(gain1)
    print("gain2:")
    print(gain2)
    print("gain3:")
    print(gain3)

    XOR = (gain2+gain3)/(2*gain1)
    
    return XOR

def plotInOut_adder(indices):

    #%% Initial Configuration
    k1 = 1.
    k2 = 10. 
    m1 = 1
    m2 = 10
    
    n_col = 6
    n_row = 5
    N = n_col*n_row
    
    Nt_fire = 1e6
    
    dt_ratio = 40
    Nt_SD = 1e5
    Nt_MD = 1e5
    
    
    dphi_index = -1
    dphi = 10**dphi_index
    d0 = 0.1
    d_ratio = 1.1
    Lx = d0*n_col
    Ly = (n_row-1)*np.sqrt(3)/2*d0+d0
    
    x0 = np.zeros(N)
    y0 = np.zeros(N)
    
    phi0 = N*np.pi*d0**2/4/(Lx*Ly)
    d_ini = d0*np.sqrt(1+dphi/phi0)
    D = np.zeros(N)+d_ini
    #D = np.zeros(N)+d0 
    
    x0 = np.zeros(N)
    y0 = np.zeros(N)
    for i_row in range(1, n_row+1):
        for i_col in range(1, n_col+1):
            ind = (i_row-1)*n_col+i_col-1
            if i_row%2 == 1:
                x0[ind] = (i_col-1)*d0
            else:
                x0[ind] = (i_col-1)*d0+0.5*d0
            y0[ind] = (i_row-1)*np.sqrt(3)/2*d0
    y0 = y0+0.5*d0
    
    mass = np.zeros(N) + 1
    k_list = np.array([k1, k2, k1 * k2 / (k1 + k2)])
    k_type = indices #np.zeros(N, dtype=np.int8)
    #k_type[indices] = 1
    
    # Steepest Descent to get energy minimum      
    #x_ini, y_ini, p_now = MD_YFixed_ConstV_SP_SD_DiffK(Nt_SD, N, x0, y0, D, mass, Lx, Ly, k_list, k_type)
    x_ini, y_ini, p_now = FIRE_YFixed_ConstV_DiffK(Nt_fire, N, x0, y0, D, mass, Lx, Ly, k_list, k_type)

    # skip the steepest descent for now to save time
    #x_ini = x0
    #y_ini = y0

    # calculating the bandgap - no need to do this in this problem
    w, v = DM_mass_DiffK_Yfixed(N, x_ini, y_ini, D, mass, Lx, 0.0, Ly, k_list, k_type)
    w = np.real(w)
    v = np.real(v)
    freq = np.sqrt(np.absolute(w))
    ind_sort = np.argsort(freq)
    freq = freq[ind_sort]
    v = v[:, ind_sort]
    ind = freq > 1e-4
    eigen_freq = freq[ind]
    eigen_mode = v[:, ind]
    w_delta = eigen_freq[1:] - eigen_freq[0:-1]
    index = np.argmax(w_delta)
    F_low_exp = eigen_freq[index]
    F_high_exp = eigen_freq[index+1]

    plt.figure(figsize=(6.4,4.8))
    ax = plt.axes()
    plt.scatter(np.arange(0, len(eigen_freq)), eigen_freq, marker='x', color='blue')
    plt.xlabel(r"Index $(k)$", fontsize=16)
    plt.ylabel(r"Frequency $(\omega)$", fontsize=16)
    plt.title("Frequency Spectrum", fontsize=16, fontweight="bold")
    plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
    props = dict(facecolor='green', alpha=0.1)
    myText = r'$\omega_{low}=$'+"{:.2f}".format(F_low_exp)+"\n"+r'$\omega_{high}=$'+"{:.2f}".format(F_high_exp)+"\n"+r'$\Delta \omega=$'+"{:.2f}".format(max(w_delta))
    #plt.text(0.78, 0.15, myText, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16, bbox=props)
    plt.text(0.2, 0.8, myText, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16, bbox=props)
    plt.hlines(y=7, xmin=0, xmax=50, linewidth=1, linestyle='dashdot', color='limegreen', alpha=0.9)
    plt.hlines(y=10, xmin=0, xmax=50, linewidth=1, linestyle='dotted', color='brown', alpha=0.9)
    plt.text(51, 5, '$\omega=7$', fontsize=12, color='limegreen', alpha=0.9)
    plt.text(51, 12, '$\omega=10$', fontsize=12, color='brown', alpha=0.9)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()

    print("specs:")

    print(F_low_exp)
    print(F_high_exp)
    print(max(w_delta))


    # specify the input ports and the output port
    SP_scheme = 0
    digit_in = SP_scheme//2
    digit_out = SP_scheme-2*digit_in 

    ind_in1 = int((n_col+1)/2)+digit_in - 1
    ind_in2 = ind_in1 + 2
    ind_out = int(N-int((n_col+1)/2)+digit_out)
    ind_fix = int((n_row+1)/2)*n_col-int((n_col+1)/2)

    B = 1
    Nt = 1e4 # it was 1e5 before, i reduced it to run faster

    # we are designing an and gait at this frequency
    Freq_Vibr1 = 7
    Freq_Vibr2 = 10

    # case 1, input [1, 1]
    Amp_Vibr1 = 1e-2
    Amp_Vibr2 = 1e-2

    # changed the resonator to one in MD_functions file and vibrations in x direction
    freq_fft, fft_in1, fft_in2, fft_x_out, fft_y_out, mean_cont, nt_rec, Ek_now, Ep_now, cont_now = MD_VibrSP_ConstV_Yfixed_DiffK_Freqs(k_list, k_type, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr1, Freq_Vibr2, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)
    

    fig = plt.figure(figsize=(6.4,4.8))
    ax = plt.axes()
    plt.plot(freq_fft, fft_in1, color='green', label='Input1', linestyle='dotted')
    plt.plot(freq_fft, fft_in2, color='blue', label='Input2', linestyle=(0, (3, 5, 1, 5)))
    plt.plot(freq_fft, fft_x_out, color='red', label='Output', linestyle='dashed')
    plt.xlabel("Frequency", fontsize=16)
    plt.ylabel("Amplitude of FFT", fontsize=16)
    plt.title("Logic Gate Response - input = 11", fontsize=16)
    plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
    plt.legend(loc='upper right', fontsize=16)
    #plt.axvline(x=Freq_Vibr, color='purple', linestyle='solid', alpha=0.5)
    #myText = 'Gain='+"{:.3f}".format(gain1)
    #plt.text(0.5, 0.9, myText, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

    x_in1, x_in2, x_out = MD_VibrSP_ConstV_Yfixed_DiffK2_Freqs(k_list, k_type, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr1, Freq_Vibr2, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

    fig = plt.figure(figsize=(6.4,4.8))
    ax = plt.axes()
    plt.plot(x_in1, color='green', label='Input1', linestyle='solid')
    plt.plot(x_in2, color='blue', label='Input2', linestyle='dotted')
    plt.plot(x_out-np.mean(x_out, axis=0), color='red', label='Output', linestyle='solid')
    plt.xlabel("Time Steps", fontsize=16)
    plt.ylabel("Displacement", fontsize=16)
    plt.title("Logic Gate Response - input = 11", fontsize=16)
    plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
    plt.legend(loc='upper right', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

    # case 2, input [1, 0]
    Amp_Vibr1 = 1e-2
    Amp_Vibr2 = 0

    # changed the resonator to one in MD_functions file and vibrations in x direction
    freq_fft, fft_in1, fft_in2, fft_x_out, fft_y_out, mean_cont, nt_rec, Ek_now, Ep_now, cont_now = MD_VibrSP_ConstV_Yfixed_DiffK_Freqs(k_list, k_type, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr1, Freq_Vibr2, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

    fig = plt.figure(figsize=(6.4,4.8))
    ax = plt.axes()
    plt.plot(freq_fft, fft_in1, color='green', label='Input1', linestyle='dotted')
    plt.plot(freq_fft, fft_in2, color='blue', label='Input2', linestyle=(0, (3, 5, 1, 5)))
    plt.plot(freq_fft, fft_x_out, color='red', label='Output', linestyle='dashed')
    plt.xlabel("Frequency", fontsize=16)
    plt.ylabel("Amplitude of FFT", fontsize=16)
    plt.title("Logic Gate Response - input = 10", fontsize=16)
    plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
    plt.legend(loc='upper right', fontsize=16)
    #plt.axvline(x=Freq_Vibr, color='purple', linestyle='solid', alpha=0.5)
    #myText = 'Gain='+"{:.3f}".format(gain2)
    #plt.text(0.5, 0.9, myText, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

    x_in1, x_in2, x_out = MD_VibrSP_ConstV_Yfixed_DiffK2_Freqs(k_list, k_type, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr1, Freq_Vibr2, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)
    
    fig = plt.figure(figsize=(6.4,4.8))
    ax = plt.axes()
    plt.plot(x_in1, color='green', label='Input1', linestyle='solid')
    plt.plot(x_in2, color='blue', label='Input2', linestyle='dotted')
    plt.plot(x_out-np.mean(x_out, axis=0), color='red', label='Output', linestyle='solid')
    plt.xlabel("Time Steps", fontsize=16)
    plt.ylabel("Displacement", fontsize=16)
    plt.title("Logic Gate Response - input = 10", fontsize=16)
    plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
    plt.legend(loc='upper right', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

    # case 3, input [0, 1]
    Amp_Vibr1 = 0
    Amp_Vibr2 = 1e-2

    # changed the resonator to one in MD_functions file and vibrations in x direction
    freq_fft, fft_in1, fft_in2, fft_x_out, fft_y_out, mean_cont, nt_rec, Ek_now, Ep_now, cont_now = MD_VibrSP_ConstV_Yfixed_DiffK_Freqs(k_list, k_type, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr1, Freq_Vibr2, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)


    fig = plt.figure(figsize=(6.4,4.8))
    ax = plt.axes()
    plt.plot(freq_fft, fft_in1, color='green', label='Input1', linestyle='dotted')
    plt.plot(freq_fft, fft_in2, color='blue', label='Input2', linestyle=(0, (3, 5, 1, 5)))
    plt.plot(freq_fft, fft_x_out, color='red', label='Output', linestyle='dashed')
    plt.xlabel("Frequency", fontsize=16)
    plt.ylabel("Amplitude of FFT", fontsize=16)
    plt.title("Logic Gate Response - input = 01", fontsize=16)
    plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
    plt.legend(loc='upper right', fontsize=16)
    #plt.axvline(x=Freq_Vibr, color='purple', linestyle='solid', alpha=0.5)
    #myText = 'Gain='+"{:.3f}".format(gain3)
    #plt.text(0.5, 0.9, myText, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

    x_in1, x_in2, x_out = MD_VibrSP_ConstV_Yfixed_DiffK2_Freqs(k_list, k_type, B, Nt, N, x_ini, y_ini, D, mass, [Lx, Ly], Freq_Vibr1, Freq_Vibr2, Amp_Vibr1, ind_in1, Amp_Vibr2, ind_in2, ind_out)

    fig = plt.figure(figsize=(6.4,4.8))
    ax = plt.axes()
    plt.plot(x_in1, color='green', label='Input1', linestyle='solid')
    plt.plot(x_in2, color='blue', label='Input2', linestyle='dotted')
    plt.plot(x_out-np.mean(x_out, axis=0), color='red', label='Output', linestyle='solid')
    plt.xlabel("Time Steps", fontsize=16)
    plt.ylabel("Displacement", fontsize=16)
    plt.title("Logic Gate Response - input = 01", fontsize=16)
    plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
    plt.legend(loc='upper right', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return 0

creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 30)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)

toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selNSGA2)

indices = pickle.load(open('indices.pickle', 'rb'))
outputs = pickle.load(open('outputs.pickle', 'rb'))

ploting = []
# plot the pareto front again without annotation
plt.figure(figsize=(4,4))
for i in indices:
    ind = np.array(i)
    showPacking(ind)

print(outputs)
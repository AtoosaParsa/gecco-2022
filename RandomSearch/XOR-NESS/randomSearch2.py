from switch_binary import switch
import matplotlib.pyplot as plt
import random
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import pickle
from scipy.stats import norm
import operator
from ConfigPlot import ConfigPlot_DiffStiffness3

with open('outs.dat', "rb") as f:
    outs = pickle.load(f)
f.close()

with open('samples.dat', "rb") as f:
    samples = pickle.load(f)
f.close()

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

print("done", flush=True)
fig = plt.figure(figsize=(6.4,4.8))
ax = plt.axes()
n, bins, patches = plt.hist(x=outs, bins='auto', color='#0504aa', alpha=0.7, cumulative=False)#, grid=True)

# fitting a normal distribution
mu, std = norm.fit(outs)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 1000)
p = norm.pdf(x, mu, std)

#plt.plot(x, p, linewidth=2)
myText = "Mean={:.3f}, STD={:.3f}".format(mu, std)
plt.text(0.5, 0.9, myText, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='medium', color='g')
plt.xlabel('XOR-Ness')
plt.ylabel('Counts')
plt.title('Random Search', fontsize='medium')
#plt.xlim([0, 8])
plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
#plt.show()
plt.savefig("histogram2.jpg", dpi=300)

sortedList = list(zip(*sorted(zip(samples,outs), key=operator.itemgetter(1))))

showPacking(sortedList[0][0])
print(sortedList[1][0])

showPacking(sortedList[0][-1])
print(sortedList[1][-1])

showPacking(sortedList[0][-2])
print(sortedList[1][-2])

# multi-objective optimization to evolve AND and XOR gates in the same material

# imports for DEAP
import time, array, random, copy, math
import numpy as np
from deap import algorithms, base, benchmarks, tools, creator
import matplotlib.pyplot as plt
import seaborn
#seaborn.set(style='whitegrid')
import pandas as pd

# imports for the simulator
from ConfigPlot import ConfigPlot_EigenMode_DiffMass, ConfigPlot_YFixed_rec, ConfigPlot_DiffMass_SP
from MD_functions import MD_VibrSP_ConstV_Yfixed_DiffK
from MD_functions import FIRE_YFixed_ConstV_DiffK
from DynamicalMatrix import DM_mass_DiffK_Yfixed
from plot_functions import Line_single, Line_multi
from ConfigPlot import ConfigPlot_DiffStiffness
import random
import matplotlib.pyplot as plt
import pickle
from os.path import exists
from scoop import futures
import multiprocessing
import os

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
    Nt = 1e4 # it was 1e5 before, i reduced it to run faster

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
    
#cleaning up  the data files
try:
    os.remove("results.pickle")
except OSError:
    pass
try:
    os.remove("logs.pickle")
except OSError:
    pass
try:
    os.remove("hofs.pickle")
except OSError:
    pass
try:
    os.remove("hostfile")
except OSError:
    pass
try:
    os.remove("scoop-python.sh")
except OSError:
    pass

# start of the optimization:
random.seed(a=42)

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

# parallelization?
toolbox.register("map", futures.map)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)
# also save the population of each generation
stats.register("pop", copy.deepcopy)

def main():
    toolbox.pop_size = 50
    toolbox.max_gen = 250
    toolbox.mut_prob = 0.8

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    hof = tools.HallOfFame(1, similar=np.array_equal) #can change the size

    def run_ea(toolbox, stats=stats, verbose=True, hof=hof):
        pop = toolbox.population(n=toolbox.pop_size)
        pop = toolbox.select(pop, len(pop))
        return algorithms.eaMuPlusLambda(pop, toolbox, mu=toolbox.pop_size, 
                                        lambda_=toolbox.pop_size, 
                                        cxpb=1-toolbox.mut_prob, #: no cross-over?
                                        mutpb=toolbox.mut_prob, 
                                        stats=stats, 
                                        ngen=toolbox.max_gen, 
                                        verbose=verbose,
                                        halloffame=hof)

    res,log = run_ea(toolbox, stats=stats, verbose=True, hof=hof)

    return res, log, hof


if __name__ == '__main__':

    print("starting")
    res, log, hof = main()
    print("done")
    pickle.dump(res, open('results.pickle', 'wb'))
    pickle.dump(log, open('logs.pickle', 'wb'))
    pickle.dump(hof, open('hofs.pickle', 'wb'))

    # print info for best solution found:
    #print("-----")
    #print(len(hof))
    #best = hof.items[0]
    #print("-- Best Individual = ", best)
    #print("-- Best Fitness = ", best.fitness.values)


    # print hall of fame members info:
    #print("- Best solutions are:")
    # for i in range(HALL_OF_FAME_SIZE):
    #     print(i, ": ", hof.items[i].fitness.values[0], " -> ", hof.items[i])
    #print("Hall of Fame Individuals = ", *hof.items, sep="\n")

    #fronts = tools.emo.sortLogNondominated(res, len(res))

    #for i,inds in enumerate(fronts):
    #    counter = 0
    #    for ind in inds:
    #        counter += 1
    #        if counter == 1 or counter == 15 or counter==30:
    #            print("####")
    #            evaluateAndPlot(ind)

    #logbook.record(gen=0, evals=30, **record)
    avg = log.select("avg")
    std = log.select("std")
    avg_stack = np.stack(avg, axis=0)
    avg_f1 = avg_stack[:, 0]
    avg_f2 = avg_stack[:, 1]
    std_stack = np.stack(std, axis=0)
    std_f1 = std_stack[:, 0]
    std_f2 = std_stack[:, 1]

    plt.figure(figsize=(6.4,4.8))
    plt.plot(avg_f1, color='blue')
    plt.fill_between(list(range(0, toolbox.max_gen+1)), avg_f1-std_f1, avg_f1+std_f1, color='cornflowerblue', alpha=0.2)
    plt.xlabel("Generations")
    plt.ylabel("Average Fitness")
    plt.title("Average Fitness of Individuals in the Population - F1", fontsize='small')
    plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.show()
    plt.savefig("avg_F1.jpg", dpi = 300)

    plt.figure(figsize=(6.4,4.8))
    plt.plot(avg_f2, color='blue')
    plt.fill_between(list(range(0, toolbox.max_gen+1)), avg_f2-std_f2, avg_f2+std_f2, color='cornflowerblue', alpha=0.2)
    plt.xlabel("Generations")
    plt.ylabel("Average Fitness")
    plt.title("Average Fitness of Individuals in the Population - F2", fontsize='small')
    plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.show()
    plt.savefig("avg_F2.jpg", dpi = 300)

    seaborn.set(style='whitegrid')

    plot_colors = seaborn.color_palette("Set1", n_colors=10)
    fig, ax = plt.subplots(1, figsize=(4,4))
    for i,inds in enumerate(fronts):
        par = [toolbox.evaluate(ind) for ind in inds]
        print("fronts:")
        print(par)
        df = pd.DataFrame(par)
        df.plot(ax=ax, kind='scatter', 
                        x=df.columns[0], y=df.columns[1], 
                        color=plot_colors[i])
    plt.xlabel('$f_1(\mathbf{x})$');plt.ylabel('$f_2(\mathbf{x})$');
    plt.title("Pareto Front", fontsize='small')

    plt.show()

    plt.savefig("paretoFront.jpg", dpi = 300)
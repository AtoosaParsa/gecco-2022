from switch_binary import switch
import matplotlib.pyplot as plt
import random
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import pickle

m1 = 1
m2 = 10

N = 30
N_light = 9

samples = []
print("sampling", flush=True)
for i in range(0, 5001):
    samples.append(np.random.randint(low=0, high=2, size=N))

print("sampling done", flush=True)

num_cores = multiprocessing.cpu_count()
outs = Parallel(n_jobs=num_cores)(delayed(switch.evaluate)(m1, m2, N_light, samples[i]) for i in range(0, 5001))

print("done", flush=True)

f = open('outs.dat', 'ab')
pickle.dump(outs , f)
f.close()

f = open('samples.dat', 'ab')
pickle.dump(samples , f)
f.close()

n, bins, patches = plt.hist(x=outs, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)#, grid=True)
plt.xlabel('Andness')
plt.ylabel('Counts')
plt.title('Random Search')
#plt.xlim([0, 8])
plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
plt.show()
plt.savefig("histogram.jpg", dpi=300)

import numpy as np
import problexity as pb
from problexity.ComplexityCalculator import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

C_IDX = [0,0,0,0,0,1,1,1,2,2,2,2,2,2,3,3,3,4,4,4,5,5]
R_IDX = [0,0,0,0,1,1,2,2,2,3,3,3]

# Configure problems
q = 16
repeats = 10
n_samples_range = np.linspace(1000,50,q).astype(int)
n_features_range = np.linspace(100,2,q).astype(int)

# MEASURES x SAMPLES x FEATURES x REPEATS
c_times = np.load('c_times.npy')
r_times = np.load('r_times.npy')

# Gather measures
clf_measures = pb.classification.__all__
reg_measures = pb.regression.__all__

clf_scale = np.linspace(0,1,len(clf_measures))
reg_scale = np.linspace(0,1,len(reg_measures))

fig, ax = plt.subplots(2,1,figsize=(14,15/1.618), sharey=True)

for measure_idx, (measure_name, x) in enumerate(zip(clf_measures, clf_scale)):
    print(measure_name, x)
    data = c_times[measure_idx].ravel()
    xloc = np.ones_like(data) * x
    print(data.shape)
    
    bp = ax[0].boxplot(data, positions=[x], widths=[.02], sym='')
    
    #for key in bp:
    #    for v in bp[key]:
    #        v.set_color(C_COLORS[C_IDX[measure_idx]])
    #        v.set_linewidth(2)

for measure_idx, (measure_name, x) in enumerate(zip(reg_measures, reg_scale)):
    print(measure_name, x)
    data = r_times[measure_idx].ravel()
    xloc = np.ones_like(data) * x
    print(data.shape)
    sdata = data - np.min(data)
    sdata = sdata / np.max(sdata) 
    
    #ax[1].scatter(xloc, data, s=100, alpha=.01)
    
    bp = ax[1].boxplot(data, positions=[x], widths=[.02], sym='')
    #for key in bp:
    #    for v in bp[key]:
    #        v.set_color(R_COLORS[R_IDX[measure_idx]])
    #        v.set_linewidth(2)


ax[0].set_xticks(clf_scale, clf_measures)
ax[1].set_xticks(reg_scale, reg_measures)

ax[0].set_title('Classification task')
ax[1].set_title('Regression task')

for i in range(2):
    ax[i].set_xlim(-.05,1.05)
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    ax[i].set_ylabel('Time [seconds]')
    ax[i].set_xlabel('Measure')
    ax[i].set_yscale('log')
    ax[i].grid(ls=":")

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('figures/time.png')
plt.savefig('figures/time.eps')
import numpy as np
import problexity as pb
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Configure problems
q = 32
repeats = 10
n_samples_range = np.linspace(1000,50,q).astype(int)
n_features_range = np.linspace(100,2,q).astype(int)

# Gather measures
clf_measures = pb.classification.__all__
reg_measures = pb.regression.__all__

# MEASURES x SAMPLES x FEATURES x REPEATS
c_times = np.load('c_times.npy')
r_times = np.load('r_times.npy')

#c_times[:,0,:,:] = 0

vmin = np.min([np.min(c_times), np.min(r_times)])
vmax = np.max([np.max(c_times), np.max(r_times)])

# Plot definition

# Analyze classification
for measure_idx, measure_name in enumerate(clf_measures): 
    # Gather measure
    measure = getattr(pb.classification, measure_name)
    
    data = np.median(c_times[measure_idx], axis=-1)
    
    print(measure)
    print(data.shape)
    
    fig, ax = plt.subplots(1,1,figsize=(7,7))
    im = ax.imshow(data, vmin=0, cmap='twilight')
    ax.set_title('clf - %s' % measure.__name__)
    ax.set_xlabel('number of features')
    ax.set_ylabel('number of samples')
    ax.set_xlim(q-.5, -.5)
    ax.set_ylim(q-.5, -.5)
    
    ax.set_yticks(list(range(q)), n_samples_range)
    ax.set_xticks(list(range(q)), n_features_range)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)
    
    plt.savefig('figures/clf_%i.png' % measure_idx)
    plt.savefig('figures/clf_%i.eps' % measure_idx)
    plt.savefig('foo.png')
    
# Analyze regression
for measure_idx, measure_name in enumerate(reg_measures): 
    # Gather measure
    measure = getattr(pb.regression, measure_name)
    
    data = np.median(r_times[measure_idx], axis=-1)
    
    print(measure)
    print(data.shape)
    
    fig, ax = plt.subplots(1,1,figsize=(7,7))
    im = ax.imshow(data, vmin=0, cmap='twilight')
    ax.set_title('reg - %s' % measure.__name__)
    ax.set_xlabel('number of features')
    ax.set_ylabel('number of samples')
    ax.set_xlim(q-.5, -.5)
    ax.set_ylim(q-.5, -.5)
    
    ax.set_yticks(list(range(q)), n_samples_range)
    ax.set_xticks(list(range(q)), n_features_range)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)
    
    plt.savefig('figures/reg_%i.png' % measure_idx)
    plt.savefig('figures/reg_%i.eps' % measure_idx)
    plt.savefig('foo.png')
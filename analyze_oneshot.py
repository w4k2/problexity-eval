import numpy as np
import problexity as pb
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Configure problems
q = 16
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
fig, aax = plt.subplots(9,4,figsize=(9,12), sharex=True, sharey=True)
aax = aax.ravel()
cnt = 0

# Analyze classification
for measure_idx, measure_name in enumerate(clf_measures): 
    # Gather measure
    measure = getattr(pb.classification, measure_name)
    
    data = np.median(c_times[measure_idx], axis=-1)
    
    print(measure)
    print(data.shape)
    
    ax = aax[cnt]
    im = ax.imshow(data, vmin=0, cmap='twilight')
    ax.set_title('clf - %s' % measure.__name__)
    
    if cnt % 4 == 0:
        ax.set_ylabel('#samples')
    #ax.set_xlabel('number of features')
    ax.set_xlim(q-.5, -.5)
    ax.set_ylim(q-.5, -.5)
    
    #ax.set_yticks(list(range(q)), n_samples_range)
    #ax.set_xticks(list(range(q)), n_features_range)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)

    plt.colorbar(im, cax=cax)
    
    plt.tight_layout() 
    plt.savefig('foo.png')
    
    cnt+=1

# Analyze regression
for measure_idx, measure_name in enumerate(reg_measures): 
    # Gather measure
    measure = getattr(pb.regression, measure_name)
    
    data = np.median(r_times[measure_idx], axis=-1)
    
    print(measure)
    print(data.shape)
    
    ax = aax[cnt]
    im = ax.imshow(data, vmin=0, cmap='twilight')
    ax.set_title('reg - %s' % measure.__name__)
    if cnt % 4 == 0:
        ax.set_ylabel('#samples')
        
    if cnt // 8 == 7:
        ax.set_xlabel('#features')
    ax.set_xlim(q-.5, -.5)
    ax.set_ylim(q-.5, -.5)
    
    #ax.set_yticks(list(range(q)), n_samples_range)
    #ax.set_xticks(list(range(q)), n_features_range)
  
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)

    plt.colorbar(im, cax=cax)    #plt.colorbar(im, cax=cax)

    plt.tight_layout()    
    plt.savefig('foo.png')
    
    cnt += 1


plt.savefig('oneshot.png')
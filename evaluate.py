import numpy as np
import problexity as pb
from sklearn.datasets import make_classification, make_regression
from time import time
from tqdm import tqdm

# Configure problems
q = 32
repeats = 10
n_samples_range = np.linspace(50,1000,q).astype(int)
n_features_range = np.linspace(2,100,q).astype(int)

# Gather measures
clf_measures = pb.classification.__all__
reg_measures = pb.regression.__all__

# Prepare problem storage
c_XX, c_yy = [[] for _ in range(q)], [[] for _ in range(q)]
r_XX, r_yy = [[] for _ in range(q)], [[] for _ in range(q)]

# Generate problems
for repeat in range(repeats):
    for s, n_samples in enumerate(n_samples_range):
        for n, n_features in enumerate(n_features_range):
            rX, ry = [],[]
            cX, cy = [],[]
        
        # Iterate repeats
            config = {
                'n_samples': n_samples,
                'n_informative': n_features,
                'n_features': n_features,
                'n_redundant': 0
            }
            
            # Generate problems
            c_X, c_y = make_classification(**config)
            config.pop('n_redundant')
            r_X, r_y = make_regression(**(config))
            
            # Store
            cX.append(c_X)
            cy.append(c_y)
            
            rX.append(r_X)
            ry.append(r_y)
        
        # Store
        c_XX[s].append(cX)
        c_yy[s].append(cy)
        r_XX[s].append(rX)
        r_yy[s].append(ry)

# MEASURES x SAMPLES x FEATURES x REPEATS
c_times = np.zeros((len(clf_measures), q, q, repeats))
r_times = np.zeros((len(reg_measures), q, q, repeats))

# Measure times
pbar = tqdm(total=q*q*repeats)

for repeat in range(repeats):
    for s, n_samples in enumerate(n_samples_range):
        for f, n_features in enumerate(n_features_range):
            # Classification evaluation
            X, y = c_XX[s][f][repeat], c_yy[s][f][repeat]

            for measure_idx, measure_name in enumerate(clf_measures): 
                # Gather measure
                measure = getattr(pb.classification, measure_name)

                # Calculate measure
                start = time()
                measure(X, y)
                value = time() - start
                
                # Store measure
                c_times[measure_idx, s, f, repeat] = value
                
            # Regression evaluation
            X, y = r_XX[s][f][repeat], r_yy[s][f][repeat]

            for measure_idx, measure_name in enumerate(reg_measures): 
                # Gather measure
                measure = getattr(pb.regression, measure_name)

                # Calculate measure
                start = time()
                measure(X, y)
                value = time() - start
                
                # Store measure
                r_times[measure_idx, s, f, repeat] = value
                
            pbar.update(1)

np.save('c_times', c_times)
np.save('r_times', r_times)

# coding: utf-8

# In[20]:


import getopt
import sys

if '-i' in sys.argv:
    myopts, args = getopt.getopt(sys.argv[1:],"i:c:")

    for o, a in myopts:
        if o == '-i':
            i=int(a)
        elif o == '-c':
            n_cores=int(a)
        else:
            print("Usage: %s -i iteration -n_cores n_cores" % sys.argv[0])

    # Display input and output file name passed as the args
    print ("Running: %d with n_cores: %d" % (i, n_cores) )
    local = False
else:
    i = 0
    n_cores=11
    local = True
    print('Running locally')

import os


# In[21]:


import pandas as pd
import numpy as np
import scipy as sp
from scipy import optimize
from scipy import stats
import glob
import os
import pickle as pkl

from utils import *


# In[18]:


def obj(pars, x, error_dist, model_n, pc1_mm_norm, pc2_mm_norm, slice_mm_norm):
    
    ev = get_ev(pars, model_n, pc1_mm_norm, pc2_mm_norm, slice_mm_norm)
    
    if error_dist == 'nbinom':
        alpha = pars[-1]
        ll = np.sum(np.log(nbinom_pdf(x, ev, alpha)))
    else:
        ll = np.sum(np.log(poisson_pdf(x, ev)))
        
    if not np.isfinite(ll):
        return np.inf

    return -ll

def fit(model_n, error_dist, data, disp=False, popsize=100, workers=None, **kwargs):           
    if not error_dist in ['nbinom', 'poisson']:
        raise(IOerror('Error dist must be nbinom or poisson'))
    x = data['rate'].values
    mu_bounds = [(-8, 200)]
   
    # Define bounds
    if model_n == 1:
        bounds = mu_bounds
        
    elif model_n == 2:
        beta_pc1_bounds = [(-2, 2)]
        beta_pc2_bounds = [(-2, 2)]
        beta_slice_bounds = [(-2, 2)]
        bounds = mu_bounds + beta_pc1_bounds + beta_pc2_bounds + beta_slice_bounds

    elif model_n >= 3:
        beta_pc1_bounds = [(0, 1)]
        beta_pc2_bounds = [(0, 1)]
        cutoff_1_bounds = [(0.2, 0.6)]
        cutoff_2_bounds = [(0.4, 0.8)]
        lambda_0_bounds = mu_bounds
        lambda_1_bounds = mu_bounds
        bounds = mu_bounds + beta_pc1_bounds + beta_pc2_bounds + cutoff_1_bounds + cutoff_2_bounds + lambda_0_bounds + lambda_1_bounds
        if model_n == 4:
            k_bounds = [(1e-10, 1)]
            bounds = bounds + k_bounds
        
    if error_dist == 'nbinom':
        tau_bounds = [(1e-10, 1)]
        bounds = bounds + tau_bounds
    
    # make tuple of arguments for objective function
    # rate, error distribution, model_n, and location
    _args = (x, error_dist, model_n, data['pc1_mm_norm'].values, data['pc2_mm_norm'].values, data['slice_mm_norm'].values)

    result = sp.optimize.differential_evolution(obj, bounds, _args, 
                                                polish=True, disp=disp, 
                                                maxiter=5000, popsize=20*len(bounds), 
                                                tol=1e-5,
                                                workers=workers, **kwargs)
    if 'seed' in kwargs:
        _ = kwargs.pop('seed')
    if 'init' in kwargs:
        _ = kwargs.pop('init')
    result2 = sp.optimize.minimize(obj, result.x, _args, method='SLSQP', bounds=bounds, **kwargs)
    
    # add number of data points, easy for BIC calculation later on
    result2['n_obs'] = x.shape[0]
    result2['bounds'] = bounds
    result2['deoptim'] = result

    return result2


# In[22]:


# load data
df = pd.read_pickle('./data_fwhm-0.3.pkl')
subjects = df.subject_id.unique()
stains = df.stain.unique()
models = [4,3,2,1]
distributions = ['nbinom', 'poisson']

output_dir = './models_ML'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
import itertools
all_combs = list(itertools.product(stains))


# In[ ]:


# def is_outlier(x, upper=4):
    
#     iqr = np.subtract(*np.percentile(x, [75, 25]))
#     thresh = np.median(np.log(x))+4*iqr
#     outlier_idx = np.log(x) > thresh
#     return outlier_idx
    
# df['is_outlier'] = df.groupby(['subject_id', 'stain'])['rate'].apply(is_outlier)


# In[5]:


if local:
    idx = range(0, 12)
else:
    idx = [i]

for i in idx:
    stain = stains[i]
    for subject in subjects:
        df_to_run = df.loc[(df.subject_id==subject) & (df.stain==stain),:]

        # remove outliers, defined as 4*IQR above median in log-space
        iqr = np.subtract(*np.percentile(np.log(df_to_run['rate']).values, [75, 25]))
        thresh = np.median(np.log(df_to_run['rate']))+4*iqr
        outlier_idx = np.log(df_to_run['rate']) > thresh
        print('{} outlying data points found...'.format(np.sum(outlier_idx)))
        df_to_run = df_to_run.loc[~outlier_idx]

        for model_n in models:
            for distribution in distributions:
                print('Subject {}, stain {}, model {}, distribution {}'.format(subject, stain, model_n, distribution))
                fit_result_fn = os.path.join(output_dir, 'sub-{}_stain-{}_model-{}_distribution-{}_fitresult.pkl').format(subject, stain, model_n, distribution)
                if os.path.exists(fit_result_fn):
                    continue

                fit_result = fit(model_n, distribution, df_to_run, disp=True, workers=n_cores)
                with open(fit_result_fn, 'wb') as f:
                    pkl.dump(fit_result, f)


import pymc3 as pm
from pymc3.distributions.continuous import ExGaussian
import os
from pymc3.backends import Text
import pandas

import theano.tensor as tt
import numpy as np
import itertools


#subject_id = 15055
#fwhm = 0.3
#stain = 'CALR'

fwhms = [0.3, 1.0]
ns = [1,2,3,4,5,6]
subject_ids = [12104, 13095, 14037, 14051, 14069, 15033, 15035, 15055]
stains = ['CALR', 'FER', 'GABRA3', 'GAD6567', 'MBP', 'PARV', 'SERT', 'SMI32', 'SYN', 'TH', 'TRANSF', 'VGLUT1']
print len(list(itertools.product(fwhms, ns, stains, subject_ids)))

if 'PBS_ARRAYID' in os.environ.keys():
    PBS_ARRAYID = int(os.environ['PBS_ARRAYID'])
else:
    PBS_ARRAYID = 0

fwhm, n_clusters, stain, subject_id = list(itertools.product(fwhms, ns, stains, subject_ids))[PBS_ARRAYID]
print n_clusters, fwhm, stain, subject_id


df = pandas.read_pickle(os.path.join(os.environ['HOME'], 'data', 'post_mortem', 'dataframes', '{subject_id}_{fwhm}.pkl'.format(**locals())))


n_samples = 100000
n = 10000
df = df[~df[stain].isnull()]
stepsize = df.shape[0] / n
x = df[stain].values[::stepsize]


ds = os.path.join(os.environ['HOME'], 'traces')

trace_fn = os.path.join(os.environ['HOME'], 'data/post_mortem/traces_map/{subject_id}_{stain}_{fwhm}_{n_clusters}'.format(**locals()))

if n_clusters == 1:
    with pm.Model() as model:
        nu = pm.HalfNormal('nu', tau=5.)
        mu = pm.Normal('mu', mu=0, tau=1/100.)
        sigma = pm.HalfNormal('sigma', tau=1./10.)

        exgauss = ExGaussian('exgauss', mu=mu, sigma=sigma, nu=nu, observed=x)

        


        db = Text(trace_fn)

        step = pm.Metropolis()
        
        pm.find_MAP()

        trace_ = pm.sample(n_samples, step, trace=db, njobs=14)
        
else:
    with pm.Model() as model:
        
        w = pm.Dirichlet('w', np.ones(n_clusters) * 10)

        nu = pm.HalfNormal('nu', tau=1/25., shape=n_clusters)
        mu = pm.Normal('mu', mu=0, tau=1/25., shape=n_clusters)
        sigma = pm.HalfNormal('sigma', tau=1./10., shape=n_clusters)

        expon = pm.Mixture('mix', w,
                           [ExGaussian.dist(mu=mu[i], sigma=sigma[i], nu=nu[i]) for i in xrange(n_clusters)], observed=x)
        
        potential_formula = [tt.switch((mu[i] + nu[i])-(mu[i-1] + nu[i-1]) < 0, -np.inf, 0) for i in xrange(1, n_clusters)]
        
        
        if n_clusters == 2:
            potential_formula = potential_formula[0]
        else:
            potential_formula = tt.add(*potential_formula)
            
        order_means_potential = pm.Potential('order_means_potential', potential_formula)
        
        

        db = Text(trace_fn)
        step = pm.Metropolis()

        pm.find_MAP()
        
        trace_ = pm.sample(n_samples, step, trace=db, njobs=14)
        




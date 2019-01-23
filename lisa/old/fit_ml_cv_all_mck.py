import os
import pandas

import numpy as np
import itertools

from multiprocessing import Pool

import scipy as sp
from scipy import optimize

import pickle as pkl

n_proc = 14


fwhms = [0.3]
ns = [1,2,3,4,5,6]
subject_ids = [13095, 14037, 14051, 14069, 15033, 15035, 15055]
stains = ['CALR', 'FER', 'GABRA3', 'GAD6567', 'MBP', 'PARV', 'SERT', 'SMI32', 'SYN', 'TH', 'TRANSF', 'VGLUT1']
print len(list(itertools.product(fwhms, ns, stains, subject_ids)))

if 'PBS_ARRAYID' in os.environ.keys():
    PBS_ARRAYID = int(os.environ['PBS_ARRAYID'])
else:
    PBS_ARRAYID = 0

fwhm, n_clusters, stain, subject_id = list(itertools.product(fwhms, ns, stains, subject_ids))[PBS_ARRAYID]
print n_clusters, fwhm, stain, subject_id

def exgauss_pdf(x, mu, sigma, nu):

    nu = 1./nu

    p1 = nu / 2. * np.exp((nu/2.)  * (2 * mu + nu * sigma**2. - 2. * x))


    p2 = sp.special.erfc((mu + nu * sigma**2 - x)/ (np.sqrt(2.) * sigma))

    return p1 * p2

def mixed_exgauss_likelihood(x, w, mu, sigma, nu):

    # Create indiviudal
    pdfs = w * exgauss_pdf(x[:, np.newaxis], mu, nu, sigma)

    ll = np.sum(np.log(np.sum(pdfs, 1)))

    if ((np.isnan(ll)) | (ll == np.inf)):
        return -np.inf


    return ll

def input_optimizer(pars, x, n_clusters):

    pars = np.array(pars)

    if np.sum(pars[:n_clusters-1]) > 1:
        return np.inf

    pars = np.insert(pars, n_clusters-1, 1 - np.sum(pars[:n_clusters-1]))

    if np.any(pars[:n_clusters] < 0.05):
        return np.inf

    w = pars[:n_clusters][np.newaxis, :]
    mu = pars[n_clusters:n_clusters*2][np.newaxis, :]
    nu = pars[n_clusters*2:n_clusters*3][np.newaxis, :]
    sigma = pars[n_clusters*3:n_clusters*4][np.newaxis, :]

    return -mixed_exgauss_likelihood(x, w, mu, sigma, nu)


def _fit(input_args, disp=False, popsize=100, **kwargs):

    sp.random.seed()

    x, n_clusters = input_args

    weight_bounds = [(1e-3, 1)] * (n_clusters - 1)
    mu_bounds = [(-1., 2.5)] * n_clusters
    nu_bounds = [(1e-3, 2.5)] * n_clusters
    sigma_bounds = [(1e-3, 2.5)] * n_clusters

    bounds = weight_bounds + mu_bounds + nu_bounds + sigma_bounds

    result = sp.optimize.differential_evolution(input_optimizer, bounds, (x, n_clusters), polish=True, disp=disp, maxiter=500, popsize=popsize, **kwargs)
    result = sp.optimize.minimize(input_optimizer, result.x, (x, n_clusters), method='SLSQP', bounds=bounds, **kwargs)

    return result

class SimpleExgaussMixture(object):


    def __init__(self, data, n_clusters):

        self.data = data
        self.n_clusters = n_clusters
        self.n_parameters = n_clusters * 4 - 1
        self.likelihood = -np.inf

        self.previous_likelihoods = []
        self.previous_pars = []


    def get_likelihood_data(self, data):
        
        return mixed_exgauss_likelihood(data, self.w, self.mu, self.sigma, self.nu)
    
    def get_bic_data(self, data):
        likelihood = self.get_likelihood_data(data)
        return - 2 * likelihood + self.n_parameters * np.log(data.shape[0])
        
        
    def get_aic_data(self, data):
        likelihood = self.get_likelihood_data(data)
        return 2 * self.n_parameters - 2  * likelihood
    

    def _fit(self, **kwargs):
        return _fit((self.data, self.n_clusters), **kwargs)



    def fit(self, n_tries=1, **kwargs):
        for run in np.arange(n_tries):

            result = self._fit(**kwargs)
            self.previous_likelihoods.append(-result.fun)

            if -result.fun > self.likelihood:

                pars = result.x
                pars = np.insert(pars, self.n_clusters-1, 1 - np.sum(pars[:self.n_clusters-1]))

                self.w = pars[:self.n_clusters][np.newaxis, :]
                self.mu = pars[self.n_clusters:self.n_clusters*2][np.newaxis, :]
                self.nu = pars[self.n_clusters*2:self.n_clusters*3][np.newaxis, :]
                self.sigma = pars[self.n_clusters*3:self.n_clusters*4][np.newaxis, :]

                self.likelihood = -result.fun

        self.aic = 2 * self.n_parameters - 2 * self.likelihood
        self.bic = - 2 * self.likelihood + self.n_parameters * np.log(self.data.shape[0])



    def fit_multiproc(self, n_tries=4, n_proc=4, disp=False):

        pool = Pool(n_proc)

        print 'starting pool'
        results = pool.map(_fit, [(self.data, self.n_clusters)] * n_tries)
        print 'ready'

        print results



        pool.close()

        for result in results:
            self.previous_likelihoods.append(-result.fun)
            self.previous_pars.append(result.x)

            if -result.fun > self.likelihood:

                pars = result.x
                pars = np.insert(pars, self.n_clusters-1, 1 - np.sum(pars[:self.n_clusters-1]))

                self.w = pars[:self.n_clusters][np.newaxis, :]
                self.mu = pars[self.n_clusters:self.n_clusters*2][np.newaxis, :]
                self.nu = pars[self.n_clusters*2:self.n_clusters*3][np.newaxis, :]
                self.sigma = pars[self.n_clusters*3:self.n_clusters*4][np.newaxis, :]

                self.likelihood = -result.fun

        self.aic = 2 * self.n_parameters - 2 * self.likelihood
        self.bic = - 2 * self.likelihood + self.n_parameters * np.log(self.data.shape[0])

    def plot_fit(self):
        # Create indiviudal pds

        t = np.linspace(0, self.data.max(), 100)
        pdfs = self.w * exgauss_pdf(t[:, np.newaxis], self.mu, self.nu, self.sigma)

        sns.distplot(self.data)
        plt.plot(t, pdfs, c='k', alpha=0.5)

        plt.plot(t, np.sum(pdfs, 1), c='k', lw=2)

df_even = pandas.read_pickle(os.path.join(os.environ['HOME'], 'data/post_mortem/dataframes/{subject_id}_{fwhm}_even_double_slices.pkl'.format(**locals())))
df_even_1 = pandas.read_pickle(os.path.join(os.environ['HOME'], 'data/post_mortem/dataframes/{subject_id}_{fwhm}_uneven_double_slices.pkl'.format(**locals())))
df_even = pandas.concat((df_even, df_even_1), ignore_index=True)

df_uneven = pandas.read_pickle(os.path.join(os.environ['HOME'], 'data/post_mortem/dataframes/{subject_id}_{fwhm}_even_double_slices_1.pkl'.format(**locals())))
df_uneven_1 = pandas.read_pickle(os.path.join(os.environ['HOME'], 'data/post_mortem/dataframes/{subject_id}_{fwhm}_uneven_double_slices_3.pkl'.format(**locals())))
df_uneven = pandas.concat((df_uneven, df_uneven_1), ignore_index=True)

n_samples = 100000
n = 10000

even_stepsize = df_even.shape[0] / n
df_even = df_even[~df_even[stain].isnull()]

x_even = df_even[stain].values[::even_stepsize]
x_even -= x_even.min()
x_even /= x_even.max()


uneven_stepsize = df_uneven.shape[0] / n
df_uneven = df_uneven[~df_uneven[stain].isnull()]

x_uneven = df_uneven[stain].values[::uneven_stepsize]
x_uneven -= x_uneven.min()
x_uneven /= x_uneven.max()


s_even = SimpleExgaussMixture(x_even, n_clusters)
s_uneven = SimpleExgaussMixture(x_uneven, n_clusters)

s_even.fit_multiproc(n_tries=n_proc, n_proc=n_proc)
s_uneven.fit_multiproc(n_tries=n_proc, n_proc=n_proc)

results = [{'train':'even', 
 'test':'even',
 'll':s_even.get_likelihood_data(x_even),
 'aic':s_even.get_aic_data(x_even),
 'bic':s_even.get_bic_data(x_even)},
 {'train':'even', 
 'test':'uneven',
 'll':s_even.get_likelihood_data(x_uneven),
 'aic':s_even.get_aic_data(x_uneven),
 'bic':s_even.get_bic_data(x_uneven)},
 {'train':'uneven', 
 'test':'even',
 'll':s_uneven.get_likelihood_data(x_even),
 'aic':s_uneven.get_aic_data(x_even),
 'bic':s_uneven.get_bic_data(x_even)},
 {'train':'uneven', 
 'test':'uneven',
 'll':s_uneven.get_likelihood_data(x_uneven),
 'aic':s_uneven.get_aic_data(x_uneven),
 'bic':s_uneven.get_bic_data(x_uneven)}, 
 ]


results = pandas.DataFrame(results)
results['subject_id'], results['stain'], results['fwhm'], results['n_clusters'] = subject_id, stain, fwhm, n_clusters

results_fn = os.path.join(os.environ['HOME'], 'data', 'post_mortem', 'ml_clusters_cross_validated', '{subject_id}_{fwhm}_{stain}_{n_clusters}_all.pandas'.format(**locals()))
results.to_pickle(results_fn)


pickle_even_fn = os.path.join(os.environ['HOME'], 'data', 'post_mortem', 'ml_clusters_cross_validated', '{subject_id}_{fwhm}_{stain}_{n_clusters}_even_all.pkl'.format(**locals()))
pkl.dump(s_even, open(pickle_even_fn, 'w'))

pickle_uneven_fn = os.path.join(os.environ['HOME'], 'data', 'post_mortem', 'ml_clusters_cross_validated', '{subject_id}_{fwhm}_{stain}_{n_clusters}_uneven_all.pkl'.format(**locals()))
pkl.dump(s_uneven, open(pickle_uneven_fn, 'w'))

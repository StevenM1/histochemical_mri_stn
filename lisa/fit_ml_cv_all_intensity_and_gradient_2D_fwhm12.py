# The following script is used to fit the mixture models to the data.
# Note that this script has been ported to a .py file 
# The .py file was saved as fit_ml_cv_all_intensity.py or fit_ml_cv_all_gradient.py and run on lisa
#    This means that there are a few pieces of code that are lisa (surfsara.nl) specific.

# Import several functions and set a few style features:
import os
import pandas
import numpy as np
import itertools
from multiprocessing import Pool

import scipy as sp
from scipy import optimize
import glob
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt

# How many processes do you want to run simulataneously on Lisa? The node has 16 cores, it is advised due
#  to mememory issues you dont use all of them at the same time.
n_proc = 14

# Defining a number of functions:    
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
        
class Scaler(object):
    
    def __init__(self):
        self.min = None
        self.max = None
    
    def fit(self, X):
        self.min = X.min()
        self.max = X.max()
    
    def transform(self, X):
        X -= self.min
        X /= self.max
        
        return X

# Starting the actual fitting:

# Select the smoothing kernel
fwhms = [0.3]
# Set the number of mixtures
ns = [1,2,3,4,5,6]
# Subject ids
subject_ids = [13095, 14037, 14051, 14069, 15033, 15035, 15055]
# Stains
stains = ['CALR', 'FER', 'GABRA3', 'GAD6567', 'MBP', 'PARV', 'SERT', 'SMI32', 'SYN', 'TH', 'TRANSF', 'VGLUT1']
# data type
data_types = ['gradient_2D']
print len(list(itertools.product(fwhms, ns, stains, subject_ids, data_types)))

# PBS array is used on lisa so that you can submit alot of similiar jobs to the cue. We have 504 jobs to submit 
#    (12 stains * 6 number of clusters * 7 tissue blocks)
if 'PBS_ARRAYID' in os.environ.keys():
    PBS_ARRAYID = int(os.environ['PBS_ARRAYID'])
else:
    PBS_ARRAYID = 0

fwhm, n_clusters, stain, subject_id, data_type = list(itertools.product(fwhms, ns, stains, subject_ids, data_types))[PBS_ARRAYID]
print n_clusters, fwhm, stain, subject_id, data_type

results = []
# The different training and test set partitions
partitions = [{'train_name': 'All-Data-In-Mask'},
               {'train_name': 'CV_set1_1', 'test_name': 'CV_set1_2'},
               {'train_name': 'CV_set1_2', 'test_name': 'CV_set1_1'},
               {'train_name': 'CV_set2_1', 'test_name': 'CV_set2_2'},
               {'train_name': 'CV_set2_2', 'test_name': 'CV_set2_1'},
               {'train_name': 'CV_set3_1', 'test_name': 'CV_set3_2'},
               {'train_name': 'CV_set3_2', 'test_name': 'CV_set3_1'}]

for partition in partitions:
    print('Current train set: %s' % partition['train_name'])
   
    # Always fit model to train data
    # Load data
    train_set = pandas.read_pickle(os.path.join(os.environ['HOME'], 
                        'data/post_mortem/crossval_%s_feature_vectors/%s_%s_%s.pkl' %(data_type, subject_id, fwhm, partition['train_name'])))

    # Select stain
    train_set = train_set[stain]

    # Remove nan-values
    train_set = train_set[~pandas.isnull(train_set)]

    # Select reasonable subsample to decrease computational burden.
    # Get random evenly spaced (should be evenly spaced to make sure a representative sub sample
    # of the spatial organisation of the STN is selected) sub sample of 15% to reduce size
    # Getting exactly 15% is tricky because of varying original shapes (sample sizes), but this
    # approach gets the value closest to and minimum of 15%
    step_size = int(train_set.shape[0] / (train_set.shape[0]*.15))
    train_set = train_set[::step_size]

    # The train and test partitions are normalized per partition seperately. In a previous attempt we normalised 
    #   the test partition by the train partition. This however in the end led tot inf values for the the loglikelihood.
    #   Since it could happen that the test partition had values that were outside the min/max of the train partition and 
    #   would therefore get a zero as likelihood -> log(0) == inf.
    scaler = Scaler()
    scaler.fit(train_set)
    train_set = scaler.transform(train_set)

    # Check if model is already saved to disk. If so, load model
    path_name = os.path.join(os.environ['HOME'], 'data', 'post_mortem', 'ml_clusters_cross_validated_%s' %data_type)
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    pickle_fn = os.path.join(os.environ['HOME'], 'data', 'post_mortem', 'ml_clusters_cross_validated_%s' %data_type, '%s_%s_%s_%s_%s.pkl' %(subject_id, fwhm, stain, n_clusters, partition['train_name']))

    if os.path.isfile(pickle_fn):
        # Model was already trained & saved, so load
        with open(pickle_fn, 'r') as f:
            model = pkl.load(f)
    else:
        # Model does not yet exist, so train now
        # Create & fit model on train set
        model = SimpleExgaussMixture(train_set, n_clusters)
        model.fit_multiproc(n_tries=n_proc, n_proc=n_proc)
        pkl.dump(model, open(pickle_fn, 'w'))

    # Append to results
    results.append({'train': partition['train_name'], 'test': partition['train_name'],
                   'll': model.get_likelihood_data(train_set),
                   'aic': model.get_aic_data(train_set),
                   'bic': model.get_bic_data(train_set)})


    # If a test-set is provided, check cross-validation model fit
    if 'test_name' in partition.keys():
        # Load test data
        test_set = pandas.read_pickle(os.path.join(os.environ['HOME'], 'data/post_mortem/crossval_%s_feature_vectors/%s_%s_%s.pkl' %(data_type, subject_id, fwhm, partition['test_name'])))
        
        # select stain
        test_set = test_set[stain]    

        # Remove nan/Nones
        test_set = test_set[~pandas.isnull(test_set)]

        # Subsample
        step_size = int(test_set.shape[0] / (test_set.shape[0]*.15))
        test_set = test_set[::step_size]

        # Scale
        scaler = Scaler()
        scaler.fit(test_set)
        test_set = scaler.transform(test_set)

        # Append to results
        results.append({'train': partition['train_name'], 'test': partition['test_name'],
                       'll': model.get_likelihood_data(test_set),
                       'aic': model.get_aic_data(test_set),
                       'bic': model.get_bic_data(test_set)})

results = pandas.DataFrame(results)
results['subject_id'], results['stain'], results['fwhm'], results['n_clusters'] = subject_id, stain, fwhm, n_clusters
# Saving the data to pandas
results_fn = os.path.join(os.environ['HOME'], 'data', 'post_mortem', 'ml_clusters_cross_validated_%s' %(data_type), '%s_%s_%s_%s_all.pandas' %(subject_id, fwhm, stain, n_clusters))
results.to_pickle(results_fn)

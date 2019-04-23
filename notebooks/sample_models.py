
# coding: utf-8

# In[60]:


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
else:
    i = 0
    n_cores=11
    print('Running locally')

# set theano flags
import os
os.environ["THEANO_FLAGS"] = "compiledir=./theano/%s/" %(str(i))


# In[61]:


import pandas as pd
import pymc3 as pm
import numpy as np
import glob
import os
import pickle as pkl


# In[62]:


# code for smooth switch point:
# https://gist.github.com/junpenglao/f7098c8e0d6eadc61b3e1bc8525dd90d
import theano.tensor as tt
from pymc3.distributions.transforms import ElemwiseTransform, Transform

class Ordered(ElemwiseTransform):
    name = "ordered"

    def backward(self, y):
        out = tt.zeros(y.shape)
        out = tt.inc_subtensor(out[0], y[0])
        out = tt.inc_subtensor(out[1:], tt.exp(y[1:]))
        return tt.cumsum(out)

    def forward(self, x):
        out = tt.zeros(x.shape)
        out = tt.inc_subtensor(out[0], x[0])
        out = tt.inc_subtensor(out[1:], tt.log(x[1:] - x[:-1]))
        return out

    def forward_val(self, x, point=None):
        x, = draw_values([x], point=point)
        return self.forward(x)

    def jacobian_det(self, y):
        return tt.sum(y[1:])

ordered = Ordered()


class Composed(Transform):
    def __init__(self, transform1, transform2):
        self._transform1 = transform1
        self._transform2 = transform2
        self.name = '_'.join([transform1.name, transform2.name])

    def forward(self, x):
        return self._transform2.forward(self._transform1.forward(x))

    def forward_val(self, x, point=None):
        return self.forward(x)

    def backward(self, y):
        return self._transform1.backward(self._transform2.backward(y))

    def jacobian_det(self, y):
        y2 = self._transform2.backward(y)
        det1 = self._transform1.jacobian_det(y2)
        det2 = self._transform2.jacobian_det(y)
        return det1 + det2
    
def logistic(L, x0, k=50, t_=np.linspace(0., 1., 1000)):
    x0 = x0*(t_.max()-t_.min()) + t_.min()  # scale x0 to t_
    return L/(1+tt.exp(-k*(t_-x0)))


# In[156]:


def sample_single_model(df, model_n, distribution, dep_var='rate', n_cores=15):

    with pm.Model() as model:
        intercept = pm.Normal('intercept', mu=np.log(df[dep_var].mean()), sd=3)
        tune_steps = 500

        if model_n == 1:
            # no change
            ev = np.exp(intercept)

        elif model_n == 2:
            # gradient along pc axis 1
            beta_pca_1 = pm.Normal('beta_pca_1', mu=0, sd=3)
            ev = np.exp(intercept + beta_pca_1*df['pc1_mm_norm'].values)

        elif model_n == 3:
            # 3 sectors along pc axis 1
            delta_center_1 = pm.Normal('delta_center_1', mu=0, sd=3)
            delta_center_3 = pm.Normal('delta_center_3', mu=0, sd=3)

            ev = np.exp(intercept +                         delta_center_1*((df['pc1_mm_perc'].values<0.333).astype(int)) +                         delta_center_3*((df['pc1_mm_perc'].values>0.667).astype(int)))

        elif model_n == 4:
            # gradient along pc axis 1+2+3
            beta_pca_1 = pm.Normal('beta_pca_1', mu=0, sd=3)
            beta_pca_2 = pm.Normal('beta_pca_2', mu=0, sd=3)
            beta_slice = pm.Normal('beta_slice', mu=0, sd=3)

            ev = np.exp(intercept + beta_pca_1*df['pc1_mm_norm'].values +                         beta_pca_2*df['pc2_mm_norm'].values +                         beta_slice*df['slice_mm_norm'].values)

        elif model_n == 5:
            # cut-offs estimated, smoothness of cut-off estimated
            nbreak = 3
            lambdad = pm.Normal('lambdad', 0, sd=1, shape=3-1)
            k = pm.HalfNormal('k', 20)  # always assume positive
            trafo = Composed(pm.distributions.transforms.LogOdds(), Ordered())
            b = pm.Beta('b', 4., 4., shape=nbreak-1, transform=trafo,
                        testval=[0.33, 0.67])
            ev = np.exp(intercept + logistic(lambdad[0], b[0], k=-k, t_=df['pc1_mm_norm'].values) +
                                    logistic(lambdad[1], b[1], k=k, t_=df['pc1_mm_norm'].values))
            tune_steps = 1500
            
        elif model_n == 6:
            # cut-offs estimated in 3D, smoothness of cut-off estimated
            beta_pca_1 = pm.HalfNormal('beta_pca_1', sd=3)
            beta_pca_2 = pm.HalfNormal('beta_pca_2', sd=3)
            beta_slice = pm.HalfNormal('beta_slice', sd=3)
            k = pm.HalfNormal('smoothness', sd=3)
            trafo = Composed(pm.distributions.transforms.LogOdds(), Ordered())
            b = pm.Beta('b', 4., 4., shape=2, transform=trafo, testval=[0.33, 0.67])
            lambdad = pm.Normal('lambdad', mu=0, sd=1, shape=3-1)
            
            # use principal components in percentage for 0-1 scale
            O_vec = beta_pca_1*df['pc1_mm_norm'].values +                     beta_pca_2*df['pc2_mm_norm'].values +                     beta_slice*df['slice_mm_norm'].values
            O_vec = (O_vec-O_vec.min())/(O_vec.max()-O_vec.min())
        
            ev = np.exp(intercept + intercept*logistic(lambdad[0], b[0], k=k, t_=O_vec) +
                                    intercept*logistic(lambdad[1], b[1], k=k, t_=O_vec))
            tune_steps = 1500

        # define likelihood
        if distribution == 'poisson':
            likelihood = pm.Poisson('y', mu=ev, observed=df[dep_var])
        else:
            alpha = pm.HalfCauchy('alpha', beta=2)
            likelihood = pm.NegativeBinomial('y', mu=ev, alpha=alpha, observed=df[dep_var])

        model.name = str(model_n) + '_' + distribution
        traces = pm.sample(cores=n_cores, tune=tune_steps)
        
        return model, traces


# In[157]:


# load data
df = pd.read_pickle('./data_fwhm-0.3.pkl')
subjects = df.subject_id.unique()
stains = df.stain.unique()
models = [6,5,4,3,2,1]
distributions = ['poisson', 'negativebinomial']

output_dir = './models'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# In[158]:


import itertools
all_combs = list(itertools.product(subjects, stains))


# In[162]:


# i = 1


# In[163]:


subject, stain = all_combs[i]
df_to_run = df.loc[(df.subject_id==subject) & (df.stain==stain),:]


# In[164]:


for model_n in models:
    for distribution in distributions:
        print('Subject {}, stain {}, model {}, distribution {}'.format(subject, stain, model_n, distribution))
        trace_fn = os.path.join(output_dir, 'sub-{}_stain-{}_model-{}_distribution-{}_type-traces.pkl').format(subject, stain, model_n, distribution)
        model_fn = os.path.join(output_dir, 'sub-{}_stain-{}_model-{}_distribution-{}_type-model.pkl').format(subject, stain, model_n, distribution)
        if os.path.exists(trace_fn):
            continue
        
        model, traces = sample_single_model(df_to_run, model_n, distribution, n_cores=n_cores)
        with open(model_fn, 'wb') as f:
            pkl.dump(model, f)
        with open(trace_fn, 'wb') as f:
            pkl.dump(traces, f)


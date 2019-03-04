import numpy as np
from scipy.special import factorial, comb, gammaln
import scipy as sp

def logistic(L, x0, k=50, t_=np.linspace(0., 1., 1000)):
    x0 = x0*(t_.max()-t_.min()) + t_.min()  # scale x0 to t_
    return L/(1+np.exp(-k*(t_-x0)))

def nbinom_pdf(x, mu, tau):
    """ Parametrisation from pymc3: https://docs.pymc.io/api/distributions/discrete.html#pymc3.distributions.discrete.NegativeBinomial
    except tau = 1/alpha so that tau is bound within (0, 1]
    """
    
    alpha = 1/tau
    alpha_plus_mu = mu + alpha
    return np.exp(np.log(comb(x + alpha - 1, x))+ alpha*np.log(alpha/alpha_plus_mu) + x*np.log(mu/alpha_plus_mu))

def poisson_pdf(k, mu):
    """ Poisson PDF """
    return np.exp(-mu + k*np.log(mu) - gammaln(k+1))

def get_ev(pars, model_n, pc1_mm_norm, pc2_mm_norm, slice_mm_norm, return_projection_axis=False):
    """ Calculates expected rate (ev) based on parameters & location """
    if model_n == 1:
        ev = np.exp(pars[0])
    elif model_n == 2:
        mu = pars[0]
        beta_pc1 = pars[1]
        beta_pc2 = pars[2]
        beta_slice = pars[3]
        ev = np.exp(mu + \
                    beta_pc1*pc1_mm_norm \
                    + beta_pc2*pc2_mm_norm \
                    + beta_slice*slice_mm_norm)
        
        if return_projection_axis:
            O_vec = beta_pc1*pc1_mm_norm \
                    + beta_pc2*pc2_mm_norm \
                    + beta_slice*slice_mm_norm
            O_vec = (O_vec-O_vec.mean())/(O_vec.std())
            return O_vec
            
    elif model_n >= 3:
        mu = pars[0]
        beta_pc1 = pars[1]
        beta_pc2 = pars[2]

        cutoff_1 = pars[3]
        cutoff_2 = pars[4]
        cutoff_1, cutoff_2 = np.sort([cutoff_1, cutoff_2])
#         each sector is at least 20% of the full STN size
        if cutoff_2 - cutoff_1 < .2:
            return np.inf

        lambda_0 = pars[5]
        lambda_1 = pars[6]
        
        # since we assume the cut-off axis has a fixed length, we only need 2 parameters to estimate this vector
#        beta_slice = np.sqrt(1 - beta_pc1**2 - beta_pc2**2)
        beta_pc2 = beta_pc1+(1-beta_pc1)*beta_pc2
        beta_slice = 1-beta_pc2

        O_vec = beta_pc1*pc1_mm_norm + \
                beta_pc2*pc2_mm_norm + \
                beta_slice*slice_mm_norm
        # standardize new axis
        O_vec = (O_vec-O_vec.mean())/(O_vec.std())
             
        if return_projection_axis:
            return O_vec
        
        if model_n == 3:
            is_sector_1 = O_vec < np.percentile(O_vec, cutoff_1*100)
            is_sector_3 = O_vec > np.percentile(O_vec, cutoff_2*100)
        
            ev = np.exp(mu + \
                        lambda_0*is_sector_1 + \
                        lambda_1*is_sector_3)
        elif model_n == 4:
            k = 1/pars[7]
            ev = np.exp(mu + \
                        logistic(lambda_0, cutoff_1, k=-k, t_=O_vec) + \
                        logistic(lambda_1, cutoff_2, k=k, t_=O_vec))
    return ev
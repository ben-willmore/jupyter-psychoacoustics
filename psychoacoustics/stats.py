'''
Misc stats functions
'''

# pylint: disable=C0103, R0912, R0914

import numpy as np
from statsmodels.tools.tools import add_constant
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families

def logistic(x, a, b):
    '''
    Calculate logistic function of ndarray x, with offset a and slope 1/b
    '''
    return  1/(1+np.exp((-x+a)/b))

def probit_fit(x, resp):
    '''
    Probit fit with 95% CIs
    '''

    # binomial GLM with probit link
    model = GLM(resp, add_constant(x),
                family=families.Binomial(),
                link=families.links.probit())
    mod_result = model.fit(disp=0)
    xt = np.linspace(np.min(x), np.max(x), 100)

    r_hat = mod_result.predict(add_constant(xt))
    pred_summ = mod_result.get_prediction(add_constant(xt)).summary_frame(alpha=0.05)
    ci_5, ci_95 = pred_summ['mean_ci_lower'], pred_summ['mean_ci_upper']

    return mod_result.params, r_hat, (xt, ci_5, ci_95)

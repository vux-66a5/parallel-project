import numpy as np

def clique_energy_ho(d, p, th, quant):
    '''
    Computes clique energy: e = th^(p-2)*d^2*mask + d^p*(1-mask)
    e = clique_engergy_ho(d, p, th, quant)

    Input arguments ----------------------
    d           clique difference
    p           power law exponent  
    th          it defines a region over which the potential grows quadratically
    quant       it defines whether or not the potential is quantized
    '''

    if quant == 'no':
        d = np.abs(d)
    elif quant == 'yes':
        d = np.abs(np.round(d/2/np.pi)*2*np.pi)

    if th != 0:
        mask = (d <= th)
        e = th ** (p - 2) * d ** 2 * mask + d ** p * (1 - mask)
    else:
        e = d ** p

    return e 


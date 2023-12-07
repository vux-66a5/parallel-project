import numpy as np
import sys
sys.path.insert(0, 'C:\\Users\\ADMIN\\Desktop\\parallel-project\\main\\utils\\puma')

from clique_energy_ho import clique_energy_ho

def energy_ho(kappa, psi, base, p, cliques, disc_bar, th, quant):
    ''' 
    energy_ho   Energy from kappa labeling and psi phase measurements.
    erg = energy_ho(kappa,psi,base,p,cliques,disc_bar,p,th,quant) returns the energy of kappa labeling given the 
    psi measurements image, the base ROI image (having ones in the region of interest (psi) and a passe-partout
    made of zeros), the exponent p, the cliques matrix (each row indicating a displacement vector corresponding
    to each clique), the disc_bar (complement to one of the quality maps), a threshold th defining a region for
    which the potential (before a possible quantization) is quadratic, and quant which is a flag defining whether
    the potential is or is not quantized.
    (see J. Bioucas-Dias and G. Valadão, "Phase Unwrapping via Graph Cuts"
    submitted to IEEE Transactions Image Processing, October, 2005).
    SITE: www.lx.it.pt/~bioucas/ 
    '''

    m, n = np.array(psi).shape
    cliquesm, cliquesn = cliques.shape  # Size of input cliques
    maxdesl = np.max(np.max(np.abs(cliques)))   # This is the maximum clique length used 

    # Here we put a passe-partout (constant length = maxdesl + 1) in the images kappa and psi 
    base_kappa = np.zeros((2*maxdesl + 2 + m, 2*maxdesl + 2 + n))
    base_kappa[maxdesl+1:maxdesl+2+m-1, maxdesl+1:maxdesl+2+n-1] = kappa

    psi_base = np.zeros((2*maxdesl + 2 + m, 2*maxdesl + 2 + n))
    psi_base[maxdesl+1:maxdesl+2+m-1, maxdesl+1:maxdesl+2+n-1] = psi

    z = disc_bar.shape[2]
    base_disc_bar = np.zeros((2 * maxdesl + 2 + m, 2 * maxdesl + 2 + n, z))
    base_disc_bar[maxdesl + 1:maxdesl + 2 + m-1, maxdesl + 1:maxdesl + 2 + n-1, :] = disc_bar 

    
   
    for t in range(cliquesm):
        # The allowed start and end pixels of the "interpixel" directed edge
        # base_start = np.roll(base, shift=(-cliques[t, 0], -cliques[t, 1]), axis=(0, 1)) * base
        # base_end = np.roll(base, shift=(cliques[t, 0], cliques[t, 1]), axis=(0, 1)) * base

        '''
        By convention the difference images have the same size as the
        original ones; the difference information is retrieved in the
        pixel of the image that is subtracted (end of the diff vector)
        '''
        auxili = np.roll(base_kappa, shift=(cliques[t, 0], cliques[t, 1]), axis=(0, 1))

        # Compute t_dkappa
        t_dkappa = np.zeros((2*maxdesl+2+m, 2*maxdesl+2+n, cliquesm))
        t_dkappa[:, :, t] = (base_kappa - auxili)

        # Circular shift psi_base
        auxili2 = np.roll(psi_base, shift=(cliques[t, 0], cliques[t, 1]), axis=(0, 1))

        # Compute dpsi
        dpsi = auxili2 - psi_base

        '''
        Beyond base, we must multiply by
        circshift(base,[cliques(t,1),cliques(t,2)]) in order to
        account for frontier pixels that can't have links outside ROI
        '''
        shifted_base = np.roll(base, shift=(cliques[t, 0], cliques[t, 1]), axis=(0, 1))

        # Compute a
        a = (2 * np.pi * t_dkappa[:, :, t] - dpsi) * base * shifted_base * base_disc_bar[:, :, t]

    erg = np.sum(np.sum(np.sum((clique_energy_ho(a,p,th,quant)))))

    return erg 



    
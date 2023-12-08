path_name = "C:\\Users\\ADMIN\\Desktop\\parallel-project"

import numpy as np
import sys
sys.path.insert(0,  path_name + '\\main\\utils\\puma')

from energy_ho import energy_ho
from mincut import mincut
from clique_energy_ho import clique_energy_ho

def puma_ho(psi, p, **kwargs):
    #Default values
    potential = {'quantized': 'yes', 'threshold': 0}
    cliques = np.array([[1, 0], [0, 1]])
    size_psi = np.array(psi).shape
    qualitymaps = np.zeros((size_psi[0], size_psi[1], 2))
    qual = 0 
    schedule = [1]

    # test for number of required parameters
    # Error out if there are not at least the two required input arguments
    if len(kwargs) != 0:
        raise ValueError('Wrong number of required parameters')
    
    # Read the optional parameters
    if len(kwargs) % 2 == 1:
        raise ValueError('Optional parameters should always go by pairs')
    elif len(kwargs) != 0:
        for i in range(1, len(kwargs), 2):
            # change the value of parameter
            if kwargs[i] == 'potential':    
                potential = kwargs[i+1]
            elif kwargs[i] == 'cliques':
                cliques = kwargs[i+1]
            elif kwargs[i] == 'qualitymaps':
                qualitymaps = kwargs[i+1]
                qual = 1
            elif kwargs[i] == 'schedule':
                schedule = kwargs[i+1]
            else:
                # Hmmm, something wrong with the parameter string
                raise ValueError(f"Unrecognized parameter: '{kwargs[i]}'")
    
    if qual == 1 and qualitymaps.shape[2] != cliques.shape[0]:
        raise ValueError("'qualitymaps must be a 3D matrix whos 3D size is equal to no. cliques. Each plane on qualitymaps corresponds to a clique.'")
    
    # INPUT AND INITIALIZING 
    th = potential['threshold']
    quant = potential['quantized']

    m, n = np.array(psi).shape        # size of input
    kappa = np.zeros((m, n))    # initial labeling 
    kappa_aux = np.copy(kappa)
    iter = 0
    erglist = []

    cliquesm, cliquesn = cliques.shape  # size of input cliques
    if qual == 0:
        qualitymaps = np.zeros((size_psi[0], size_psi[1], cliques.shape[0]))

    disc_bar = 1 - qualitymaps

    # "maxdesl" is the maximum clique length used.
    maxdesl = np.max(np.max(np.abs(cliques)))

    #We define "base" which is a mask having ones in the region of interest(psi) and zeros upon a passe-partout
    #having a constant length maxdesl+1.
    base = np.zeros((2 * maxdesl + 2 + m, 2 * maxdesl + 2 + n))
    base[maxdesl + 1:maxdesl + 1 + m, maxdesl + 1:maxdesl + 1 + n] = np.ones((m, n))

    # Initialize source and sink outside the loop
    source = np.zeros((m, n, cliquesm))
    sink = np.zeros((m, n, cliquesm))
    auxiliar1 = np.zeros((m, n, cliquesm))


    # PROCESSING 
    for jump_size in schedule:
        possible_improvment = 1
        erg_previous = energy_ho(kappa,psi,base,p,cliques,disc_bar,th,quant)
        

        while possible_improvment != 0:
            iter += 1
            # erglist.append(erg_previous)
            remain = []
            

            #Here we put a passe-partout (constant length = maxdesl+1) in the images kappa and psi
            base_kappa = np.zeros((2*maxdesl+2+m,2*maxdesl+2+n))
            base_kappa[maxdesl+1:maxdesl+2+m-1,maxdesl+1:maxdesl+2+n-1] = kappa

            psi_base = np.zeros((2*maxdesl+2+m,2*maxdesl+2+n))
            psi_base[maxdesl+1:maxdesl+2+m-1,maxdesl+1:maxdesl+2+n-1] = psi

            base_start = np.zeros((2*maxdesl+2+m, 2*maxdesl+2+n, cliques.shape[0]))
            base_end = np.zeros((2*maxdesl+2+m, 2*maxdesl+2+n, cliques.shape[0]))
            t_dkappa = np.zeros((2*maxdesl+2+m, 2*maxdesl+2+n, cliques.shape[0]))
            a = np.zeros((2*maxdesl+2+m, 2*maxdesl+2+n, cliques.shape[0]))
            A = np.zeros((2*maxdesl+2+m, 2*maxdesl+2+n, cliques.shape[0]))
            D = np.zeros((2*maxdesl+2+m, 2*maxdesl+2+n, cliques.shape[0]))
            C = np.zeros((2*maxdesl+2+m, 2*maxdesl+2+n, cliques.shape[0]))
            B = np.zeros((2*maxdesl+2+m, 2*maxdesl+2+n, cliques.shape[0]))
            source = np.zeros((2*maxdesl+2+m, 2*maxdesl+2+n, cliques.shape[0]))
            sink = np.zeros((2*maxdesl+2+m, 2*maxdesl+2+n, cliques.shape[0]))

            for t in range(cliquesm):

                # The allowed start and end pixels of the "interpixel" directed edge
                base_start[:, :, t] = np.roll(base, (-(cliques[t, 0]), -(cliques[t, 1])), axis=(0, 1)) * base
                base_end[:, :, t] = np.roll(base, (cliques[t, 0], cliques[t, 1]), axis=(0, 1)) * base

                '''
                By convention the difference images have the same size as the
                original ones; the difference information is retrieved in the
                pixel of the image that is subtracted (end of the diff vector)
                '''
                auxili = np.roll(base_kappa, (cliques[t, 0], cliques[t, 1]), axis=(0, 1))
                t_dkappa[:, :, t] = (base_kappa - auxili)

                auxili2 = np.roll(psi_base, (cliques[t, 0], cliques[t, 1]), axis=(0, 1))
                dpsi = auxili2 - psi_base

                '''
                Beyond base, we must multiply by
                circshift(base,[cliques(t,1),cliques(t,2)]) in order to
                account for frontier pixels that can't have links outside ROI
                '''
                a[:, :, t] = (2 * np.pi * t_dkappa[:, :, t] - dpsi) * base * np.roll(base, (cliques[t, 0], cliques[t, 1]), axis=(0, 1))
                                                                                    
                A[:, :, t] = clique_energy_ho(np.abs(a[:, :, t]), p, th, quant) * base * np.roll(base, (cliques[t, 0], cliques[t, 1]), axis=(0, 1))
                    
                D[:, :, t] = np.copy(A[:, :, t])
                C[:, :, t] = clique_energy_ho(np.abs(2 * np.pi * jump_size + a[:, :, t]), p, th, quant) * base * np.roll(base, (cliques[t, 0], cliques[t, 1]), axis=(0, 1))
                    
                B[:, :, t] = clique_energy_ho(np.abs(-2 * np.pi * jump_size + a[:, :, t]), p, th, quant) * base * np.roll(base, (cliques[t, 0], cliques[t, 1]), axis=(0, 1))
                    
                
                # The circshift by [-cliques(t,1),-cliques(t,2)] is due to the fact that differences are retrieved in the
                # "second=end" pixel. Both "start" and "end" pixels can have source and sink connections.
                source[:, :, t] = np.roll(((C[:, :, t] - A[:, :, t]) * ((C[:, :, t] - A[:, :, t]) > 0).astype(int)), (-cliques[t, 0], -cliques[t, 1]), axis=(0, 1)) * base_start[:, :, t]
                                                                                                           
                sink[:, :, t] = np.roll(((A[:, :, t] - C[:, :, t]) * ((A[:, :, t] - C[:, :, t]) > 0).astype(int)), (-cliques[t, 0], -cliques[t, 1]), axis=(0, 1)) * base_start[:, :, t]
                
                source[:, :, t] += np.roll((D[:, :, t] - C[:, :, t]) * ((D[:, :, t] - C[:, :, t]) > 0).astype(int), (-cliques[t, 0], -cliques[t, 1]), axis=(0, 1)) * base_end[:, :, t]
                                                                                                            
                sink[:, :, t] += np.roll((C[:, :, t] - D[:, :, t]) * ((C[:, :, t] - D[:, :, t]) > 0).astype(int), (-cliques[t, 0], -cliques[t, 1]), axis=(0, 1)) * base_end[:, :, t]
                                                                                                          
        
            # Get rid of the "passe-partout"
            # Tạo mảng chỉ số kiểu số nguyên
            indices_rows = (np.r_[:maxdesl+1, m:m+maxdesl+1]).astype(int)
            indices_cols = (np.r_[:maxdesl+1, n:n+maxdesl+1]).astype(int)

            # Cắt bỏ các vị trí thừa
            source = np.delete(source, indices_rows, axis=0)
            source = np.delete(source, indices_cols, axis=1)

            sink = np.delete(sink, indices_rows, axis=0)
            sink = np.delete(sink, indices_cols, axis=1)

            auxiliar1 = B + C - A - D

            auxiliar1 = np.delete(auxiliar1, indices_rows, axis=0)
            auxiliar1 = np.delete(auxiliar1, indices_cols, axis=1)

            base_start = np.delete(base_start, indices_rows, axis=0)
            base_start = np.delete(base_start, indices_cols, axis=1)

            base_end = np.delete(base_end, indices_rows, axis=0)
            base_end = np.delete(base_end, indices_cols, axis=1)

            # Construct the "remain" and the "sourcesink" matrices
            remain_list = []

            for t in range(cliquesm):
                start = np.where(base_end[:,:,0] != 0)[0]
                endd = np.where(base_end[:,:,0] != 0)[0]
                auxiliar2 = auxiliar1[:,:,t]

                endd_python = list(endd - 1)

                au2_m, au2_n = auxiliar2.shape
                au2_raw = []
                for col in range(au2_n):
                    for row in range(au2_m):
                        au2_raw.append(auxiliar2[au2_m - 1][au2_n - 1])


                auxiliar2_endd = []
                for arg in endd_python:
                    if arg > auxiliar2.shape[0] - 1:
                        auxiliar2_endd.append(0)
                    else:
                        auxiliar2_endd.append(np.array(au2_raw[arg]))


                auxiliar2_endd = np.array(auxiliar2_endd)

                auxiliar3 = np.column_stack((start, endd, auxiliar2_endd, np.zeros_like(endd)))

                # Append auxiliar3 to remain_list
                remain_list.append(auxiliar3)
                

            remain = np.concatenate(remain_list, axis=0)
            
            sourcefinal = np.sum(source, axis=2)
            sinkfinal = np.sum(sink, axis=2)

            sourcesink = np.column_stack((np.arange(1, m*n + 1), sourcefinal.flatten(), sinkfinal.flatten()))

            # Kappa relabeling
            flow, cutside = mincut(sourcesink, remain)
            
            # Chuyển đổi cột 1 của cutside về dạng mảng 1 chiều (flattened)
            cutside_flattened = cutside[:, 0] - 1  # Trừ 1 vì Python bắt đầu từ 0

            # Lấy giá trị từ kappa theo cutside_flattened và reshape thành mảng 2 chiều
            kappa_aux = kappa.flatten()[cutside_flattened]
            kappa_aux = np.reshape(kappa_aux, kappa.shape)

            multiplier = (1 - cutside[:, 1]) * jump_size
            multiplier_2d = multiplier.reshape((kappa.shape[0], kappa.shape[1]))


            # Thực hiện ánh xạ lại nhãn theo quy tắc tương tự như MATLAB
            kappa_aux += multiplier_2d

            # Check energy improvement
            erg_actual = energy_ho(kappa_aux, psi, base, p, cliques, disc_bar, th, quant)

            if (int(erg_actual - erg_previous)) == 0:
                possible_improvment = 0
                unwph = 2 * np.pi * kappa + psi
                
            else:
                erg_previous = erg_actual
                kappa = kappa_aux

            # Clear variables
            del base_start, base_end, t_dkappa, a, A, B, C, D

    return unwph, iter, erglist

                


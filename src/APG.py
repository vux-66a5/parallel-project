import os
import time
import numpy as np
import h5py
from matplotlib import pyplot as plt

def APG(x_init, F, dF, R, proxR, gamma, n_iters, opts):
    '''
    =====================================================
    The accelerated proximal gradient (APG) algorithm aiming to solve the
    optimization problems in the following form:
                min J(x) = F(x) + R(x),
                 x
    where F(x) is differentiable and R(x) is non-differentiable.
    ---------------------------------------------------------------------
    Input:      - x_init    : Initial estimate.
                - F         : Function handle for F(x).
                - dF        : Function handle for the gradient of F(x).
                - R         : Function handle for R(x).
                - proxR     : Function handle for the proximal operator of R(x).
                - gamma     : The step size.
                - n_iters   : Numbers of iterations.
                - opts      : Optional settings
    Output:     - x         : The estimated solution x.
                - J_vals    : The values of J(x) during the iterations.
                - E_vals    : The values of the error function during the iterations.
                - runtimes  : The runtimes stored during the iterations.
    '''

    # initialization
    x = x_init
    z = x
    J_vals = np.full(n_iters + 1, np.nan)
    E_vals = np.full(n_iters + 1, np.nan)
    runtimes = np.full(n_iters, np.nan)


    if F(x) == None:
        J_vals[0] = R(x)
    else:
        J_vals[0] = F(x) + R(x)
        
    if hasattr(opts, 'errfunc') and callable(opts.errfunc):
        E_vals[0] = opts.errfunc(z)
    
    if hasattr(opts, 'display') and opts.display:
        fig = plt.figure()
        fig.set_unit('normalized')
        fig.set_position([0.2, 0.2, 0.6, 0.5])

    if hasattr(opts, 'autosave') and opts.autosave:
        foldername = 'cache'
        while os.path.exists(foldername):
            foldername += '_new'
        os.mkdir(foldername)

    timer = time.time()

    for iter in range(n_iters):
        # gradient projection update
        x_next = proxR(z - gamma*dF(z), gamma)
        if F(x_next) == None:
            J_vals[iter+1] = R(x_next)
        else:
            J_vals[iter+1] = F(x_next) + R(x_next)
        
        z = x_next + (iter/(iter+3))*(x_next - x)

        # record runtime 
        runtimes[iter] = time.time() - timer

        # calculate error metric
        if hasattr(opts, 'errfunc') and callable(opts.errfunc):
            E_vals[iter+1] = opts.errfunc(z)

        # print status
        if hasattr(opts, 'verbose') and opts.verbose:
            print('iter: %4d | objective: %10.4e | stepsize: %2.2e | runtime: %5.1f s' % (iter, J_vals[iter + 1], gamma, runtimes[iter]))
        
        if hasattr(opts, 'autosave') and opts.autosave:
            filename = f"{foldername}/iter_{iter}.h5"
            with h5py.File(filename, 'w') as file:
                file.create_dataset('x', data=x)

        x = x_next

        # display intermediate results 
        if hasattr(opts, 'display') and opts.display:
            plt.subplot(1, 2, 1)
            plt.imshow(np.abs(x[:, :, 0]), cmap='gray')  # Chọn một colormap phù hợp
            plt.colorbar()

            plt.subplot(1, 2, 2)
            plt.imshow(np.angle(x[:, :, 0]), cmap='hsv')  # Chọn một colormap phù hợp
            plt.colorbar()

            plt.show()

        return x, J_vals, E_vals, runtimes



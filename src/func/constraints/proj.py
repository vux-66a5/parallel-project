import numpy as np

constraint = 'None'
absorption = 'None'
support = 'None'

def proj(x):
    '''
    =================================================================
    Calculate the projection operator onto the constraint set.

    Input:    - x : The 2D image to be projected.
    Output:   - y : Projection of x onto the constraint set.
    =================================================================
    '''
    global constraint
    global absorption
    global support

    y = x

    if constraint == 'None':
        return y
    elif constraint.lower() == 'a':
        y = np.minimum(np.abs(x), absorption) * np.exp(1j * np.angle(x))
    elif constraint.lower() == 's':
        y = x * support
    elif constraint.lower() == 'as':
        y = np.minimum(np.abs(x), absorption) * np.exp(1j * np.angle(x)) * support
    else:
        raise ValueError("Invalid constraint. Should be 'A'(absorption), 'S'(support), 'AS'(both), or 'none'.")

    return y
import numpy as np

constraint = 'None'
absorption = 'None'
support = 'None'

def indicator(x):
    '''
    ===============================================================================
    Calculate the indicator function of the constraint set.
    -------------------------------------------------------------------------------
    Input:    - x   : The complex-valued 2D transmittance of the sample.
    Output:   - val : The function value.
    ===============================================================================
    '''
    global constraint
    global absorption
    global support

    val = 0

    if constraint == 'None':
        return val 
    elif constraint.lower() == 'a':
        if np.sum(np.sum(np.abs(x) > absorption + np.finfo(np.float64).eps)) == 0:
            val = 0
        else:
            val = np.inf
    elif constraint.lower() == 's':
        x_res = x - x * support
        if np.sum(np.sum(np.abs(x_res))) == 0:
            val = 0 
        else:
            val = np.inf
    elif constraint.lower() == 'as':
        x_res = x - x * support
        if sum(sum(abs(x_res))) == 0 and np.sum(np.sum(np.abs(x) > absorption + np.finfo(np.float64).eps)) == 0:
            val = 0
        else:
            val = np.inf
    else:
        raise ValueError("Invalid constraint. Should be 'A'(absorption), 'S'(support), 'AS'(both), or 'none'.")
    

    print(val)

    return val

'''
data = [
    [-0.1993 - 0.1069j, 0.2086 + 0.1119j, -0.2301 - 0.1053j, 0.2621 + 0.0863j, -0.3008 - 0.0524j],
    [0.1591 + 0.1649j, -0.1190 - 0.1023j, 0.0835 + 0.0387j, -0.0455 + 0.0183j, 0.0008 - 0.0598j],
    [-0.1354 - 0.2339j, 0.0375 + 0.1128j, 0.0613 - 0.0018j, -0.1722 - 0.0829j, 0.2982 + 0.1216j],
    [0.1389 + 0.3152j, 0.0302 - 0.1537j, -0.2022 + 0.0137j, 0.3898 + 0.0792j, -0.5924 - 0.0952j],
    [-0.1831 - 0.4035j, -0.0707 + 0.2313j, 0.3242 - 0.0915j, -0.5882 + 0.0197j, 0.8546 - 0.0536j]
]

print(indicator(data))'''
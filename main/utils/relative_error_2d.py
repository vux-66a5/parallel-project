path_name = "C:\\Users\\vuxxw\\PycharmProjects\\Group16\\parallel-project"
import numpy as np
import sys

sys.path.insert(0, path_name + '\\main\\utils\\puma')
from puma.puma_ho import puma_ho

def relative_error_2d(x_es,x_gt,region):

    def norm2(x):
        val = np.linalg.norm(x.flatten(), 2)
        return val
    
    x_es = x_es[region['x1']:region['x2'], region['y1']:region['y2']]
    x_gt = x_gt[region['x1']:region['x2'], region['y1']:region['y2']]

    amp_es = np.abs(x_es)
    pha_es = puma_ho(np.angle(x_es), 1)

    pha_gt = np.angle(x_gt)
    pha_es = pha_es - np.mean(pha_es) + np.mean(pha_gt)

    x_es = amp_es * np.exp(1j * pha_es)

    re = norm2(x_es - x_gt) / norm2(x_gt)

    return re
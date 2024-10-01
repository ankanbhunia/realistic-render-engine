import numpy as np
from scipy.optimize import linear_sum_assignment

cost_matrix = np.random.rand(10,4) #### IoU matrix, row is pred_seg, col is gt
r, c = linear_sum_assignment(-cost_matrix) ### We want to match the higher IoU
#### keeping the pred_seg indices (r) with corresponding gt (c)
#### if num(gt) > num(pred_seg), assign all gt without match to blank mask
import utils
import numpy as np

a = np.random.rand(5,5)

if len(a.shape) == 2 or a.shape[2] == 1:
    d = np.dstack((a,a,a))
print(d.shape)
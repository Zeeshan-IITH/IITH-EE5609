import numpy as np
a=np.array([[1,1,1],[1,2,-3],[2,-1,3]])
print(np.poly(np.linalg.eigvals(a)))

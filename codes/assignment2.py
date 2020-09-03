import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
fig.set_size_inches(10,10)
ax = fig.add_subplot(111, projection='3d')

x_1=np.array([-10,10])
y_1=np.array([13,-7])
z_1=np.array([-10,10])

x_2=np.array([-10,10])
y_2=np.array([-7,3])
z_2=np.array([-13,7])
ax.plot(x_1,y_1,z_1,color='green',label='line_1')
ax.plot(x_2,y_2,z_2,color='red',label='line 2')
plt.show()

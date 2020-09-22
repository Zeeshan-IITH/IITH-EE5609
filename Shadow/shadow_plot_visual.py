# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 23:35:30 2020

@author: Zeeshan
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

point  = np.array([1, 2, 3])
normal = np.array([1, 1, 2])

# a plane is a*x+b*y+c*z+d=0
# [a,b,c] is the normal. Thus, we have to calculate
# d and we're set
d = -point.dot(normal)

# create x,y
xx, yy = np.meshgrid(range(10), range(10))

# calculate corresponding z
z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]

dir_rays = np.array( [-0.2, -0.5, -1] ).transpose()


obj=np.array([5,3,5]).transpose()
x=-(np.dot(normal,obj))/(np.dot(normal,dir_rays))

points=np.zeros((3,100))

for i in range(100):
    points[:,i]=obj+i*(x/100)*dir_rays
x_co=points[0,:]
y_co=points[1,:]
z_co=points[2,:]
# plot the surface

plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(xx, yy, z)
plt3d.set_xlabel('X Label')
plt3d.set_ylabel('Y Label')
plt3d.set_zlabel('Z Label')
plt3d.scatter(x_co,y_co,z_co,color='red')
plt.show()

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 21:31:23 2020

@author: Zeeshan
"""
"""
Code to draw a picture of the shadow of a 3d object

grid_3d is a discretized version of a cube
hemisphere_3d is a discretized version of a hemisphere
sphere_3d is a discretized version of a sphere
These are all 3-by-N matrices, where N is the number of discrete points representing the object.

orth_basis is an orthonormal basis representing the plane (the sheet of paper)

dir_rays is a vector which gives the direction along which you shine the light
dir_rays must be linearly independent of orth_basis

You can choose the object obj_3d as one among the cube, hemisphere or sphere
You can also change the resolution of the grid (resol_val), but making this smaller would make the code run slower.
"""

import numpy as np
import matplotlib.pyplot as plt

# resolution of grid
resol_val = 0.02

# generate a cubical grid in [1,2]^3 with resolution resol_val
#       large values or resol_val might cause problems
grid_1d = np.arange(1,2,resol_val)
gridsize_1d = np.size(grid_1d)
gridsize_1d_sq = gridsize_1d ** 2
grid_3d = np.zeros([3,gridsize_1d**3])
for i in range(gridsize_1d**3):
    grid_3d[0,i] = grid_1d[i%gridsize_1d]
    grid_3d[1,i] = grid_1d[(int(i/gridsize_1d))%gridsize_1d]
    grid_3d[2,i] = grid_1d[(int(i/gridsize_1d_sq))%gridsize_1d]

# generate a hemisphere of radius 0.5 centered at [1.5,1.5,0.5]'
hemisphere_3d = grid_3d.copy()
center_loc = np.array([1.5,1.5,1]).transpose()
for i in range(gridsize_1d**3):
    if np.linalg.norm(hemisphere_3d[:,i]-center_loc)>0.5:
        hemisphere_3d[:,i] = center_loc

#generate a sphere of radius 0.5 centered at [1.5,1.5,1.5]
sphere_3d = grid_3d.copy()
center_loc = np.array([1.5,1.5,1.5]).transpose()
for i in range(gridsize_1d**3):
    if np.linalg.norm(sphere_3d[:,i]-center_loc)>0.5:
        sphere_3d[:,i] = center_loc


# choose which object must be projected
# options: grid_3d, hemisphere_3d, sphere_3d
obj_3d = sphere_3d.copy()
#print(np.shape(obj_3d))

# describe the plane onto which the 3d object is to be projected
# must be described by an orthonormal basis
#orth_basis = np.array([[1/np.sqrt(2),-1/np.sqrt(2),0],[-1.3,-1,1]]).transpose()
orth_basis = np.array([[1.0,0,0],[0,1.0,0]]).transpose()
orth_basis[:,0] = orth_basis[:,0] / np.linalg.norm(orth_basis[:,0]) 
orth_basis[:,1] = orth_basis[:,1] / np.linalg.norm(orth_basis[:,1])
print("Projection onto the plane spanned by the following vectors")
print(orth_basis)

# the direction of the light rays
dir_rays = np.array( [0, 0, -1] ).transpose()
dir_rays = dir_rays/np.linalg.norm(dir_rays)

# Your code goes here
"""
Input to this part is orth_basis, obj_3d and dir_rays
Output is shadow_vecs 2-by-N matrix, where N is the number of points.
"""

shadow_vecs=np.zeros([2,obj_3d.shape[1]])
temp=np.zeros(obj_3d.shape)
A=np.array((orth_basis[:,0].transpose(),orth_basis[:,1].transpose(),-dir_rays.transpose())).transpose()
B=np.linalg.inv(A)
x=B@obj_3d
#################
shadow_vecs[0,:]=x[0,:]
shadow_vecs[1,:]=x[1,:]

## shadow_vecs is a 2-by-N matrix, where N is the number of points. 


plt.plot(shadow_vecs[0,:],shadow_vecs[1,:],'.k')

# some cosmetics, to ensure that the horizontal and vertical axes have the same scale
xmin = np.min(shadow_vecs[0,:])
xmax = np.max(shadow_vecs[0,:])
ymin = np.min(shadow_vecs[1,:])
ymax = np.max(shadow_vecs[1,:])


maxlen = max(xmax-xmin,ymax-ymin)

plt.xlim(xmin-0.2, xmin+maxlen+0.2)
plt.ylim(ymin-0.2, ymin+maxlen+0.2)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

#Program to plot  the asymptotes of a hyperbola
#Code by GVV Sharma
#Released under GNU GPL
#October 3, 2020

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA



#Generating points on a standard hyperbola 
def hyper_gen(y):
	x = np.sqrt(1+y**2)
	return x
#Orthogonal matrix
omat = np.array([[0,1],[-1,0]]) 
def dir_vec(A,B):
  return B-A

def norm_vec(A,B):
  return np.matmul(omat, dir_vec(A,B))

#Generate line points
def line_gen(A,B):
  len =10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB

def line_dir_pt(m,A,k1,k2):
  len = 10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(k1,k2,len)
  for i in range(len):
    temp1 = A + lam_1[i]*m
    x_AB[:,i]= temp1.T
  return x_AB


#Intersection of two lines
def line_intersect(n1,A1,n2,A2):
  N=np.vstack((n1,n2))
  p = np.zeros(2)
  p[0] = n1@A1
  p[1] = n2@A2
  #Intersection
  P=np.linalg.inv(N)@p
  return P

#Intersection of two lines
def perp_foot(n,cn,P):
  m = omat@n
  N=np.block([[n],[m]])
  p = np.zeros(2)
  p[0] = cn
  p[1] = m@P
  #Intersection
  x_0=np.linalg.inv(N)@p
  return x_0
#setting up plot
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
len = 100
y = np.linspace(-2,2,len)

#Coefficients

a = 8
b = 10/2
c = -3
d = -2
e = 4



#hyper parameters
V = np.array(([a,b],[b,c]))
u = 0.5*np.array(([d,e]))
f = -2
Vinv = LA.inv(V)

#Eigenvalues and eigenvectors
D_vec,P = LA.eig(V)
D = np.diag(D_vec)
#print(D_vec)

#Angle between asymptotes
#theta1 = np.arccos((np.sqrt(D_vec[0])-np.sqrt(-D_vec[1]))/(np.sqrt(D_vec[0])+np.sqrt(-D_vec[1])))
theta = (np.pi)/2
Q = np.array(([np.cos(theta),-np.sin(theta)],[np.sin(theta), np.cos(theta)]))

#print(theta,Q)


sig1 = np.sqrt(np.absolute(D_vec))
ineg = np.array(([1, 0],[0,-1]))
sig2 = ineg@sig1
#print(sig1,sig2)

#Direction vectors of the asymptotes
n1 = sig1@P.T
n2 = sig2@P.T
m1 = omat@n1
m2 = omat@n2
#print(n1,n2,m1,m2)

#Constant for asymptotes
K = u.T@Vinv@u
uconst = u.T@Vinv@u-f
#print(K,uconst)


#Hyperbola axis parameters
a = np.sqrt(np.abs(uconst/D_vec[0]))
b = np.sqrt(np.abs(uconst/D_vec[1]))
#print(a,b)

#Generating the Standard Hyperbola
x = hyper_gen(y)
xStandardHyperLeft = np.vstack((-x,y))
xStandardHyperRight = np.vstack((x,y))

#Hyperbola Foci without parameters
V1old = np.array([1,0])
V2old = -V1old

#Affine Parameters
c = -Vinv@u
cuconst=2*np.transpose(c)@V@c-f
print(c)
R =  np.array(([0,1],[1,0]))
ParamMatrix = np.array(([a,0],[0,b]))
cParamMatrix = np.array(([a,0],[0,b]))
if uconst < 0:
	#Generating the eigen hyperbola
	xeigenHyperLeft = R@ParamMatrix@xStandardHyperLeft
	xeigenHyperRight = R@ParamMatrix@xStandardHyperRight

	#Generating the actual hyperbola
	xActualHyperLeft = P@ParamMatrix@R@xStandardHyperLeft+c[:,np.newaxis]
	xActualHyperRight = P@ParamMatrix@R@xStandardHyperRight+c[:,np.newaxis]

	#Generating the actual Foci
	V1 = P@R@ParamMatrix@V1old+c
	V2 = P@R@ParamMatrix@V2old+c
else:
	#Generating the eigen hyperbola
	xeigenHyperLeft = ParamMatrix@xStandardHyperLeft
	xeigenHyperRight = ParamMatrix@xStandardHyperRight

	#Generating the actual hyperbola
	xActualHyperLeft = P@ParamMatrix@xStandardHyperLeft+c[:,np.newaxis]
	xActualHyperRight = P@ParamMatrix@xStandardHyperRight+c[:,np.newaxis]

	#Generating the actual Foci
	V1 = P@V1old+c
	V2 = P@V2old+c
if cuconst > 0:
	#Generating the eigen hyperbola
	cxeigenHyperLeft = R@ParamMatrix@xStandardHyperLeft
	cxeigenHyperRight = R@ParamMatrix@xStandardHyperRight

	#Generating the actual hyperbola
	cxActualHyperLeft = P@ParamMatrix@R@xStandardHyperLeft+c[:,np.newaxis]
	cxActualHyperRight = P@ParamMatrix@R@xStandardHyperRight+c[:,np.newaxis]

	#Generating the actual Foci
	cV1 = P@R@ParamMatrix@V1old+c
	cV2 = P@R@ParamMatrix@V2old+c
else:
	#Generating the eigen hyperbola
	cxeigenHyperLeft = ParamMatrix@xStandardHyperLeft
	cxeigenHyperRight = ParamMatrix@xStandardHyperRight

	#Generating the actual hyperbola
	cxActualHyperLeft = P@ParamMatrix@xStandardHyperLeft+c[:,np.newaxis]
	cxActualHyperRight = P@ParamMatrix@xStandardHyperRight+c[:,np.newaxis]

	#Generating the actual Foci
	cV1 = P@V1old+c
	cV2 = P@V2old+c

#Generating the conjugate Hyperbola
#xActualHyperLeftConj = Q@xActualHyperLeft +c[:,np.newaxis]
#xActualHyperRightConj = Q@xActualHyperRight+c[:,np.newaxis]

#Generating the Asymptotes
k1 = 0.5
k2 = -0.5
x_AB = line_dir_pt(m1,c,k1,k2)

k1 = 0.5
k2 = -0.5
x_BC = line_dir_pt(m2,c,k1,k2)

#Generating the Axis
x_V = line_gen(V1,V2)


#plotting all points
hyper_coords = np.vstack((V1,V2,c)).T
plt.scatter(hyper_coords[0,:], hyper_coords[1,:])
vert_labels = ['$F_1$','$F_2$','$c$']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (hyper_coords[0,i], hyper_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center


#Plotting all lines
plt.plot(x_V[0,:],x_V[1,:],label='Axis')
plt.plot(x_AB[0,:],x_AB[1,:],label='Asymptote 1')
plt.plot(x_BC[0,:],x_BC[1,:],label='Asymptote 2')

#Plotting the eigen hyperbola
#plt.plot(xeigenHyperLeft[0,:],xeigenHyperLeft[1,:],label='Standard hyperbola',color='b')
#plt.plot(xeigenHyperRight[0,:],xeigenHyperRight[1,:],color='b')

#Plotting the actual hyperbola
plt.plot(xActualHyperLeft[0,:],xActualHyperLeft[1,:],label='Actual hyperbola',color='r')
plt.plot(xActualHyperRight[0,:],xActualHyperRight[1,:],color='r')

#Plotting the conjugate hyperbola
plt.plot(cxActualHyperLeft[0,:],cxActualHyperLeft[1,:],label='Actual hyperbola',color='g')
plt.plot(cxActualHyperRight[0,:],cxActualHyperRight[1,:],color='g')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

plt.show()

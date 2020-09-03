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
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')


#vectors parallel to the given lines
v_1=np.array([1,-1,1])
v_2=np.array([2,1,2])

#the points on the lines
x_1=np.array([1,2,1])
x_2=np.array([2,-1,-1])
diffx=x_2-x_1   #Difference between the points i.e. vector along the points

a=np.array([[np.dot(v_1,v_1),-np.dot(v_1,v_2)],[np.dot(v_2,v_1),-np.dot(v_2,v_2)]])
b=np.array([np.dot(v_1,diffx),np.dot(v_2,diffx)])


x=np.linalg.solve(a,b)
print("the vlaue of lambda_1,lambda_2 is ",x[0],x[1])
print(np.allclose(np.dot(a,x),b))

a_1=x_1+np.multiply(x[0],v_1)
b_1=x_2+np.multiply(x[1],v_2)
x_3=np.array([a_1[0],b_1[0]])
y_3=np.array([a_1[1],b_1[1]])
z_3=np.array([a_1[2],b_1[2]])
ax.plot(x_3,y_3,z_3,color='blue',label='line 3')

print("The shortest distance between the lines is ",np.linalg.norm(a_1-b_1))
xa=np.array([a_1[0],b_1[0]])
ya=np.array([a_1[1],b_1[1]])
za=np.array([a_1[2],b_1[2]])

ax.scatter(xa,ya,za)

plt.show()






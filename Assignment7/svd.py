import numpy as np
import math as mt
uc=np.array([[-3/mt.sqrt(14),-2/mt.sqrt(13),1/mt.sqrt(10)],[-2/mt.sqrt(14),3/mt.sqrt(13),0],[1/mt.sqrt(14),0,3/mt.sqrt(13)]])
sc=np.array([[mt.sqrt(70),0],[0,0],[0,0]])
sp=np.array([[1/mt.sqrt(70),0,0],[0,0,0]])
vc=np.array([[-2/mt.sqrt(5),1/mt.sqrt(5)],[1/mt.sqrt(5),2/mt.sqrt(5)]])
print(uc@sc@np.transpose(vc))

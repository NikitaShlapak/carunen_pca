import numpy as np
import matplotlib.pyplot as pl

x11 = np.array([0,0,0])
x12 = np.array([1,0,0])
x13 = np.array([1,0,1])
x14 = np.array([1,1,0])

x21 = np.array([0,0,1])
x22 = np.array([0,1,0])
x23 = np.array([0,1,1])
x24 = np.array([1,1,1])

x1 = np.stack([x11,x12,x13,x14])
x2 = np.stack([x21,x22,x23,x24])
print("x1=\n",x1,"\nx2=\n",x2)
x1m = np.dot(x1.T,x1)
x2m = np.dot(x2.T,x2)
x1=x1.T
x2=x2.T
print("x1=\n",x1,"\nx2=\n",x2)
X = (x1m+x2m)/8
Xtest= np.dot(X.T,X)
print("X=\n",X,"Xt=\n",Xtest)
h1,h2 = np.linalg.eig(X)
h3 = h2.T

a = 4*np.dot(h3[2],X)

ind = np.argpartition(h1, -2)[-2:]

#print(ind,h1)

enums = h1[ind]
evects = h3[ind]

print("enums:\n",x11.shape,"\nevects:\n",evects.shape)

y11 = np.dot(evects,x11)
y12 = np.dot(evects,x12)
y13 = np.dot(evects,x13)
y14 = np.dot(evects,x14)

y21 = np.dot(evects,x21)
y22 = np.dot(evects,x22)
y23 = np.dot(evects,x23)
y24 = np.dot(evects,x24)

y1 = np.stack([y11,y12,y13,y14]).T
y2 = np.stack([y21,y22,y23,y24]).T
print("y1=\n",y1,"\ny2=\n",y2)

fig1, ax1 = pl.subplots(1,1, figsize=(5, 5))
ax1.scatter(y1[0],y1[1], c = "red")
ax1.scatter(y2[0],y2[1], c = "blue")
ax1.grid()
ax1.set_xlim([-1,1])
ax1.set_ylim([0,2])

fig = pl.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(x1[0],x1[1],x1[2], c = "red")
ax.scatter(x2[0],x2[1],x2[2], c = "blue")
import load as data
import numpy as np
import matplotlib.pyplot as pl

from sklearn.cluster import DBSCAN


"""
for pack in data.out03:
    for line in pack:
        print(line,len(line))
    print(len(pack),"\n\n")
"""
X0 = np.array(data.out03[0])
X1 = np.array(data.out03[1])
X2 = np.array(data.out03[2])
X3 = np.array(data.out03[3])
X4 = np.array(data.out03[4])
"""
X0m = np.dot(X0.T,X0)
X1m = np.dot(X1.T,X1)
X2m = np.dot(X2.T,X2)
X3m = np.dot(X3.T,X3)
X4m = np.dot(X4.T,X4)

X = (X0m + X1m + X2m + X3m + X4m)/320
"""
X = data.X

print("corr matrix:\n",X)

h1,h2 = np.linalg.eig(X)
h3 = h2.T

ind = np.argpartition(h1, -2)[-2:]

#print(ind,h1)

enums = h1[ind]
evects = h3[ind]

print("enums:\n",enums,"\nevects:\n",evects)

Y0=np.ones([64,2])
Y0 = np.dot(evects,X0.T)

Y1=np.ones([64,2])
Y1 = np.dot(evects,X1.T)

Y2=np.ones([64,2])
Y2 = np.dot(evects,X2.T)

Y3=np.ones([64,2])
Y3 = np.dot(evects,X3.T)

Y4=np.ones([64,2])
Y4 = np.dot(evects,X4.T)
#print(Y0)

fig, ax = pl.subplots(1,5, figsize=(21, 13))
labels = ["До падения ОР","Середина процесса","Cразу после падения ОР","Перед окончанием процессa","После переходного процесса"]
for plot in ax:
    plot.grid()
    plot.set_ylim([-6,2])
    plot.set_xlim([-8,8])
    j=tuple(ax).index(plot)
    ax[j].set_title(labels[j])

ax[0].scatter(Y0[0],Y0[1], c = "black")

ax[1].scatter(Y1[0],Y1[1], c = "green")

ax[2].scatter(Y2[0],Y2[1], c = "blue")

ax[3].scatter(Y3[0],Y3[1], c = "purple")

ax[4].scatter(Y4[0],Y4[1], c = "red")
"""
data = np.concatenate([Y0.T,Y1.T,Y2.T,Y3.T,Y4.T])

  
#print(data,data.shape)

#pl.scatter(data[0],data[1], c = "blue")


dataset = DBSCAN(eps=0.7, min_samples=7)
dataset.fit(data)
print(dataset.labels_)

from sklearn.decomposition import PCA
fig, ax = pl.subplots(1,1, figsize=(5, 5))
pca = PCA(n_components=2).fit(data)
pca_2d = pca.transform(data)
for i in range(0, pca_2d.shape[0]):
    if dataset.labels_[i] == 0:
        c1 = ax.scatter(pca_2d[i, 0], pca_2d[i, 1], c='red', marker='o')
    elif dataset.labels_[i] == 1:
        c2 = ax.scatter(pca_2d[i, 0], pca_2d[i, 1], c='purple', marker='o')
    elif dataset.labels_[i] == 2:
        c3 = ax.scatter(pca_2d[i, 0], pca_2d[i, 1], c='blue', marker='o')
    
    elif dataset.labels_[i] == -1:
        c6 = ax.scatter(pca_2d[i, 0], pca_2d[i, 1], c='black', marker='+')
 
ax.legend([c1, c2, c3, c6], ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Noise'])
pl.title('DBSCAN finds 4 clusters and noise')
ax.grid()
pl.show()

""" 

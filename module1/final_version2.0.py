import numpy as np
import matplotlib.pyplot as pl
from sklearn.cluster import DBSCAN

import load
import function

for pack in range(len(load.out04)):
    X = np.array(load.out04[pack])
    a=0.228
    b=4
    
    X1=np.array(X)
    #X1 = np.array([line - np.average(line) for line in X1])
    dbscan = DBSCAN(eps=a, min_samples=b)     
    dbscan.fit(X1)
    print(dbscan.labels_,len(dbscan.labels_),max(dbscan.labels_))
    cluters = list(set(dbscan.labels_))
    print("clusters",cluters)
    fig, ax = pl.subplots(1,1,figsize=(13, 8))
     
    pca_2d = function.carunen(X1)
    colors=["red", "green", "blue", "purple","red", "green", "blue", "purple","red", "green", "blue", "purple",'black']
    marker= ['o','o','o','o','*','*','*','*','+','+','+','+','+']
    c=[[],[],[],[],[],[]]
    for i in range(0, pca_2d.shape[0]):
        for j in cluters:
            #print(j)
            if dbscan.labels_[i] == j:
                c[j+1]=ax.scatter(pca_2d[i, 0], pca_2d[i, 1], c=colors[j], marker=marker[j])
    label = {-1:'Noise'} 
    for j in range(len(cluters)-1):
        label[j]="Cluster "+str(j+1)
    #label = {j:"Cluster "+str(j+1) for j in range(len(cluters)-1)}
    
    print(label)
    ax.legend(c, label.values())
    ax.grid()
    str1 = 'Unknown CPR is inserted' +'\n'+ 'found '+str(len(cluters)-1) +' clusters and noise'
    pl.title(str1, {'fontsize' : 25},pad=10)
    ax.set_ylim([-2,2])
    ax.set_xlim([-1,1])
    pl.show()
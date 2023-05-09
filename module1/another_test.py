import numpy as np
import matplotlib.pyplot as pl
import function
from sklearn.cluster import DBSCAN

X = [[4.402, 4.266, 4.18, 4.184, 4.281, 4.344, 4.828],
     [4.531, 4.398, 4.387, 4.332, 4.375, 4.535, 4.875],
     [6.836, 6.684, 6.512, 6.379, 6.402, 6.488, 6.324],
     [6.977, 6.609, 6.512, 6.422, 6.465, 6.684, 6.48],
     [6.199, 5.938, 5.816, 5.773, 5.844, 6.074, 6.023],
     [6.434, 6.121, 6.004, 6.031, 6.113, 6.289, 6.344],
     [0, 5.961, 5.891, 5.738, 5.848, 6.172, 5.922],
     [6.344, 6.098, 0, 5.863, 5.949, 6.051, 5.93],
     [6.055, 5.754, 5.832, 5.73, 5.734, 5.906, 5.914],
     [5.496, 5.332, 5.258, 5.188, 5.223, 5.293, 4.828],
     [5.668, 5.586, 5.457, 5.27, 5.352, 5.344, 4.82],
     [5.746, 5.613, 5.418, 5.41, 5.496, 5.465, 5.012],
     [5.309, 4.973, 5.074, 4.984, 5.051, 5.355, 5.211],
     [5.129, 4.91, 4.836, 4.863, 4.945, 5.18, 4.953],
     [5.16, 4.957, 4.887, 4.824, 4.906, 5.09, 5.219],
     [5.074, 4.863, 4.895, 4.953, 4.926, 5.125, 5.031],
     [6.352, 6.047, 5.906, 5.949, 5.988, 6.18, 5.934],
     [6.34, 5.988, 5.969, 5.977, 5.996, 6.137, 5.883],
     [6.281, 6.051, 5.996, 6.004, 6.047, 6.203, 5.93],
     [6.348, 5.941, 5.934, 5.914, 5.934, 6.141, 5.875],
     [6.457, 6.176, 6.07, 6.016, 6.125, 6.188, 6.07],
     [6.301, 6.062, 6, 5.977, 5.965, 6.254, 6.078],
     [2.09, 2.039, 1.977, 1.984, 2.035, 2.102, 2.008],
     [2.027, 1.938, 1.961, 1.977, 2, 2.082, 1.941],
     [0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0],
     [6.867, 6.801, 6.621, 6.484, 6.562, 6.668, 6.105],
     [6.859, 6.758, 6.508, 6.465, 6.48, 6.512, 5.945],
     [7.031, 6.934, 6.711, 6.645, 6.727, 6.773, 6.094],
     [0, 0, 0, 0, 0, 0, 0],
     [3.234, 0, 2.922, 2.93, 2.988, 3, 2.973],
     [5.574, 5.422, 5.312, 5.234, 5.238, 5.344, 4.762],
     [4.508, 4.312, 4.25, 4.242, 4.336, 4.508, 4.73],
     [4.551, 4.426, 4.285, 4.301, 4.344, 4.473, 4.762],
     [6.5, 6.652, 6.371, 6.211, 6.277, 6.332, 6.266],
     [0, 6.969, 6.711, 6.754, 0, 6.672, 6.535],
     [6.301, 5.945, 5.914, 5.766, 0, 6.035, 6.156],
     [6.215, 5.945, 5.898, 6.012, 6.152, 6.23, 6.246],
     [6.008, 5.82, 5.605, 5.578, 5.688, 5.914, 5.781],
     [6.18, 5.898, 5.773, 5.785, 5.773, 5.91, 5.832],
     [6.199, 6.078, 5.793, 5.805, 5.938, 6.098, 6.051],
     [5.57, 5.488, 5.211, 5.258, 5.207, 5.258, 4.797],
     [5.551, 5.383, 5.328, 5.262, 5.297, 5.277, 4.789],
     [5.516, 5.535, 5.391, 5.25, 5.289, 5.352, 4.77],
     [5.316, 0, 4.906, 4.969, 5.047, 5.266, 5.387],
     [5.398, 5.09, 5.031, 5.062, 5.086, 5.332, 5.199],
     [5.324, 4.984, 4.926, 5.023, 5.02, 5.297, 5.441],
     [5.152, 4.84, 4.852, 0, 4.922, 5.152, 5.184],
     [6.281, 6.156, 5.988, 6.031, 5.977, 6.203, 5.91],
     [6.266, 6.039, 5.836, 5.902, 5.98, 6.152, 5.773],
     [6.324, 6.18, 6.016, 6.055, 6.078, 6.191, 6.023],
     [6.281, 6.055, 5.957, 6.012, 5.98, 6.184, 5.801],
     [6.25, 6.035, 6.168, 5.906, 6.082, 6.211, 6.02],
     [6.301, 6.098, 0, 5.988, 5.973, 6.191, 5.918],
     [2.098, 1.988, 2.023, 2, 2.113, 2.172, 2.02],
     [2.059, 1.957, 1.965, 1.953, 2.023, 2.07, 1.945],
     [7.02, 6.691, 0, 0, 6.406, 6.508, 5.945],
     [7.031, 6.797, 6.594, 6.562, 6.598, 6.68, 5.953],
     [0, 0, 0, 0, 0, 0, 0],
     [6.934, 6.773, 6.629, 6.574, 6.453, 6.535, 6.176],
     [6.852, 6.633, 6.625, 6.504, 6.551, 6.816, 6.074],
     [7.012, 6.691, 6.559, 6.477, 6.484, 6.621, 6.027],
     [2.578, 2.477, 2.445, 0, 2.57, 2.688, 2.641],
     [0, 5.359, 5.258, 5.285, 5.262, 5.305, 4.703]]

X1 = np.array(X)
X1 = np.array([line - np.average(line) for line in X1])
print(X1, X1.shape)



dbscan = DBSCAN(eps=0.5, min_samples=5)

dbscan.fit(X1)
print(dbscan.labels_, len(dbscan.labels_), max(dbscan.labels_))
cluters = list(set(dbscan.labels_))
print(cluters)
fig, ax = pl.subplots(1, 1)

pca_2d = function.carunen(X1)
colors = ["black", "red", "green", "blue", "purple", "red", "green", "blue", "purple", "red", "green", "blue", "purple"]
marker = ["+", 'o', 'o', 'o', 'o', '*', '*', '*', '*', '+', '+', '+', '+']
c = [[], []]
for i in range(0, pca_2d.shape[0]):
    for j in cluters:
        if dbscan.labels_[i] == j:
            c[cluters[j] + 1] = ax.scatter(pca_2d[i, 0], pca_2d[i, 1], c=colors[j + 1], marker=marker[cluters[j] + 1])

label = ["Cluster " + str(j + 2) for j in cluters]
label[0] = 'Noise'
ax.legend(c, label)
ax.grid()
str1 = 'DBSCAN finds ' + str(max(dbscan.labels_) + 1) + '  clusters and noise'
pl.title(str1)
ax.set_ylim([-10, 10])
ax.set_xlim([-10, 10])
pl.show()

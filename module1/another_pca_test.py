import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import load


def carunen(X, n=2):
    # X = np.array(X)
    Xcor = np.corrcoef(X, rowvar=False)
    for i in range(Xcor.shape[0]):
        for j in range(Xcor.shape[1]):
            # print(Xcor[i,j], Xcor[i,j]==np.nan, str(Xcor[i,j])=='nan')
            if str(Xcor[i,j])=='nan':
                print(Xcor[i,j],i,j)
                Xcor[i,j] = 0
    print(Xcor[0])

    h1, h2 = np.linalg.eig(Xcor)
    h3 = h2.T

    ind = np.argpartition(h1, -n)[-n:]

    enums = h1[ind]
    evects = h3[ind]

    Y = np.ones([len(X), n])
    Y = np.dot(evects, X.T).T

    return (Y)

X = np.array(load.out04).reshape(5,60*7)
# print(X)
# X1 = np.array(X)
pca_2d = carunen(X,11)
print(pca_2d)


# for pack in range(len(load.out04)):
#     X = np.array(load.out04[pack])
#     X1 = np.array(X)
#     pca_2d = carunen(X1,11)
#     print(pca_2d)
# print(np.array(load.out04)[0])

# pca = PCA(n_components=11)

# comps= carunen(data,11)
# print(comps)
# state_components = []

# for i in range(len(data)):
#     state = data[i]
#     pca.fit(state)
#     state_components.append(pca.singular_values_)
# plot_data = []
# i = 0
# colors = ['r', 'g', 'b', 'black', [1, 1, 0]]
#
# fig, ax = plt.subplots()
# for state in state_components:
#     ax.scatter(x=state[3], y=state[6], c=colors[i])
#     i = i + 1
#     plot_data.append([state[3], state[6]])
# plot_data = np.array(plot_data).T
# ax.set_xlim([-3.5, 3.5])
# ax.set_ylim([-1.5, 1.5])
# ax.grid(True)
# plt.show()

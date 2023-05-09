import numpy as np

def carunen (X,n=2):
    X=np.array(X)
    Xcor = np.corrcoef(X, rowvar=False)

    h1,h2 = np.linalg.eig(Xcor)
    h3 = h2.T
    
    ind = np.argpartition(h1, -n)[-n:]

    enums = h1[ind]
    evects = h3[ind]

    
    Y = np.ones([len(X),n])
    Y = np.dot(evects,X.T).T

    return(Y)

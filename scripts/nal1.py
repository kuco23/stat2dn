import numpy as np
from scipy.stats import norm as N
from scipy.linalg import schur
import matplotlib.pyplot as plt
from matplotlib import cm
from defaultAx import getAx

def getData(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            data.append(float(line.strip()))
    return data

moment = lambda X, k:  sum([x**k for x in X]) / len(X)

def g(x1,x2,x3):
    b = (x1 / x2 - x2 /x3)**(-1)
    a = b / x1 + 1
    return (a, b)

def JordanG(x1,x2,x3):
    t = (x1 * x3 - x2**2)**2
    J12 = (x1 * x3**2 - 3 * x2**2 * x3) / t
    J13 = -x2**2 / t
    return np.array([
        [-x2 * x3 / t, J12, J13],
        [x2 / x1**2 * x3, J12 / x1, J13 / x1]
    ])

def confidenceArea(X, alpha):
    alpha2 = (1 - np.sqrt(1 - alpha))/2
    nalpha2 = N.ppf(1 - alpha2)
    
    m = {i: moment(X,i) for i in range(1,7)}
    E = np.array([
        [m[2] - m[1]**2, m[3] - m[1]*m[2], m[4] - m[1]*m[3]],
        [m[3] - m[1]*m[2], m[4] - m[2]**2, m[5] - m[2]*m[3]],
        [m[4] - m[1]*m[3], m[5] - m[2]*m[3], m[6] - m[3]**2]
    ])
    Jg = JordanG(m[1], m[2], m[3])
    A = Jg @ E @ Jg.T
    L, Q = schur(A)

    L = np.diagflat(np.sqrt(np.diagonal(L)))
    return ((Q @ L @ Q.T) / np.sqrt(len(X)), g(m[1], m[2], m[3]), nalpha2)
    
if __name__ == '__main__':
    alpha = 0.05
    k = 300
    file = 'podatki_1.txt'
    X = getData(file)
    A, theta, nalpha2 = confidenceArea(X, alpha)

    P1 = np.zeros((k,k))
    P2 = np.zeros((k,k))
    t = np.linspace(-nalpha2, nalpha2, k)
    for i in range(k):
        for j in range(k):
            u = np.array([t[i], t[j]])
            v = theta + A @ u
            P1[i,j] = v[0]
            P2[i,j] = v[1]

    ax = getAx('', 'a', 'b')
    ax.scatter(P1.flatten(), P2.flatten(), s=1, c='gray')
    plt.show()
    

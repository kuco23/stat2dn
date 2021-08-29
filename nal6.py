from operator import truediv
import math
import numpy as np
from scipy.linalg import lstsq, schur
from scipy.stats import f as F
from scipy.stats import t as T
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D

def getData(path):
    first = True
    names = []
    data = []
    with open(path, 'r') as file:
        for line in file:
            sline = line.split()
            if first:
                names = sline
                first = False
            else:
                data.append(list(map(float, sline)))
    return (names, data)

def FTestConstants(X,y,b,alpha):
    m, n = X.shape[1]-1, len(y)
    yavg = sum(y) / n
    yaprx = np.matmul(X, b)
    Fstat = truediv(
        sum([(yaprx[i] - yavg)**2 / m for i in range(n)]),
        sum([(y[i] - yaprx[i])**2 / (n-m-1) for i in range(n)])
    )
    Falpha = F.ppf(1-alpha, m, n-m-1)
    return (Fstat, Falpha)

def TTestConstants(X,y,b,alpha):
    m, n = X.shape[1]-1, len(y)
    e0 = np.zeros(m+1)
    e0[0] = 1
    XtX = np.matmul(np.transpose(X), X)
    invXtX00 = np.linalg.solve(XtX, e0)[0]
    vkr = np.linalg.norm(y - np.matmul(X, b))
    Tstat = b[0] / (math.sqrt(invXtX00 / (n-m-1)) * vkr)
    Talpha2 = T.ppf(1-alpha/2, n-m-1)
    return (Tstat, Talpha2)

def linearRegresh(path, const=True):
    names, data = getData(path)
    data = np.asarray(data)
    X = data[:,:-1]
    y = data[:, -1]
    if const:
        X = np.insert(X, 0, values=np.ones(len(y)), axis=1)
    b = lstsq(X,y)[0]
    return (X, y, b)

def getConfidenceElipsoid(X, y, b):
    n, m = X.shape
    XtX = np.matmul(np.transpose(X),X)
    S, Q = schur(XtX)
    Falpha = F.ppf(1-alpha, m, n-m)
    vkr = np.linalg.norm(y - np.matmul(X, b))**2
    r = (m * vkr * Falpha) / (n - m)
    L = np.diagonal(S)
    return (Q, L, r)

def drawElipsoid(A, b, r, k=30):
    sqrtr = math.sqrt(r)

    # get data for a sphere
    T1 = np.linspace(0, 2 * np.pi, k)
    T2 = np.linspace(0, 2 * np.pi, k)
    X = np.zeros((k,k))
    Y = np.zeros((k,k))
    Z = np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            X[i,j] = np.cos(T1[i]) * np.sin(T2[j])
            Y[i,j] = np.sin(T1[i]) * np.sin(T2[j])
            Z[i,j] = np.cos(T2[i])

    # construct the elipsoid
    U1 = np.zeros((k,k))
    U2 = np.zeros((k,k))
    U3 = np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            xyz = np.array([X[i,j], Y[i,j], Z[i,j]])
            u = b + np.matmul(A, sqrtr * xyz)
            [U1[i,j], U2[i,j], U3[i,j]] = u

    # triangulate
    [T1, T2] = np.meshgrid(T1, T2)
    X,Y,Z = X.flatten(), Y.flatten(), Z.flatten()
    tri = Delaunay(np.array([X, Y, Z]).T)
    print(tri.simplices)
    
    # plot the elipsoid
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(
        X, Y, Z,
        triangles=tri.simplices
    )
    plt.show()

def testIfInInterval(X, y, b, Q, L, b0):
    (n, m) = X.shape
    L = np.diagflat(np.sqrt(L))
    vkr = np.linalg.norm(y - X @ b)**2
    Falpha = F.ppf(1-alpha, m, n-m)
    v1 = np.linalg.norm((L @ Q.T) @ (b - b0))**2
    v2 = m * vkr * Falpha / (n - m)
    return (v1, v2)

def getBonferi(X,y,b,alpha):
    m, n = X.shape[1], len(y)
    I = []
    for j in range(m):
        ej = np.zeros(m)
        ej[j] = 1
        invXtX00 = np.linalg.solve(X.T @ X, ej)[j]
        vkr = np.linalg.norm(y - X @ b)
        Talpha2 = T.ppf(1-alpha/(2*m), n-m)
        Tstat = Talpha2 * vkr * math.sqrt(invXtX00 / (n-m))
        I.append((b[j] - Tstat, b[j] + Tstat))
    return I
        

if __name__ == '__main__':
    alpha = 0.05
    path = 'podatki_6.txt'
    
    X, y, b = linearRegresh(path, False)
    #Q, L, r = getConfidenceElipsoid(X,y,b)
    #drawElipsoid(np.matmul(Q, np.diagflat(np.power(L, -1/2))), b, math.sqrt(r))

    #print(testIfInInterval(X, y, b, Q, L, np.array([1, -1, -0.45])))
        
    I = getBonferi(X, y, b, alpha)


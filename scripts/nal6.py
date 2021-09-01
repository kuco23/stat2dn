from operator import truediv
import math
from itertools import product
import numpy as np
from scipy.linalg import lstsq, schur
from scipy.stats import f as F
from scipy.stats import t as T
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import cm
from matplotlib.colors import LightSource
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
    data = np.asarray(data)
    return (names, data[:,:-1], data[:,-1])

def linearRegresh(X, y, const=True):
    if const: X = np.insert(X, 0, values=np.ones(len(y)), axis=1)
    return lstsq(X,y)[0]

def FTestConstants(X,y,b,alpha):
    m, n = X.shape[1]-1, len(y)
    yavg, yaprx = sum(y) / n, X @ b
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
    invXtX00 = np.linalg.solve(X.T @ X, e0)[0]
    vkr = np.linalg.norm(y - X @ b)
    Tstat = b[0] / (math.sqrt(invXtX00 / (n-m-1)) * vkr)
    Talpha2 = T.ppf(1-alpha/2, n-m-1)
    return (Tstat, Talpha2)

def getConfidenceElipsoidData(X, y, b):
    n, m = X.shape
    S, Q = schur(X.T @ X)
    Falpha = F.ppf(1-alpha, m, n-m)
    vkr = np.linalg.norm(y - X @ b)**2
    r = math.sqrt((m * vkr * Falpha) / (n - m))
    L = np.diagonal(S)
    return (Q, L, r)

def getBonferiCorrection(X,y,b,alpha):
    n, m = X.shape
    I = []
    for j in range(m):
        ej = np.zeros(m)
        ej[j] = 1
        invXtXjj = np.linalg.solve(X.T @ X, ej)[j]
        vkr = np.linalg.norm(y - X @ b)
        Talpha2 = T.ppf(1-alpha/(2*m), n-m)
        Tstat = Talpha2 * vkr * math.sqrt(invXtXjj / (n-m))
        I.append((b[j] - Tstat, b[j] + Tstat))
    return I

def drawElipsoid(ax, A, b, k=30):
    # sphere parametrization
    T = np.linspace(0, 2 * np.pi, k)
    X = np.zeros((k,k))
    Y = np.zeros((k,k))
    Z = np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            X[i,j] = np.cos(T[i]) * np.sin(T[j])
            Y[i,j] = np.sin(T[i]) * np.sin(T[j])
            Z[i,j] = np.cos(T[j])

    # map sphere to elipsoid
    U1 = np.zeros((k,k))
    U2 = np.zeros((k,k))
    U3 = np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            xyz = np.array([X[i,j], Y[i,j], Z[i,j]])
            [U1[i,j], U2[i,j], U3[i,j]] = b + A @ xyz

    # triangulate the elipsoid
    [T1, T2] = np.meshgrid(T, T)
    tri = Delaunay(np.array([T1.flatten(), T2.flatten()]).T)
    
    # plot the elipsoid
    ax.plot_trisurf(
        U1.flatten(), U2.flatten(), U3.flatten(),
        triangles=tri.simplices, cmap=cm.inferno,
        lightsource = LightSource()
    )
    return ax

def drawCuboid(ax, I):
    # get centers and sizes from interval list
    center = [(a+b)/2 for (a,b) in I]
    size = [b - a for (a,b) in I]

    # do some magic
    o = [a - b / 2 for a, b in zip(center, size)]
    l, w, h = size
    x = np.array([[o[0], o[0] + l, o[0] + l, o[0], o[0]],
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]])
    y = np.array([[o[1], o[1], o[1] + w, o[1] + w, o[1]],
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  
         [o[1], o[1], o[1], o[1], o[1]],          
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]])
    z = np.array([[o[2], o[2], o[2], o[2], o[2]],                        
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],    
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],                
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]])

    # plot cuboid
    ax.plot_surface(
        x, y, z,
        cmap=cm.binary_r, alpha=0.2,
        lightsource = LightSource()
    )
    return ax

def testIfInInterval(X, y, b, Q, L, b0):
    (n, m) = X.shape
    L = np.diagflat(np.sqrt(L))
    vkr = np.linalg.norm(y - X @ b)**2
    Falpha = F.ppf(1-alpha, m, n-m)
    v1 = np.linalg.norm((L @ Q.T) @ (b - b0))**2
    v2 = m * vkr * Falpha / (n - m)
    return (v1, v2)
        

if __name__ == '__main__':
    alpha = 0.05
    path = 'podatki_6.txt'
    names, X, y = getData(path)
    
    b = linearRegresh(X, y, False)
    Q, L, r = getConfidenceElipsoidData(X,y,b)
    A = r * Q @ np.diagflat(np.power(L, -1/2))
    I = getBonferiCorrection(X, y, b, alpha)

    ax = plt.figure().gca(projection='3d')
    ax = drawElipsoid(ax, A, b, 40)
    ax = drawCuboid(ax, I)
    plt.show()


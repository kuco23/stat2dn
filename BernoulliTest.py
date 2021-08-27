import numpy as np
from math import ceil, floor
import matplotlib.pyplot as plt

def FBin(n,p0,k):
    b = 1
    s = 0
    q = pow(1-p0,n)
    p = 1
    for i in range(k+1):
        t = b * p * q
        s += t
        b *= (n-i) / (i+1)
        p *= p0
        q /= (1-p0)
    return (s,t)

def dpFBin(n,p0,k):
    b = 1
    s = 0
    q = pow(1-p0,n)
    p = 1
    for i in range(k+1):
        t = b * (i * p / p0 * q - (n-i) * p * q / (1-p0))
        s += t
        b *= (n-i) / (i+1)
        p *= p0
        q /= (1-p0)
    return (s,t)

def getRandomizedTest(n,alpha,p0):
    t = n * p0
    A = np.zeros((2,2))
    b = np.zeros((2,1))
    for C1 in range(0,ceil(t)+1):
        (FBinC1, A[0,0]) = FBin(n,p0,C1)
        (dpFBinC1, A[1,0]) = dpFBin(n,p0,C1)
        for C2 in range(max(C1+1,floor(t)),n+1):
            (FBinC2, A[0,1]) = FBin(n,p0,C2)
            (dpFBinC2, A[1,1]) = dpFBin(n,p0,C2)
            b[0] = alpha - (1 - (FBinC2 - (FBinC1 - A[0,0])))
            b[1] = dpFBinC2 - (dpFBinC1 - A[1,0])
            gamma = np.linalg.solve(A,b)
            if 0 <= gamma[0] <= 1 and 0 <= gamma[1] <= 1:
                return (C1,C2), tuple(map(float, gamma))

def drawC(n,k,alpha):
    h = 1 / (k+2)
    p0 = linspace(h,1-h,k)
    C = np.zeros((k,2))
    for i in range(k):
        p = p0[i]
        (C1,C2), (g1, g2) = getRandomizedTest(n,alpha,p)
        C[i,0] = C1
        C[i,1] = C2
    plt.plot(p0, C[:,1], 'r', po, C[:,1], 'b')
    plt.show()
                
if __name__ == '__main__':
    n, k = 18, 100
    alpha = 0.05
    

    

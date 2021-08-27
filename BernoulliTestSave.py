from math import ceil, floor
import numpy as np
import matplotlib.pyplot as plt

def binomialValuesGen(n,p0,k):
    b = 1
    s1, s2 = 0, 0
    q = pow(1-p0,n)
    p = 1
    for i in range(n+1):
        t1 = b * p * q
        t2 = b * (i * p / p0 * q - (n-i) * p * q / (1-p0))
        s1 += t1
        s2 += t2
        b *= (n-i) / (i+1)
        p *= p0
        q /= (1-p0)
        if i >= k: yield (s1,s2), (t1,t2)

def getRandomizedTest(n,alpha,p0):
    _s = floor(n * p0)
    A = np.zeros((2,2))
    b = np.zeros((2,1))
    gen1 = binomialValuesGen(n,p0,0)
    for C1 in range(0,n+1):
        s = max(_s, C1 + 1)
        gen2 = binomialValuesGen(n,p0,s)
        (Fbin1, dpFbin1), (A[0,0], A[1,0]) = next(gen1)
        for C2 in range(s,n+1):
            (Fbin2, dpFbin2), (A[0,1], A[1,1]) = next(gen2)
            b[0] = alpha - (1 - (Fbin2 - (Fbin1 - A[0,0])))
            b[1] = dpFbin2 - (dpFbin1 - A[1,0])
            gamma = np.linalg.solve(A,b)
            if 0 <= gamma[0] <= 1 and 0 <= gamma[1] <= 1:
                return (C1,C2), tuple(map(float, gamma))

def getCData(n,k,alpha):
    h = 1 / (k+2)
    p0 = np.linspace(h,1-h,k)
    C = np.zeros((k,3))
    for i in range(k):
        p = p0[i]
        (C1,C2), (g1, g2) = getRandomizedTest(n,alpha,p)
        C[i,0] = p
        C[i,1] = C1
        C[i,2] = C2
    return C

def inversionInterval(n,k,alpha):
    h = 1 / (k+2)
    p0 = np.linspace(h, 1-h, k)
    data = np.zeros((n+1,3))
    for t in range(n+1):
        data[t][0] = t
        s = 0
        for p in p0:
            (C1,C2), (g1,g2) = getRandomizedTest(n,alpha,p)
            if C1 <= t <= C2:
                if s == 0:
                    s = p if s != h else 0
                    data[t][1] = s
            elif s != 0:
                data[t][2] = p
                break
        else: data[t][2] = 1
    return data

def buildInversionIntervalTable(n,k,alpha,rnd):
    for d in data:
        s = '[' if d[1] != 0 else '('
        t = ']' if d[2] != 1 else ')'
        print(f'{d[0]} & {s}{d[1]}, {d[2]}{t} \\\\')

def getCoverData(n,k,alpha):
    h = 1 / (k+2)
    p0 = np.linspace(h,1-h,k)
    data = np.zeros((k,2))
    I = inversionInterval(n,k,alpha)
    for i,p in enumerate(p0):
        s = 0
        gen = binomialValuesGen(n,p,0)
        for t in range(n+1):
            prob = next(gen)[1][0]
            if I[t][1] <= p <= I[t][2]:
                s += prob
        data[i][0] = p
        data[i][1] = s
    return data

def getAx(data, title, xlab, ylab):
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_alpha(0.5)
    ax.spines['bottom'].set_alpha(0.5)
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    return ax
                
if __name__ == '__main__':
    n, k = 18, 1000
    alpha = 0.05
    #data = getCoverData(n,k,alpha)
    
            

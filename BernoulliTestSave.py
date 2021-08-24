from math import ceil, floor
import numpy as np

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
                
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    n, k = 100, 100
    alpha = 0.05
    p0 = np.linspace(0.01,1-0.01, k)

    C = np.zeros((k,2))
    for i in range(k):
        p = p0[i]
        (C1,C2), (g1, g2) = getRandomizedTest(n,alpha,p)
        print(i,C1,C2)
        C[i,0] = C1
        C[i,1] = C2

    plt.plot(p0, C[:,1], 'r', p0, C[:,1], 'b')
    plt.show()
            
            

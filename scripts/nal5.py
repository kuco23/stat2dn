import math
from operator import mul
from functools import reduce
from numpy.random import choice
from scipy.stats import multinomial, chi2

prod = lambda k: reduce(mul,k)
lambd = lambda n,t: prod([
    pow(n * pi[i] / t[i], t[i]) if t[i] != 0 else 1
    for i in range(len(t))
])
tau = lambda n,t: sum([
    (t[i] - n * pi[i])**2 / (n * pi[i]) for i in range(len(t))
])

def testPower(n, condition):
    s = 0
    for t0 in range(n+1):
        for t1 in range(n+1-t0):
            for t2 in range(n+1-t0-t1):
                t3 = n - t0 - t1 - t2
                if condition([t0,t1,t2,t3]):
                    s += multinomial.pmf([t0,t1,t2,t3],n,pi)
    return s

def testValues(n, t):
    a = n
    b = -(t[0] + t[3] - t[1]) / 3
    c = -t[2] / 9
    D = math.sqrt(b**2 - 4*a*c)

    p = [0,0,0,0]
    p[2] = (-b + D) / (2 * a)
    p[3] = (t[3]*(p[2] + 1/3)) / (t[0] + t[3])
    p[1] = 2/3 - 2*p[2]
    p[0] = 1 - p[1] - p[2] - p[3]
    p = [round(pi,10) for pi in p]
    if all([0 <= pi <= 1 for pi in p]):
        r = prod([(t[i]/n)**t[i] if t[i] != 0 else 1 for i in range(4)])
        q = prod([p[i]**t[i] if t[i] != 0 else 1 for i in range(4)])
        return q / r

def initTest(N, n):
    z = 0
    vals = [0,1,2,3]
    probs = [1/3, 1/3, 1/6, 1/6]
    for N in range(N):
        T = {0: 0, 1: 0, 2: 0, 3: 0}
        values = choice(vals, n, p=probs)
        for v in values: T[v] += 1
        s = testValues(n, list(T.values()))
        if -2 * math.log(s) > chi: z += 1
    print(z)
            

if __name__ == '__main__':
    #initTest(N,n)

    alpha = 0.05
    pi = [3/32, 7/32, 9/32, 13/32]
    chi = chi2.ppf(1-alpha, 3)

    '''
    for n in [35, 65, 95, 125]:
        condition = lambda t: -2*math.log(lambd(n, t)) > chi
        s = testPower(n, condition)
        print(f'${n}$ & ${s}$ \\\\')
    '''

    initTest(9500, 45)

    
    
    
    

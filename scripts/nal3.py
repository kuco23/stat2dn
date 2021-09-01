from math import log, sqrt
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from defaultAx import getAx

mu = lambda theta: -theta / ((1 - theta) * log(1 - theta))
sigma = lambda theta: sqrt(
    -theta / ((1 - theta)**2 * log(1 - theta)) - mu(theta)**2
)
trans = lambda c, theta, n: (c / n - mu(theta)) / (sigma(theta) / sqrt(n))

def getData(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            data.append(float(line.strip()))
    return data

def getTest(alpha, theta0, n):
    C = 1
    while True:
        val1 = norm.cdf(trans(C, theta0, n))
        val2 = norm.cdf(trans(C-1, theta0, n))
        gamma = (alpha + val1 - 1) / (val1 - val2)
        if 0 <= gamma < 1: break
        C += 1
    return (C, gamma)

def getConfidenceInterval(t, alpha, n, k=1000):
    a,b = [0, 0]
    theta = np.linspace(1/(k+2), 1-1/(k+2), k)
    for theta0 in theta:
        Ft = norm.cdf(trans(t, theta0, n))
        Ftm = norm.cdf(trans(t-1, theta0, n))
        if Ft <= alpha/2 and b == 0:
            b = theta0
        if Ftm <= 1 - alpha/2 and a == 0:
            a = theta0
            break
    return [a,b]

def plotPowerFunction(n, C, gamma, k=100):
    b = []
    theta = np.linspace(1/(k+2), 1-1/(k+2), k)
    for theta0 in theta:
        val1 = norm.cdf(trans(C, theta0, n))
        val2 = norm.cdf(trans(C-1, theta0, n))
        b.append(1 - val1 + gamma * (val1 - val2))
        
    title, xlab, ylab = '', 'theta', 'beta(theta)'
    ax = getAx(title, xlab, ylab)
    ax.plot(t, b)
    plt.show()

def plotThetaDiff(k=100):
    theta = np.linspace(1/(k+2),1-1/(k+2),k)
    r = np.log(theta)*np.log(1-theta)*(1-theta)

    ax = getAx('', 'theta', 'log(theta)*log(1-theta)*(1-theta)')
    ax.plot(theta,r)
    plt.show()
        

if __name__ == '__main__':
    n = 50
    theta0 = 0.7
    alpha = 0.05

    #C, gamma = getTest(alpha, theta0, n)
    #plotPowerFunction(n, C, gamma)

    path = 'podatki_3.txt'
    data = getData(path)
    t = sum(data)
    I = getConfidenceInterval(t, alpha, len(data))

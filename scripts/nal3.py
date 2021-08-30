from math import log, sqrt
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

n = 50
theta0 = 0.7
alpha = 0.05

mu = lambda theta: -theta / ((1 - theta) * log(1 - theta))
sigma = lambda theta: sqrt(
    -theta / ((1 - theta)**2 * log(1 - theta)) - mu(theta)**2
)
trans = lambda c, theta: (c / n - mu(theta)) / (sigma(theta) / sqrt(n))

C = 1
while True:
    val1 = norm.cdf(trans(C, theta0))
    val2 = norm.cdf(trans(C-1, theta0))
    gamma = (alpha + val1 - 1) / (val1 - val2)
    if 0 <= gamma < 1:
        print(C, gamma)
        break
    C += 1

print(f'C={C}')
print(f'gamma={gamma}')

b = []
t = np.linspace(0.001, 0.999, 100)
for theta in t:
    val1 = norm.cdf(trans(C, theta))
    val2 = norm.cdf(trans(C-1, theta))
    b.append(1 - val1 + gamma * (val1 - val2))

title, xlab, ylab = '', 'theta', 'beta(theta)'
fig, ax = plt.subplots(1, 1, figsize=(15, 8))
ax.set_title(title)
ax.set_xlabel(xlab)
ax.set_ylabel(ylab)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_alpha(0.5)
ax.spines['bottom'].set_alpha(0.5)
ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    
ax.plot(t, b)
plt.show()

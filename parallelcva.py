import cva
import numpy as np
from matplotlib import pyplot as plt
import random
import time
from scipy import integrate

paths = 100000
steps = 1000

swap_term = 0.5
payments = 20
notional = 1


a = 0.01
b = 0.05
sigma = 0.001

r0 = 0.05

start = time.time()
m = cva.cva()
exposures = m.main(paths,steps, swap_term, payments, notional, a, b, sigma, r0)
x = np.arange(0,10,0.1)
print(f"Time takenL: {time.time() - start}")

# print(integrate.trapz(x,exposures))
plt.plot(exposures[1].get())
plt.show()
# for i in range(10):
#     plt.plot(exposures[i],lw=0.8, alpha=0.8)
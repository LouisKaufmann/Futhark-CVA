import cva
import numpy as np
from matplotlib import pyplot as plt
import random
import time
from scipy import integrate

paths = 100000
steps = 100

swap_term = np.array([1,0.5], dtype=np.float32)
payments =  np.array([10,20], dtype=np.int64)
notional =  np.array([1,1], dtype=np.float32)
netting =  np.array([1,-1], dtype=np.int64)

a = 0.01
b = 0.05
sigma = 0.001

r0 = 0.05

start = time.time()
m = cva.cva()
exposures = m.main(paths,steps,netting, swap_term, payments, notional, a, b, sigma, r0)

print(f"Time takenL: {time.time() - start}")
print(exposures[0])

plt.plot(exposures[1].get())
plt.show()

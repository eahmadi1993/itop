from mpc.sim import NonlinearSystem
import numpy as np
import matplotlib.pyplot as plt

sys_nl = NonlinearSystem(0.1, 1,1)
XX = []
YY = []
x = np.ones((4,1))
u = np.zeros((2,1))
for i in range(1000):
    x = sys_nl.update_nls_states(x,u)
    XX.append(x[0])
    YY.append(x[1])

plt.plot(XX,YY)
plt.show()
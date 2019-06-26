import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf


x = np.linspace(0, 2, 100)
y = np.sqrt(x) * np.sin(2 * np.pi * x) + np.random.random_sample(100)

mod = smf.quantreg('y ~ bs(x, df=12)', dict(x=x, y=y))
res = mod.fit(q=0.999)
print(res.summary())

plt.plot(x, y, '.')
plt.plot(x, res.predict(), 'r')
plt.show()
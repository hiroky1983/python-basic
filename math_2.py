# べき乗
import numpy as np # NumPyをインポート
import matplotlib.pyplot as plt # matplotlib.pyplotをインポート

x = np.linspace(-2, 2)

y_2 = 2**x
y_3 = 3**x

plt.plot(x, y_2, label="2^x")
plt.plot(x, y_3, label="3^x")
plt.legend()

plt.xlabel("x", size=14)
plt.ylabel("y", size=14)
plt.grid()
plt.show()
# ネイピア数
print(np.e)
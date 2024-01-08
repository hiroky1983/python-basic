# 関数の描画
## NumPyの変換
import numpy as np # NumPyをインポート
import matplotlib.pyplot as plt # matplotlib.pyplotをインポート

a = [0, 1, 2, 3, 4, 5] # リストaを定義
b = np.array(a) # NumPy配列に変換
print(b) # bを表示

#linespace
x = np.linspace(-5,5) # -5から5までの範囲を50分割したNumPy配列を生成

print(x)
print(len(x))

##１次関数
######数式の練習
# $$y=2x+1$$

x = np.linspace(-5,5)
y = 2*x + 1

plt.plot(x,y)
plt.show()

## グラフの装飾

x = np.linspace(-3,3)
y_1 = 1.5*x
y_2 = -2*x+1

plt.xlabel("x value", size=14)
plt.ylabel("y value", size=14)

plt.title("my graph")
plt.grid()

plt.plot(x,y_1, label="y_1")
plt.plot(x,y_2, label="y_2", linestyle="dashed")

plt.legend()
plt.show()

##二次関数、三次関数
x = np.linspace(-4,4)
y_1 = 2*x +1
y_2 = x**2 -4
y_3 = 0.5*x**3 -6*x

plt.plot(x, y_1, label="1st")
plt.plot(x, y_2, label="2nd")
plt.plot(x, y_3, label="3rd")
plt.legend()

plt.xlabel("x", size=14)
plt.ylabel("y", size=14)
plt.grid()
plt.show()

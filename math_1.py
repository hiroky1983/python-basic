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

##二次関数、三次関数

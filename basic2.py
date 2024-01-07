#配列 リスト
a = [1,2,3,4,5]
print(a)

b = 3000
c = [b,3002, 2.1, "hello world"]
# 範囲指定して要素を取り出す
print(c[0:2])

# タプル
# タプルは要素の変更ができない

d = (1,2,3,4,5)
print(d)
print(d[0])

# 辞書
# キーと値をセットで保存する
e = {"apple":100, "orange":200, "banana":300}
print(e)
print(e["apple"])
# 値の変更
e["apple"] = 500
print(e["apple"])
# 値の追加
e["grape"] = 400
print(e)

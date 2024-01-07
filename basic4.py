# 関数
def say_hello():
    print('Hello World')
  
say_hello()

## 関数の引数
def say_hello2(name):
    print('Hello ' + name)

say_hello2('Bob')

## 関数の戻り値
def say_hello3(name):
    return 'Hello ' + name

print(say_hello3('jenkins'))

# クラス
class User:
    def __init__(self, name):
        self.name = name

    def say_hello(self):
        print('Hello ' + self.name)

bob = User('smith')
bob.say_hello()
# ファイルの保存と読み込み
greet = 'Good Morning, Good Afternoon, Good Evening'
with open('greet.txt', 'w') as f:
    f.write(greet)

# ファイルの読み込み
with open('greet.txt', 'r') as f:
    print(f.read())

    
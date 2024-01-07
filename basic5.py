# __init__メソッドは、クラスのインスタンスを作成するときに呼び出される特殊なメソッドです。
class User:
    def __init__(self, name):
        self.name = name

    def say_hello(self):
        print('Hello ' + self.name)

bob = User('smith')
bob.say_hello()
# __call__メソッドは、インスタンスを関数のように呼び出せるようにする特殊なメソッドです。
class User:
    def __init__(self, name):
        self.name = name

    def say_hello(self):
        print('Hello ' + self.name)

    def __call__(self):
        print('call ' + self.name)

bob = User('smith')
bob()

# README.mdを作成する
text = '# python-basic \n pythonの基本文法を学ぶ'
with open('README.md', 'w') as f:
    f.write(text)

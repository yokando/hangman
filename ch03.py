#
# Chapter4
'''
関数
'''
# 自前で定義"
def f(x):
    return x+1
print(f(4))

def f():
    return 1+1
print(f())

def f(x,y,z):
    return x+y+z
print(f(1,2,3))
print("")

def even_odd():
    n=input("type a number:")
    n=int(n)
    if n % 2 == 0:
        print("n isi even.")
    else:
        print("n is odd.")
even_odd()
even_odd()
even_odd()          
print("")

def f(x=2):
    return x**x
# デフォルト値
print(f())
# 引数の値
print(f(4))
print("")

# グローバル変数
x=100
def f():
    global x
    x += 1
    print(x)
f()
print("")
print("以上は自作の関数でした。")
print("以下は組み込み関数です。")
print("")


# 組み込み関数

# 文字列の長さ
print(len("Monty"))
print("")

#
# 整数を文字列に変換
print(str(100))
print("")

# 整数化
print("100")
print(int(1.23))
print("")

# 浮動小数点数化
print(float(100))
print("")

# 入力をうながす
age=input("Enter your age:")
iage=int(age)
if iage < 21:
      print("You are yong!")
else:
      print("You are old!")
print("")

# ミス入力対策（例外処理）
try:
    a=input("type a number: ")
    b=input("type another: ")
    a=int(a)
    b=int(b)
    print(a/b)
except(ZeroDivisionError, ValueError):
    print("Invalid input.")



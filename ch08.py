#
# Chapter8
#
# 既存モジュールの実行

#　乗数関数
import math
x=3
y=2
print('')
print("乗数：",x,y,math.pow(x,y))

# ランダム生成関数
import random
print('')
print("randum:",random.randint(0,100))

# 統計関数
import statistics
nums=[1,5,33,12,46,33,2]
print('')
print("mean:",statistics.mean(nums))
print("median:",statistics.median(nums))
print("mode:",statistics.mode(nums))

#
# 自作モジュールの実行
import ch08s1
print('')
print(ch08s1.print_hello())

print("")
import ch08s2

print("")
import ch08s3





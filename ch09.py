#
# Chapter9
'''
ファイル操作
'''
import os
print(os.getcwd())

# 書き込み
st = open("ch09_out.txt","w")
st.write("Hi from Python!")
st.close()

# 日本語書き込み
st = open("ch09_out.txt","w",encoding="utf-8")
st.write("Pyhonnからこんにちは！")
st.close()

# 読み込み
my_list=[]
with open("ch09_out.txt","r",encoding="utf-8") as st:
    my_list.append(st.read())
print(my_list)

# CSVファイル書き込み
import csv
with open("ch09_out.csv","w",newline='') as st:
    w=csv.writer(st,delimiter=",")
    w.writerow([1,2,3])
    w.writerow([4,5,6])

# CSVファイル読み込み
with open("ch09_out.csv","r") as st:
    r=csv.reader(st,delimiter=",")
    for row in r:
         print(",".join(row))




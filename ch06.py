#
# Chapter 6

"""文字列の操作
   line two
   line thre
"""
author="Kafka"
print(author[0],author[1],author[2],author[-2],author[-1])

print("cat"+"in"+"hat")
print("sawyer"*2)

print("Hello".upper())
print("Hello".lower())
print("hello".capitalize())
print("Hello{}{}".format(" William"," Faulkner!"))

print("Hello William Faulkner!".split(" "))
words=["Hello","William","Faulkner","!"]
print(" ".join(words))
print("   hello   ".strip())
print("Hello".replace("l","L"))
print("Hello".index("o"))
print("Cat" in "Cat in the hat.")
print("Bat" in "Cat in the hat.")
print("Bat" not in "Cat in the hat.")

print("Cat \"in the\" hat.")
print("Cat 'in the' hat.")
print("Cat\nin the\nhat.")

fict=["トルストイ","カミユ","オーウェル","ハクスリー","オースティン"]
print(fict[0:3])
ivan="死の代わりにひとつの光があった。"
print(ivan[6:])
      
      
      
      
      


      

      

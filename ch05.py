#
# Chapter 5
#
print("Hello".upper())
print("Hello".replace("o","@"))
#
# List
fruit=list()
print(fruit)
fruit=["Apple","Orange","Pear"]
print(fruit)
fruit.append("Banana")
fruit.append("Peach")
print(fruit)
print(fruit[0])
print(fruit[2])
print(fruit[4])
fruit[2]="Strawberry"
print(fruit)
item=fruit.pop()
print(item)
print(fruit)
#
fruit2=["Peach"]
print(fruit+fruit2)
print("Pear" in fruit)
print("Pear" not in fruit)
print(len(fruit))
#
guess=input("Which fruit do yo eat? :")
if guess in fruit:
    print("OK!")
else:
    print("NO!")
#
lists=[]
color=["red","yellow","white"]
number=[0,1,2,3,4]
lists.append(fruit)
lists.append(color)
lists.append(number)
print(lists)
l2=lists[2]
print(l2)
number.append(5)
print(number)
print(lists)
#
# Tupple
tp1=("only_one",)
print(tp1)
rndm=("Jackson",1958,True)
print(rndm)
print(rndm[1])
print(1958 in rndm)
print("Jackson" not in rndm)
#
# Dictionary
books={"Dracula":"Stoker",
       "1984":"Orwel",
       "The Trial":"Kafka"}
del books["The Trial"]
print(books)
books["The Trial"]="Kafka"
print(books)






    
    
        






#
# Chapter 7
'''
ループ
'''
name="Ted"
for character in name:
    print (character)

shows=("GOT","Narcos","Vice")
for show in shows:
    print(show)

tv=["GOT","Narcos","Vice"]
i=0
for show in tv:
    new=tv[i]
    new=new.upper()
    tv[i]=new
    i += 1
print(tv)

tv=["GOT","Narcos","Vice"]
for i, new in enumerate(tv):
    new=tv[i]
    new=new.lower()
    tv[i]=new
print(tv)
print("")

for i in range(1,5):
    print(i)
print("")

x=3
while x > 0:
    print('{}'.format(x))
    x -= 1
print("Happy New Year!")
print("")

for i in range(1,5):
    print(i)
    break
print("")

for i in range(1,5):
    if i== 3:
        continue
    print(i)






    

    
 

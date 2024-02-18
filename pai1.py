#
# calculete pai by Leibniz eq.

pai = 4.0*(1.0)
i = 1
j = 0
while i < 1001:
    j = i % 2
    k = i % 100
    x = i
#    print(i,j)
    if j == 0 :
         pai = pai + 4.0/(2.0*x+1.0)
    else :
         pai = pai - 4.0/(2.0*x +1.0)
    if k == 0 :
         print("i pai=", i, pai)
    i = i+1
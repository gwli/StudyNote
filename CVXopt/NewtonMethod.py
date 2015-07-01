# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# http://baike.baidu.com/view/643093.htm#7

# <codecell>

def fd(x):
    return 3*((x-3)**2)

# <codecell>

def newtonMethod(n,assum):
    time = n
    x = assum
    Next =0
    A= f(x)
    B =fd(x)
    print('A = ' + str(A) + ',B = ' + str(B) + ',time = ' + str(time))
    if f(x)==0.0:
        return time,x
    else:
        Next = x-A/B
        print('Next x = '+ str(Next))
        if A == f(Next):
            print ('Meet f(x) = 0,x = ' + str(Next)) 
            else:
                return newtonMethod(n+1,Next)
            newtonMethod(0,4.0)

# <codecell>



import numpy as np
import random
'''
zip(): 将多个数组或元组压缩成一个元组
    zip后得到一个zip类型的对象，可以直接解压。一旦调用了解压或者做list或tuple做转换，zip对象变为空。这应该是内存方面的优化回收。
zip(*): 将一个数组或元组解压成多个数组或元组
'''
def zip_test():
    a=[1,2,3]
    b=[4,5,6]
    c=[7,8,9]
    z=zip(a,b,c)
#     zip(*z)
    z2t=tuple(z)
    z2l=list(z)
    print("zip:{}, type:{}".format(z,type(z)))
    print("zip to list:{}".format(z2l))
    print("zip to tuple:{}".format(z2t))
    print("unzip")
    for x in zip(*z2t):
        print(" {}".format(x))
    
    a1=np.array(a)
    b1=np.array(b)
    c1=np.concatenate((a1,b1))
    print(c1)
    
    a=[[1,2],[3,4]]
    b=[[5,6],[7,8]]
    z=zip(a,b)
    print(list(z))

def tuple_add():
    a=(1,2)
    b=(3,)
    c=a+b 
    print(c)
    print(type(c))

def dict():
    a={'1':'a','3':'b','2':'c'}
    print(a)
    a=sorted(a.items())
    print(type(a))
    a=[2,3,1]
    print(a)
    print(sorted(a))

def find():
    a='hello.jpg.png'
    index1=a.find('.')
    index2=a.rfind('.')
    index3=a.index('.')
    print(index1,index2,index3)
    print(a[:index1],a[:index2],a[:index3])

#sort、reverse、shuffle
def reverse():
    a=[10,2,5,3]
    a.sort()
    b=a[::-1]
    random.shuffle(b)
    print(b)

reverse()

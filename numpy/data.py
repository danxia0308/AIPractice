import numpy as np

'''
默认dtype为np.float64
使用astype来改变其dtype。
从文件中读取时需要指明正确的dtype，否则读取数据有问题。
np.float=np.float64, np.int=np.int64
'''
def dtype_basic():
    a=np.random.random(4)
    print(a)
    print('DefaultDtype={}'.format(a.dtype))
    #直接改变dtype，会按照指定的dtype大小来读取数据，造成数据不一致。
    a.dtype=np.float32
    print('Change to float32, then size*2:\n{}'.format(a))
    a.dtype=np.float16
    print('Change to float16, then size*4:\n{}'.format(a))
    a.dtype=np.float
    print('Change to float, then size not changed:\n{}'.format(a))
    #正确的转换dtype的方法是使用astype。
    print(a.astype(np.float32))
dtype_basic()
'''
python字典存为npy文件后，load出来还是numpy.ndarray类型的，需要调用item()来获得字典结构。
'''
def npy():
    sub_d={'1':'one','2':'two'}
    d={}
    d['0']=sub_d
    np.save('dict.npy',d)
    x=np.load('dict.npy')
    print(type(x),type(x.item()))
    print(x.item()['0'])
    
    
    
def array():
    a=[]
    a.append(['keys1','sophie'])
    a.append(['key2','kate'])
    print(a)


    

# d={}
# dd1={}
# dd2={}
# d['key1']=dd1
# dd1['name']='sophie'
# d['key2']=dd2
# dd2['name']='kate'
# print(d)
# 
# b=[['key1','sophie'],['key2','kate']]
# b_arr=np.array(b)
# print(type(b_arr))
# a=np.zeros((2,2))
# a[0][0]='key1'
# a[0][1]='sophie'
# a[1][0]='key2'
# a[1][1]='kate'
# print(a)

# student = np.dtype([('name','S20'), ('age', 'i1'), ('marks', 'f4')]) 
# print(type(student))


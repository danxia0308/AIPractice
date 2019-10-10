'''
np.flip
size/ndim/shape/dtype
'''
import numpy as np
import os
import imageio
import matplotlib.pyplot as plt

'''
logical_and
flip

'''


def sum():
    print(np.sum((1,2)))
    print(np.mean([1,2,3,0]))
sum()

def logical():
    a=np.array([1,2,5,6])
    b=np.array([1,2,5,4])
    c=np.logical_and(a,b)
    print(c)
    print(np.sum(c))
    c=a-b
    c=np.where(c==0,1,0)
    print(np.sum(c))

def flip(img):
    imgud = np.flip(img, 0) #flipud
    imglr = np.flip(img, 1) #fliplr
    return imgud, imglr

def cal_size():
    arr = np.zeros([2,2])
    print('size={}'.format(arr.size))
    print('dtype={}'.format(arr.dtype))
    print('ndim={}'.format(arr.ndim))
    print('shape={}'.format(arr.shape))

def add():
    a=np.array([[1,1],[2,2]])
    b=np.array([[3,3],[4,4]])
    print(np.concatenate([a,b]))
    print(np.concatenate([[0],[1,2]]))

def dim():
    a=np.ones(shape=(2,2))
    print(a)
    print(a.shape)
    b=np.mean(a,axis=1)
    print(b)
    print(b.shape)

def where():
    a=np.array([[1,2,0,5],
                [2,0,3,2],
                [2,0,3,2]])
    b=np.array([[[1,2,0]]])
    y,x=np.where(a)
    z=np.where(y==1)
    print(x)
    print(y)
    print(z)
    z1=np.abs(np.array(z)-4)
    
    print(np.argmin(z1))

def unique():
    a=np.array([1,3,5,6,3,5,8])
    print(np.unique(a))

def slice():
    a=np.array([1,3,5,6,3,5,8])
    print(a[1:2])
    print(a[2:0])
    print(np.abs(5-6))


def sort():
    a=np.array([1,3,5,6,3,5,8])
    b=np.sort(a)
    print(b)

def shuffle():
    a=[1,3,5]
    b=np.random.shuffle(a)
    print(a)
# shuffle()

def numpyarray_pythonlist_convert():
    a_list=[[1,2],[3,4]]
    # Convert from numpy array to python list
    a_arr=np.asarray(a_list)
    b_arr=np.array(a_list)
    #Convert from python list to numpy array
    b_list=b_arr.tolist()
    a_list[0][0]=5
    print('a_arr',a_arr)
    print('b_arr',b_arr)
    print('b_list',b_list)

'''
vstack = concatenate axis=0
hstack = concatenate axis=1
'''
def stack_concatenate():
    a=np.array([[1,2],[3,4]])
    b=np.vstack((a,a))
    c=np.hstack((a,a))
    d=np.concatenate((a,a), axis=0)
    
    print(a)
    print(b)
    print(c)
    print(d)
    # add in axis=2
    a=np.array([[1,2],[3,4]])
    b=np.array([a,a])
    print(b)
    c=np.transpose(b, [1,2,0])
    print(c)

# a=np.array([[1,2],[3,4]])
# c=np.array([[1,3],[3,4]])
# b=np.where(a>1,c,1)
# print(b)

def test():
    dir_path='/Users/chendanxia/sophie/segmentation_img_set/human_sum/all/images/'
    for name in os.listdir(dir_path):
        path=dir_path+name
        img = imageio.imread(path)
        
        plt.subplot(221)
        plt.imshow(img)
        imgup,imglr=flip(img)
        plt.subplot(223)
        plt.imshow(imgup)
        plt.subplot(224)
        plt.imshow(imglr)
        plt.show()
        break;
# test()

# c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
# print(np.random.random((1, 3)))
# print(np.random.random((1, 3))*0.6+0.4)
# print((np.random.random((1, 3))*0.6+0.4).tolist())
# print(c)

# path='/Users/chendanxia/sophie/segmentation_img_set/coco2014/train2014/COCO_train2014_{:0>12d}.jpg'.format(262145)
# import os
# print(os.path.basename(path))
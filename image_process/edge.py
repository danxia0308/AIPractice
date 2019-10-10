import cv2 as cv
import numpy as np
import random
random.seed(12345)

def CannyThreshold(val):
    ratio=2
    kernel_size=3
    low_threshold=val
    hight_threshold=low_threshold*ratio
    edge=cv.Canny(img_blur,low_threshold,hight_threshold,kernel_size)
    mask=edge !=0
    dst=src_img *(mask[:,:,None].astype(src_img.dtype))
    cv.imshow("Edge Map", dst)

def ContourThreshold(val):
    ratio=2
    kernel_size=3
    low_threshold=val
    hight_threshold=low_threshold*ratio
    canny_output=cv.Canny(img_blur,low_threshold,hight_threshold,kernel_size)
    contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    drawing=np.zeros((canny_output.shape[0],canny_output.shape[1],3),dtype=np.uint8)
    for i in range(len(contours)):
        color = (random.randint(0,256),random.randint(0,256),random.randint(0,256))
        cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
    cv.imshow('Contours', drawing)

def get_mask_edge(mask):
    edge = np.where(mask==0,1,0)
    h, w = edge.shape
    edge_center = edge[1:h-1,1:w-1]
    edge_left = edge[1:h-1,1-1:w-1-1]
    edge_right = edge[1:h-1,1+1:w-1+1]
    edge_up = edge[1-1:h-1-1,1:w-1]
    edge_down = edge[1+1:h-1+1,1:w-1]
    edge_center = edge_center+edge_left+edge_right+edge_up+edge_down
    edge_center = np.where(edge_center == 5 , 0, edge_center)
    edge_center = np.where(edge_center > 0 , 1, 0)
    edge[1:h-1,1:w-1] = edge_center
    return edge

'''
执行顺序为先灰度，再blur，然后Canny求边界，最后findContours。
findContours把边界中连续的点构成一个contour，所有的contour放在contours中，同时在hierachy中返回其层级关系。
通过对比findContour和我求的get_mask_edge，两者基本重合。问题是一个contour中为什么有重复的点？
'''
def analyzeeContour():
    ratio=2
    kernel_size=3
    low_threshold=1
    hight_threshold=low_threshold*ratio
    img_path='/Users/chendanxia/sophie/segmentation_img_set/human_sum/all/masks/COCO_train2014_000000452909_seg.png'
    img_path='/Users/chendanxia/sophie/segmentation_img_set/human_sum/all/masks/COCO_val2014_000000106073_seg.png'
    src_img=cv.imread(img_path)
    src_gray=cv.cvtColor(src_img,cv.COLOR_BGR2GRAY)
    img_blur=cv.blur(src_gray,(3,3))
    canny_output=cv.Canny(img_blur,low_threshold,hight_threshold,kernel_size)
    #cv.CHAIN_APPROX_SIMPLE 压缩水平、垂直和对角线方向的像素。
    #cv.CHAIN_APPROX_NONE 存储所有轮廓点。
    contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
#     print(contours)
#     print(hierarchy)
    colors=[[0,0,255],
            [0,255,0],
            [255,0,0],
            [147,20,255 ]]
    contour_img = np.zeros(src_img.shape)
    
    mask=cv.imread(img_path, -1)
    edge=get_mask_edge(mask)
    print(np.sum(edge))
    for c in range(3):
        contour_img[:,:,c]=np.where(edge == 1, colors[2][c], contour_img[:,:,c])
    
    color_id=0
    for contour in contours:
        print(contour.shape)
        points_set=set()
        color=colors[color_id]
        color_id=color_id+1
        for point in contour:
            pixel=point[0]
            if (pixel[1],pixel[0]) in points_set:
                contour_img[pixel[1],pixel[0],:]=colors[3]
            else:
                contour_img[pixel[1],pixel[0],:]=color
            points_set.add((pixel[1],pixel[0]))
            
        print(len(points_set))
    
    
    cv.imshow('Mask', contour_img)
    cv.waitKey()
    drawing=np.zeros((canny_output.shape[0],canny_output.shape[1],3),dtype=np.uint8)
    for i in range(len(contours)):
        color = (random.randint(0,256),random.randint(0,256),random.randint(0,256))
        cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
    cv.imshow('Contours', drawing)
    cv.waitKey()
    
def test_canny():
    cv.namedWindow('Edge Map')
    cv.createTrackbar("Min Threshold", 'Edge Map',0,200,CannyThreshold)
    CannyThreshold(0)
    cv.waitKey()

def test_contour():
    cv.imshow('Source',src_img)
    max_threshold=255
    init_thresh=100
    cv.createTrackbar('Canny Thresh:', 'Source', init_thresh, max_threshold, ContourThreshold)
    ContourThreshold(init_thresh)
    cv.waitKey()

def test_equalize_hiist():
    dst = cv.equalizeHist(src_gray)
    cv.imshow('Source Image', src_gray)
    cv.imshow('Equalized Image', dst)
    cv.waitKey()

img_path='/Users/chendanxia/sophie/segmentation_img_set/human_sum/all/images/COCO_val2014_000000265550.jpg'
img_path='/Users/chendanxia/sophie/segmentation_img_set/human_sum/all/images/COCO_train2014_000000452909.jpg'
img_path='/Users/chendanxia/sophie/segmentation_img_set/human_sum/all/masks/COCO_train2014_000000452909_seg.png'
img_path='/Users/chendanxia/sophie/sophie.jpg'
img_path='/Users/chendanxia/sophie/sophie2.png'
src_img=cv.imread(img_path)
src_gray=cv.cvtColor(src_img,cv.COLOR_BGR2GRAY)
img_blur=cv.blur(src_gray,(3,3))
src_img=img_blur
test_contour()


    
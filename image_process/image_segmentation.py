import cv2
import numpy as np
import math
from enum import Enum
from image_process import hist
import random
random.seed(12345)

class Direction(Enum):
    X=0
    Y=1
    BOTH=2

colors=[[179,238,58], #olive light green
        [105,139,34], #olive dark
        [255,0,0],  #red
        [0,255,0], #green
        [0,0,255],  #blue
        ]

alpha=0.3

def get_nearst_points(y_ids, x_ids, x, y, direction=Direction.X):
    points=[]
    if direction == Direction.X or direction == Direction.BOTH:
        x_ids_selected = x_ids[np.where(y_ids == y)[0]]
        if len(x_ids_selected) != 0:
            x_ids = np.argsort(np.abs(np.array(x_ids_selected)-x))
            left_found=False;right_found=False
            for x_id in x_ids:
                x_selected = x_ids_selected[x_id]
                if not left_found and x_selected < x:
                    left_found=True
                    points.append((x_selected, y))
                if not right_found and x_selected:
                    right_found=True
                    points.append((x_selected, y))
                if left_found and right_found:
                    break
    if direction == Direction.Y or direction == Direction.BOTH:
        y_ids_selected = y_ids[np.where(x_ids == x)[0]]
        if len(y_ids_selected) != 0:
            y_id = np.argmin(np.abs(np.array(y_ids_selected)-y))
            y_selected = y_ids_selected[y_id]
            points.append((x, y_selected))
    return points

def select_points_by_hist(img, mask_x, mask_y, points, hsv):
    selected_points=[]
    bg_color= [25.1,28.9,34.2]
    for point in points:
        x_select, y_select=point
        #TODO 先使用颜色均值，以后使用直方图，看哪个效果更好。
        x_count=np.abs(x_select-mask_x)+1
        y_count=np.abs(y_select-mask_y)+1
        threshold=30
        if x_count > threshold: x_count=threshold
        if y_count > threshold: y_count=threshold
        if x_select >=mask_x:
            x_start=mask_x
            img_side_x_start=x_select
            mask_side_x_start=mask_x-(x_count-1)
        else:
            x_start=x_select
            img_side_x_start=x_select-(x_count-1)
            mask_side_x_start=mask_x
        if y_select >=mask_y:
            y_start=mask_y
            img_side_y_start=y_select
            mask_side_y_start=mask_y-(y_count-1)
        else:
            y_start=y_select
            img_side_y_start=y_select-(y_count-1)
            mask_side_y_start=mask_y
        
#         between_img_mask_mean=np.mean(img[y_start:y_start+y_count, x_start:x_start+x_count,0])
#         img_side_mean=np.mean(img[img_side_y_start:img_side_y_start+y_count,img_side_x_start:img_side_x_start+x_count,0])
#         mask_side_mean=np.mean(img[mask_side_y_start:mask_side_y_start+y_count,mask_side_x_start:mask_side_x_start+x_count,0])
        
        between_img_mask_hsv=hsv[y_start:y_start+y_count, x_start:x_start+x_count,:]
        img_side_hsv=hsv[img_side_y_start:img_side_y_start+y_count,img_side_x_start:img_side_x_start+x_count,:]
        mask_side_hsv=hsv[mask_side_y_start:mask_side_y_start+y_count,mask_side_x_start:mask_side_x_start+x_count,:]
        between_img_mask=img[y_start:y_start+y_count, x_start:x_start+x_count,:]
        img_side=img[img_side_y_start:img_side_y_start+y_count,img_side_x_start:img_side_x_start+x_count,:]
        mask_side=img[mask_side_y_start:mask_side_y_start+y_count,mask_side_x_start:mask_side_x_start+x_count,:]
        
        dist=hist.compare_hist(img_side_hsv, between_img_mask_hsv,compare_method=0) - hist.compare_hist(between_img_mask_hsv, mask_side_hsv,compare_method=0)
        if dist > 0:
            color_dist_between= hist.color_image_distance(between_img_mask, bg_color)
            color_dist_img=hist.color_image_distance(img_side, bg_color)
            if (color_dist_between < 30 and color_dist_img > 30):
                continue
            selected_points.append(point+(dist,))
    return selected_points

def get_nearst_edge(y_ids, x_ids, x, y):
    x_ids_selected = x_ids[np.where(y_ids == y)[0]]
    if len(x_ids_selected) == 0:
        return x, y
    x_id = np.argmin(np.abs(np.array(x_ids_selected)-x))
    x_selected = x_ids_selected[x_id]
    return x_selected, y 

def find_top_middle(y_ids, x_ids):
    y=y_ids[0]
    ids = np.where(y_ids==y)[0]
    xs = x_ids[ids]
    xs = np.sort(xs)
    x=xs[len(xs)//2]
    return x,y

def select_best_point(points):
    bigest_dist=0
    chosen_point=None
    for point in points:
        x, y, dist = point
        if dist > bigest_dist:
            chosen_point=(x,y)
            bigest_dist=dist
    return chosen_point

def adjust_edge(img_edge, mask_edge,img):
    dst_edge=np.zeros(mask_edge.shape)
    mask_y_ids, mask_x_ids = np.where(mask_edge > 0)
    y_ids, x_ids = np.where(img_edge > 0)
    x_middle, _ = find_top_middle(y_ids, x_ids)
    hsv=hist.get_hsv(img)
    for i, mask_y in enumerate(mask_y_ids):
        mask_x=mask_x_ids[i]
#         mask_y=mask_y_ids[i]
        if mask_y == 312:
            print(mask_y)
        points = get_nearst_points(y_ids, x_ids, mask_x, mask_y, Direction.X)
        points = select_points_by_hist(img, mask_x, mask_y, points, hsv)
        
        if len(points) == 0:
            dst_edge[mask_y,mask_x]=1
        else:
            point=select_best_point(points)
            if point == None:
                dst_edge[mask_y,mask_x]=1
                continue
            x, y=point
            if np.abs(x-mask_x) > 100:
                dst_edge[mask_y,mask_x]=1
            else:
                dst_edge[y,x]=1
#         x_adj, _=get_nearst_edge(y_ids, x_ids, mask_x, mask_y)
#         kernel=2
        
#         if math.fabs(x_adj-mask_x) > 10 and np.sum(img_edge[mask_y-kernel:mask_y+kernel, mask_x-kernel:mask_x+kernel]) == 0:
#             x_ids_selected = x_ids[np.where(y_ids == mask_y-1)]
#             arg=np.argmin(np.abs(np.array(x_ids_selected)-mask_x))
#             diff=math.fabs(x_ids_selected[arg]-mask_x)
#             print(diff, mask_x, mask_y)
#             x_adj=x_ids_selected[arg]
#             x_adj=mask_x
#         dst_edge[mask_y,x_adj]=1
    return dst_edge

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

def get_mask_from_edge(old_mask, edge):
    y_ids, x_ids = np.where(edge > 0)
    unique_y_ids=np.unique(y_ids)
    for y in unique_y_ids:
        ids=np.where(y_ids==y)[0]
        x_selected = x_ids[ids]
        if len(x_selected)==1:
            edge[y,x_selected]=1
        elif len(x_selected) == 2:
            edge[y,np.min(x_selected):np.max(x_selected)]=1
        else:
            x_selected = np.sort(x_selected)
            index=0
            size=len(x_selected)
            while(index < size):
                if (index == size-1):
#                     print(y, x_selected[index])
                    break
                if (np.sum(old_mask[y,x_selected[index]:x_selected[index+1]])/(x_selected[index+1]-x_selected[index]) > 0.5):
                    edge[y,x_selected[index]:x_selected[index+1]]=1
                    index=index+1
                else:
                    edge[y,x_selected[index]]=1
                    index=index+1
    return edge

def find_nearest_point(points, anchor_point):
    if len(points) == 0:
        return None
    if len(points) == 1:
        return points[0]
    selected_point=points[0]
    selected_dist=get_distance(selected_point, anchor_point)
    for i, point in enumerate(points):
        if i==0:
            continue
        dist=get_distance(point, anchor_point)
        if (dist < selected_dist):
            selected_dist=dist
            selected_point=point
    return selected_point

#在像素四周，共9个像素中查找相邻可用的点。
def select_available_neighbors(src_edge, dst_edge, anchor_point):
    x, y = anchor_point
    y_ids, x_ids=np.where(src_edge[y-1:y+2,x-1:x+2]==1)
    y_ids = y_ids+y-1
    x_ids = x_ids+x-1
    selected_y_ids=[]; selected_x_ids=[]
    for x_id, y_id in zip(x_ids, y_ids):
        if dst_edge[y_id,x_id] == 0:
            selected_y_ids.append(y_id)
            selected_x_ids.append(x_id)
    return selected_x_ids, selected_y_ids

#在像素距离margin的四周，查找相邻的点。
def select_available_neighbors_with_margin(src_edge, dst_edge, anchor_point, margin=1):
    x, y = anchor_point
    y_ids, x_ids=np.where(src_edge[y-margin:y+margin+1,x-margin:x+margin+1]==1)
    y_ids = y_ids+y-margin
    x_ids = x_ids+x-margin
    selected_y_ids=[]; selected_x_ids=[]
    for x_id, y_id in zip(x_ids, y_ids):
        if dst_edge[y_id,x_id] == 0:
            selected_y_ids.append(y_id)
            selected_x_ids.append(x_id)
    return selected_x_ids, selected_y_ids


def set_edge(dst_edge, x, y, type):
    dst_edge[y,x]=1
    print(x, y, type)
    


def get_distance(point1, point2):
    sqs=np.square(np.array(point1)-np.array(point2))
    return np.sqrt(np.sum(sqs))

    
def test(mask,img):
    mask_edge=get_mask_edge(mask)
    #Get the image edge
    img_blur=cv2.blur(img,(3,3))
    low_threshold = 100;ratio=2
    img_edge=cv2.Canny(img_blur,low_threshold,low_threshold*ratio,3)
    img_edge=np.where(img_edge>0,1,0)
#     dst_edge=adjust_edge(img_edge, mask_edge)
    dst_edge=adjust_edge(img_edge, mask_edge, img)
    
    
    for c in range(3):
        img[:,:,c]=np.where(img_edge==1,colors[2][c],img[:,:,c])
        img[:,:,c]=np.where(mask_edge==1,colors[3][c],img[:,:,c])
        img[:,:,c]=np.where(dst_edge==1,colors[4][c],img[:,:,c])
    old_mask=np.where(mask==0,1,0)
    dst_mask = get_mask_from_edge(old_mask,dst_edge)
#         dst_mask = mask
    for c in range(3):
        img[:,:,c]=np.where(dst_mask==1, alpha*colors[3][c]+(1-alpha)*img[:,:,c],img[:,:,c])
        
    return img

def get_contour():
    img_path='/Users/chendanxia/sophie/sophie.jpg'
    ratio=2
    kernel_size=3
    low_threshold=100
    hight_threshold=low_threshold*ratio
    src_img=cv2.imread(img_path)
    src_gray=cv2.cvtColor(src_img,cv2.COLOR_BGR2GRAY)
    img_blur=cv2.blur(src_gray,(3,3))
    canny_output=cv2.Canny(img_blur,low_threshold,hight_threshold,kernel_size)
    #cv.CHAIN_APPROX_SIMPLE 压缩水平、垂直和对角线方向的像素。
    #cv.CHAIN_APPROX_NONE 存储所有轮廓点。
    contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    selected_contours=[]
    for contour in contours:
        if len(contour) > 200:
            selected_contours.append(contour)
    contours=selected_contours
    print(contours)
    print(hierarchy)
    drawing=np.zeros((canny_output.shape[0],canny_output.shape[1],3),dtype=np.uint8)
    for i in range(len(contours)):
        color = (random.randint(0,256),random.randint(0,256),random.randint(0,256))
        cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, None, 0)
    cv2.imshow('Contours', drawing)
    cv2.waitKey()
# get_contour()
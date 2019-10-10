import cv2 as cv
import numpy as np

def get_hsv(img):
    return cv.cvtColor(img, cv.COLOR_BGR2HSV)

def get_hist(hsv_base):
    h_bins=50
    s_bins=60
    histSize=[h_bins,s_bins]
    h_range=[0,180]
    s_range=[0,256]
    ranges=h_range+s_range
    channels=[0,1]
    hist_base=cv.calcHist([hsv_base],channels, None, histSize, ranges, accumulate=False)
    cv.normalize(hist_base,hist_base,alpha=0,beta=1, norm_type=cv.NORM_MINMAX)
    return hist_base

def compare_hist(hsv_1, hsv_2, compare_method = 0):
    hist_1=get_hist(hsv_1)
    hist_2=get_hist(hsv_2)
    base_base=cv.compareHist(hist_1, hist_2, compare_method)
    print(base_base)
    if compare_method == 0:
        dist = 1-base_base
    elif compare_method == 1:
        dist = base_base
    return dist

def color_distance(color1, color2):
    sq=np.square(color1-color2)
    sum=np.sum(sq)
    dist=np.sqrt(sum)
    print(dist)
    return dist

def color_image_distance(img, color):
    img_mean=np.mean(img, axis=0)
    img_mean=np.mean(img_mean, axis=0)
    return color_distance(img_mean, color)

def test():
    path='/Users/chendanxia/sophie/sophie.jpg'
    img=cv.imread(path)
    hsv_base=get_hsv(img)
    range=10
    hsv_1=hsv_base[444:445,437:437+range,:]
    img1=img[444:445,437:437+range,:]
    hsv_2=hsv_base[470:471,425:425+range,:]
    img2=img[470:471,425:425+range,:]
    hsv_22=hsv_base[250:251,781:781+range,:]
    img22=img[250:251,781:781+range,:]
    hsv_3=hsv_base[426:427,403:403+range,:]
    hsv_4=hsv_base[584:585,274:274+range,:]
    hsv_5=hsv_base[446:447,879:879+range,:]
    img5=img[446:447,879:879+range,:]
#     compare_hist(hsv_1, hsv_1)
    color1=np.mean(img1,axis=1)
#     color1=np.mean(color1,axis=0)
    print(np.squeeze(color1))
    color2=np.mean(img2,axis=1)
    print(color2)
    color22=np.mean(img22,axis=1)
    print(color22)
    color5=np.mean(img5,axis=1)
    print(color5)
    compare_hist(hsv_1, hsv_2)
    color_distance(np.squeeze(color1), np.squeeze(color2))
    compare_hist(hsv_1, hsv_22)
    color_distance(color1, color22)
    compare_hist(hsv_1, hsv_5)
    color_distance(color1, color5)
    

# test()
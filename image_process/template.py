import cv2 as cv
import numpy as np
from skimage import exposure

src='/Users/chendanxia/sophie/证件照/remove/WechatIMG263.jpeg'
temp='/Users/chendanxia/sophie/证件照/WechatIMG83.jpeg'
rates=np.arange(1.5,1.7,0.01)
gamma_rates=np.arange(0.1, 1.5, 0.1)
gamma_id=0
zoom_rate_id=0
cv.namedWindow('result_window', cv.WINDOW_NORMAL )
cv.namedWindow( 'image_window', cv.WINDOW_NORMAL )
zoom_rate=1.53
def MatchTemplate(gamma_id):
    src_img=cv.imread(src)
    temp_img=cv.imread(temp)
    gamma=gamma_rates[gamma_id]
    print(gamma)
#     zoom_rate=rates[zoom_rate_id]
#     print(zoom_rate)
    temp_img=cv.resize(temp_img,dsize=(0,0),fx=zoom_rate,fy=zoom_rate)
    temp_img=exposure.adjust_gamma(temp_img,gamma)
    match_method=cv.TM_SQDIFF
    result=cv.matchTemplate(src_img,temp_img,match_method)
    cv.normalize( result, result, 0, 1, cv.NORM_MINMAX, -1 )
    _minVal, _maxVal, minLoc, maxLoc = cv.minMaxLoc(result, None)
    print(_minVal, _maxVal, minLoc, maxLoc)
    if (match_method == cv.TM_SQDIFF or match_method == cv.TM_SQDIFF_NORMED):
        matchLoc = minLoc
    else:
        matchLoc = maxLoc
    if matchLoc[0] + temp_img.shape[0] > src_img.shape[0] or matchLoc[1] + temp_img.shape[1] > src_img.shape[1]:
        return 1
    start0=matchLoc[0]
    end0=matchLoc[0] + temp_img.shape[0]
    start1=matchLoc[1]
    end1=matchLoc[1] + temp_img.shape[1]
#     src_img[start0:end0, start1:end1]=np.where(temp_img==255, src_img[start0:end0, start1:end1], temp_img)
    src_img[start0:end0, start1:end1]=temp_img
    cv.rectangle(src_img, matchLoc, (matchLoc[0] + temp_img.shape[0], matchLoc[1] + temp_img.shape[1]), (0,0,0), 2, 8, 0 )
    cv.rectangle(result, matchLoc, (matchLoc[0] + temp_img.shape[0], matchLoc[1] + temp_img.shape[1]), (0,0,0), 2, 8, 0 )
    cv.imshow('result_window', result)
    cv.imshow('image_window', src_img)
    cv.waitKey()

def ShowMatchTemplate(zoom_rate):
    src_img=cv.imread(src)
    temp_img=cv.imread(temp)
#     zoom_rate=rates[zoom_rate_id]
    temp_img=cv.resize(temp_img,dsize=(0,0),fx=zoom_rate,fy=zoom_rate)
    match_method=cv.TM_SQDIFF
    result=cv.matchTemplate(src_img,temp_img,match_method)
    cv.normalize( result, result, 0, 1, cv.NORM_MINMAX, -1 )
    _minVal, _maxVal, minLoc, maxLoc = cv.minMaxLoc(result, None)
    print(_minVal, _maxVal, minLoc, maxLoc)
    if (match_method == cv.TM_SQDIFF or match_method == cv.TM_SQDIFF_NORMED):
        matchLoc = minLoc
    else:
        matchLoc = maxLoc
    if matchLoc[0] + temp_img.shape[0] > src_img.shape[0] or matchLoc[1] + temp_img.shape[1] > src_img.shape[1]:
        return 1
    start0=matchLoc[0]
    end0=matchLoc[0] + temp_img.shape[0]
    start1=matchLoc[1]
    end1=matchLoc[1] + temp_img.shape[1]
    src_img[start0:end0, start1:end1]=np.where(temp_img==255, src_img[start0:end0, start1:end1], temp_img)
    cv.rectangle(src_img, matchLoc, (matchLoc[0] + temp_img.shape[0], matchLoc[1] + temp_img.shape[1]), (0,0,0), 2, 8, 0 )
    cv.rectangle(result, matchLoc, (matchLoc[0] + temp_img.shape[0], matchLoc[1] + temp_img.shape[1]), (0,0,0), 2, 8, 0 )
    cv.imshow('result_window', result)
    cv.imshow('image_window', src_img)
    cv.waitKey()

trackbar_label = 'Find the best'
# cv.createTrackbar(trackbar_label,'image_window',zoom_rate_id,20,MatchTemplate)
# MatchTemplate(zoom_rate_id)
# cv.waitKey()

# minVal=1
# rate=0
# for i in np.arange(1.369,1.371,0.0001):
#     try:
#         val=MatchTemplate(i)
#         if np.abs(val) < np.abs(minVal):
#             minVal=val;
#             rate=i 
#         print(i,val)
#     except Exception as e:
#         break
# print(minVal,rate)
# ShowMatchTemplate(rate)

cv.createTrackbar(trackbar_label,'image_window',gamma_id,15,MatchTemplate)
cv.waitKey()
# MatchTemplate(1.53)


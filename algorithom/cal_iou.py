from sklearn.metrics import confusion_matrix
import numpy as np
import math
import imageio
import os
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from tkinter import image_names

def getScoreAverage(scoreList):
    validScores = 0
    scoreSum    = 0.0
    for score in scoreList:
        if not math.isnan(score):
            validScores += 1
            scoreSum += score
    if validScores == 0:
        return float('nan')
    return scoreSum / validScores

def compute(preds, y):
    conf_mat= confusion_matrix(y.flatten(), preds.flatten(), range(0, 2))
    perclass_tp = np.diagonal(conf_mat).astype(np.float32)
    perclass_fp = conf_mat.sum(axis=0) - perclass_tp #SUM(nji) - nii
    perclass_fn = conf_mat.sum(axis=1) - perclass_tp
    iou= perclass_tp/(perclass_fp+ perclass_tp+ perclass_fn)
    iou= iou[:-1]
    mean_iou_index= getScoreAverage(iou)
    return mean_iou_index

def merge_and_save(file, iou_dict):
    iou_dict1=np.load(file).item()
    for key in iou_dict1.keys():
        if not iou_dict.get(key)==None:
            iou_dict[key]=iou_dict1[key]
    np.save(file, iou_dict)
    return iou_dict

def compute_iou():
    base_dir='/Users/chendanxia/sophie/segmentation_img_set/human_sum/all/'
#     base_dir='/home/nemo/segmentation_data/coco/'
    mask_dir=base_dir+'masks/'
    ious_file=base_dir+'ious.npy'
    iou_dict=np.load(ious_file).item()
    file_names=os.listdir(mask_dir)
    sum=len(file_names)
    print('{}/{} exists'.format(len(iou_dict.keys()),sum))
    
    file_names.sort()
#     file_names.reverse()
#     start_index=sum*0.7
#     for i, file_name in enumerate(file_names):
    i=0
    for file_name in tqdm(file_names):
#         file_name = file_names[i]
#         if i < start_index:
#             continue
        if not iou_dict.get(file_name)==None:
            continue
        mask_path=mask_dir+file_name
        try:
            img=imageio.imread(mask_path)
        except Exception as e:
            print('Fail to read {}'.format(file_name),e)
            continue
        img1=np.zeros(img.shape)
        iou=compute(img, img1)
        iou_dict[file_name]=iou
        i=i+1
        if i % 200 == 0:
            print("{} Progress {:.2f}%".format(datetime.now(),i/sum*100))
            iou_dict=merge_and_save(ious_file, iou_dict)
    merge_and_save(ious_file, iou_dict)

def compute_person_percentage():
    base_dir='/Users/chendanxia/sophie/segmentation_img_set/human_sum/all/'
    mask_dir=base_dir+'masks/'
    dst_file=base_dir+'person_percentage.npy'
    file_names=os.listdir(mask_dir)
    file_names.sort()
    person_percentage_dict={}
    for file_name in tqdm(file_names):
        mask_path=mask_dir+file_name
        try:
            img=imageio.imread(mask_path)
        except Exception as e:
            print('Fail to read {}'.format(file_name),e)
            continue
        all=img.size
        bg=np.sum(img)
        person=all-bg
        percentage=person/all 
        person_percentage_dict[file_name]=percentage
#         print(percentage)
        
    np.save(dst_file, person_percentage_dict)

def get_person_part(img, mask):
    h,w=mask.shape
    mask3=np.zeros((3,h,w))
    mask3[0]=mask
    mask3[1]=mask
    mask3[2]=mask
    mask3=np.transpose(mask3, axes=[1,2,0])
    return np.where(mask3>0,0,img)

def analysis():
    base_dir='/Users/chendanxia/sophie/segmentation_img_set/human_sum/all/'
    dst_dir=base_dir+"temp/"
    ious=np.load(base_dir+'person_percentage.npy').item()
    count=0
    selected=[]
#     all=np.load(base_dir+"all_names.npy")
    all=os.listdir(base_dir+'images')
    for key in tqdm(ious.keys()):
#         if ious[key] > 0.01 and ious[key] < 0.012:
        if ious[key] > 0.1:
            mask=imageio.imread(base_dir+'masks/'+key)
            img_name=key.replace('_seg.png','.jpg')
            count=count+1
            if img_name not in all:
                print(img_name)
                continue
            img=imageio.imread(base_dir+'images/'+img_name)
            dst_img=get_person_part(img,mask)
            imageio.imsave(dst_dir+img_name,dst_img)
#             plt.subplot(121)
#             plt.imshow(img)
#             plt.subplot(122)
#             plt.imshow(dst_img)
#             plt.show()
            
            selected.append(img_name)
    print("{}/{}".format(count,len(ious.keys())))
    print(len(selected))
    np.save(base_dir+"selected_names.npy",selected)
#     print(len(ious.keys()))
    
# compute_person_percentage()
analysis()

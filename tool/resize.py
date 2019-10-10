import cv2 as cv
import argparse
import sys
from tqdm import tqdm
import os

def resize_and_save(src_path, dst_path, args):
    src_img=cv.imread(src_path)
    dst_img=cv.resize(src_img, (args.size, args.size))
    cv.imwrite(dst_path, dst_img)

def resize_train():
    args=parseArgument()
    if not os.path.exists(args.dst):
        os.makedirs(args.dst)
    for name in tqdm(os.listdir(args.src)):
        src_path=os.path.join(args.src,name)
        dst_path=os.path.join(args.dst, name)
        if os.path.isdir(src_path):
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            for sub_name in os.listdir(src_path):
                sub_src_path = os.path.join(src_path, sub_name)
                sub_dst_path = os.path.join(dst_path, sub_name)
                resize_and_save(sub_src_path, sub_dst_path, args)
                
        else:
            resize_and_save(src_path, dst_path, args)
        
    
def parseArgument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='/home/nemo/imagenet/imagenet_train/')
    parser.add_argument('--dst', default='/home/nemo/imagenet/imagenet_train_128/')
    parser.add_argument('--size', default=128, type=int)
    return parser.parse_args(sys.argv[1:])

resize_train()
import os

path='../file/names.txt'
path='/Users/chendanxia/Downloads/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'
with open(path) as f:
    content=f.read();
    strs=content.split('\n')
    strs=[str for str in strs]
    print(content)
    print("end")
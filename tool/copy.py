import os

def copy_files(src_dir, dst_dir):
    for name in os.listdir(src_dir):
        src_path=os.path.join(src_dir,name)
        cmd='cp {} {}'.format(src_path, dst_dir)
        print(cmd)
        os.system(cmd)
    
def scp_files(src_dir, dst_dir):
    for name in os.listdir(src_dir):
        src_path=os.path.join(src_dir,name)
        cmd='scp {} {}'.format(src_path, dst_dir)
        print(cmd)
        os.system(cmd)

# copy_files('/home/nemo/segmentation_data/all/xiaoyu_images', '/home/nemo/segmentation_data/all/images')
# copy_files('/home/nemo/segmentation_data/all/xiaoyu_masks', '/home/nemo/segmentation_data/all/masks')
# copy_files('/Users/chendanxia/sophie/human_seg/xiaoyu_masks/', 'gpu:/home/nemo/segmentation_data/all/masks/')
import os
import shutil
import json
import numpy as np

num_class=2
val_per=10
path_to_train_dataset_dir='/home/kbkb/github/keras-mobilenet/train'
path_map_dir='./train'
path_map_val_dir='./val'

def rmrf_mkdir(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
        
    os.mkdir(dirname)

def file_process(dir):
    pathMap=[path_map_dir,path_map_val_dir]
    for path in pathMap:
        rmrf_mkdir(path)
        for i in range(num_class):
            os.mkdir(os.path.join(path,str(i)))
def map_file(dir):
    rootdir=os.listdir(dir)
    for subdir in rootdir:
        path=os.path.join(dir,subdir)
        train_filenames = os.listdir(path)
        index=subdir
        path_file_length_num= len(train_filenames)
        for filename in train_filenames:
            sub_filename=filename.rstrip('.jpg')
            if np.random.randint(100)>=val_per:
                os.symlink(dir+'/'+str(index)+'/'+filename,path_map_dir+'/'+str(index)+'/'+str(index)+'_'+sub_filename+'.jpg')
            else:
                os.symlink(dir+'/'+str(index)+'/'+filename,path_map_val_dir+'/'+str(index)+'/'+str(index)+'_'+sub_filename+'.jpg')

def _main(dir):
    file_process(dir)
    map_file(path_to_train_dataset_dir)

if __name__=='__main__':
    _main(path_to_train_dataset_dir)
    print('Done!')

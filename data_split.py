import numpy as np
import os
import shutil

datapath = 'train_image'
savepath = 'categorical_image'

data_file = 'train.csv'
data = np.genfromtxt(data_file, delimiter=',', dtype=str)
# unique, counts = np.unique(data[:,1], return_counts=True)
# print(dict(zip(unique, counts)))

def mk_split_dir(type, imgs, idx_min, count):
    dir = os.path.join(savepath, type)
    for cls in np.unique(data[:,1]):
        os.makedirs(os.path.join(dir,cls), exist_ok=True)

    fnames = [['%05d.jpg' % kk, data[np.argwhere(data=='%05d.jpg' % kk)[0][0]][1]] for kk in imgs[idx_min: idx_min + count]]
    for fname in fnames:
        src = os.path.join(datapath, fname[0])
        dst = os.path.join(dir, fname[1], fname[1]+'_'+fname[0])

        if not os.path.exists(dst):
            shutil.copyfile(src, dst)

imgs = np.arange(1,12994)
np.random.seed(0)
np.random.shuffle(imgs)

mk_split_dir('train', imgs, 0, 9100)
mk_split_dir('val', imgs, 9100, 1900)
mk_split_dir('test', imgs, 11000, 1994)

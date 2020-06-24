import os
import sys
import cv2
import joblib
import random
from utils import load_img_paths
from create_mask import mask_zoo

test_num = 1000
mask_type = 'simpler'  # 不规则mask的方式
mask_generator = mask_zoo[mask_type]

img_root = '/home/hanbing/hanbing_data/datasets/celebA/celebahq/data1024x1024/'
mask_root = '/home/hanbing/hanbing_data/datasets/celebA/celebahq/masks/'
dataname = 'Celeba-HQ'

save_test = '../flist/{}_test.txt'.format(dataname)
save_train = '../flist/{}_train.txt'.format(dataname)

if os.path.isdir(img_root):
    img_paths = load_img_paths(img_root)
else:
    img_paths = joblib.load(img_root)

mask_root = os.path.join(mask_root, mask_type)
os.makedirs(mask_root, exist_ok=True)
random.shuffle(img_paths)

with open(save_test, 'w') as test:
    
    for i, img_path in enumerate(img_paths[:test_num]):
        mask = mask_generator(512,512)  # 512
        mask_path = os.path.join(mask_root, '{}.png'.format(i))
        cv2.imwrite(mask_path, mask)
        test.write(img_path + ',' + mask_path + '\n')

with open(save_train, 'w') as train:
    for img_path in img_paths[test_num:]:
        train.write(img_path + '\n')

    

import os
import cv2
import imutils
import random
import joblib
import numpy as np
from glob import glob
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset, DataLoader

from core.utils import load_img_paths
from core.create_mask import mask_zoo

import albumentations as A  

class CommonDataset(Dataset):
    def __init__(self, flist, 
                    size,
                    mask_type,
                    dataname,
                    mode='train'):
        self.mode = mode
        self.img_paths = []
        with open(flist, 'r') as f:
            for line in f.readlines():
                if self.mode == 'train':
                    line = line.strip()
                    
                else:
                    line = line.strip().split(',')
                self.img_paths.append(line)

        random.shuffle(self.img_paths)
        print('>>> {}: 加载数据总共：{}张图片'.format(mode, len(self.img_paths)))

        self.size = size
        self.mask_gen = mask_zoo[mask_type]

        if dataname == 'celebahq':  # 训练的都是1024*1024，所以可以直接resize
            self.aug = A.Resize(size, size, interpolation=cv2.INTER_CUBIC)
        else:
            self.aug = A.Compose([A.PadIfNeeded(min_height=size, min_width=size),
                                A.RandomCrop(size, size)])
      



    def __getitem__(self, idx):
        
        

        line = self.img_paths[idx]
        if self.mode == 'train':
            img = Image.open(line).convert('RGB')
            mask = self.mask_gen(self.size, self.size)   # paint region is 255
        else:
            img = Image.open(line[0]).convert('RGB')
            mask = np.array(Image.open(line[1]))   # paint region is 255
            mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        img = np.array(img)
        h,w,c = img.shape

        img = self.aug(image=img)['image']
        mask = np.expand_dims(mask, axis=-1)

        img = img.astype('float32') / 255.0
        mask = 1. - mask.astype('float32') / 255.0
        img_miss = img * mask

        img = np.ascontiguousarray(img.transpose(2,0,1))
        img_miss = np.ascontiguousarray(img_miss.transpose(2,0,1))
        mask = np.ascontiguousarray(mask.transpose(2,0,1))
        return img, img_miss, mask
 
    def __len__(self):
        return len(self.img_paths)
        


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_imgs, self.next_imgs_miss, self.next_masks = next(self.loader)
        except StopIteration:
            self.next_imgs = None
            self.next_imgs_miss = None
            self.next_masks = None
            return
     
        with torch.cuda.stream(self.stream):
            self.next_imgs = self.next_imgs.cuda(non_blocking=True)
            self.next_imgs_miss = self.next_imgs_miss.cuda(non_blocking=True)
            self.next_masks = self.next_masks.cuda(non_blocking=True)
            

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        imgs = self.next_imgs
        imgs_miss = self.next_imgs_miss
        masks = self.next_masks
        if imgs is not None:
            imgs.record_stream(torch.cuda.current_stream())
        if imgs_miss is not None:
            imgs_miss.record_stream(torch.cuda.current_stream())
        if masks is not None:
            masks.record_stream(torch.cuda.current_stream())
        self.preload()
        return imgs, imgs_miss, masks



if __name__ == '__main__':
    import os
    import sys
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    path = sys.argv[1]
    dataset = CommonDataset(img_root=path)
    # img, img_miss, mask = dataset.__getitem__(1000)
    # img = img.transpose(1,2,0)
    # img_miss = img_miss.transpose(1,2,0)
    # img = (img * 255).astype('uint8')
    # img_miss = (img_miss * 255).astype('uint8')
    # mask = np.squeeze((mask * 255).astype('uint8'))
    # cv2.imwrite('./samples/dummyimgs/img.jpg', img)
    # cv2.imwrite('./samples/dummyimgs/img_miss.jpg', img_miss)
    # cv2.imwrite('./samples/dummyimgs/mask.png', mask)

    i = 0
    dataloader = DataLoader(dataset=dataset,
                           batch_size=32,
                           shuffle=True,
                           num_workers=12,
                           sampler=None,
                           pin_memory=False,
                           drop_last=True)

    for img, img_miss, mask in dataloader:
        print(i, img.shape, mask.shape)
        i += 1
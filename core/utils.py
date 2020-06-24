import os
import torch
import cv2
import imutils
import numpy as np
from collections import OrderedDict



def load_img_paths(ROOT, suffixes=['jpg', 'png', 'jpeg'], absolute=True, filtered=[]):
    '''获取该目录下的所有图片文件，格式筛选参照suffixes，可以根据filterd滤除无关的文件夹
    '''
    print('>>>开始加载图片')
    paths = []
    count = 0
    ROOT = os.path.abspath(ROOT)
    for root, dirs, files in os.walk(ROOT):
        for f in files:
            # if count > 5000:
            #     return paths
            suffix = f.split('.')[-1]
            tag = True
            if (f[0] != '.') and (suffix.lower() in suffixes):
                for t in filtered:
                    if t in root:
                        tag = False
                if tag:
                    if absolute:
                        paths.append(os.path.join(root, f))
                    else:
                        paths.append(f)
                    count += 1 
    print('>>>加载完毕, 总共{}张图片'.format(len(paths)))
    return paths

 

def load_state_dict(weight_path):
    '''保存模型有时候忘了是多gpu还是单gpu
    （它们的区别就在于state_dict的key有没有多出一个module）
    所以统一加载的时候用cpu的方式
    '''
    old_state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in old_state_dict.items():
        if ('module.' in k):
            k = k[7:] # remove `module.`
        new_state_dict[k] = v

    return new_state_dict


def load_auther_pretrained_weights(weight_path):
    old_state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        if 'de_' in k:
            number = 7 - int(k[3])
            k = 'de.{}'.format(number) + k[4:]
        elif 'fuse' in k:
            number = 7 - int(k[5])
            k = 'fuse.{}'.format(number) + k[6:]
        else:
            k = k.replace('_', '.', 1)
        new_state_dict[k] = v

    return new_state_dict




def resize(img, mask, size=512, scale=128):
    # mask[mask > 0] = 255
    row_h, row_w, _ = img.shape
    
    if max(row_h, row_w) > size:
        if row_h < row_w:
            img = imutils.resize(img, width=size, inter=cv2.INTER_LINEAR)
            mask = imutils.resize(mask, width=size, inter=cv2.INTER_NEAREST)
        else:
            img = imutils.resize(img, height=size, inter=cv2.INTER_LINEAR)
            mask = imutils.resize(mask, height=size, inter=cv2.INTER_NEAREST)
    
    h, w, _ = img.shape
    new_h, new_w = h // scale * scale, w // scale * scale
    print('new h w: ', new_h, new_w)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    print('img shape: ', img.shape, 'mask shape: ', mask.shape)
    return img, mask


def crop_resize1(img, mask, size=512, scale=256):
    info = cv2.findContours(1-mask[:,:,0].astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(info) == 3:
        info = info[1:]
    H, W, _ = img.shape
    # cv2.imwrite('mask.png', mask[:,:,0]*255)
    crop_list = []
    for cnt in info[0]:
        pad = size
        x,y,w,h = cv2.boundingRect(cnt)
        
        if max(w,h) > pad:
            pad = max(w,h)

        c_x, c_y = x + w//2, y + h//2
        left, right = c_x - pad//2, c_x + pad//2
        top, bottom = c_y - pad//2, c_y + pad//2
        
        if left < 0 and right > W:
            left, right = 0, W
        elif left < 0:
            right += abs(left)
            left = 0
            if right > W:
                right = W
        elif right > W:
            left -= abs(right-W)
            right = W
            if left < 0:
                left = 0

        if top < 0 and bottom > H:
            top, bottom = 0, H
        elif top < 0:
            bottom += abs(top)
            top = 0
            if bottom > H:
                bottom = H
        elif bottom > H:
            top -= abs(bottom-H)
            bottom = H
            if top < 0:
                top = 0
        
        coord = [top, left, bottom, right]
        # print('裁剪到的图像大小: ', (bottom-top, right-left))
        crop_img = img[top:bottom, left:right, :]
        crop_mask = mask[top:bottom, left:right, :]

        ori_crop_img = crop_img.copy()
        ori_crop_mask = crop_mask.copy()

        h, w, c = crop_img.shape
        if pad > size:
            if w > h:
                crop_img = imutils.resize(crop_img, width=size, inter=cv2.INTER_CUBIC)
                crop_mask = imutils.resize(crop_mask, width=size, inter=cv2.INTER_NEAREST)
            else:
                crop_img = imutils.resize(crop_img, height=size, inter=cv2.INTER_CUBIC)
                crop_mask = imutils.resize(crop_mask, height=size, inter=cv2.INTER_NEAREST)

        # if pad > size:
        #     crop_img = cv2.resize(crop_img, (size,size), interpolation=cv2.INTER_CUBIC)
        #     crop_mask = cv2.resize(crop_mask, (size,size), interpolation=cv2.INTER_NEAREST)

        crop_h, crop_w, _ = crop_img.shape
        # print('裁剪+resize的图像大小: ', (crop_h, crop_w))
        crop_h, crop_w = crop_h // scale * scale, crop_w // scale * scale
        # print('裁剪+resize+scale的图像大小: ', (crop_h, crop_w))
        crop_img = cv2.resize(crop_img, (crop_w,crop_h), interpolation=cv2.INTER_CUBIC)
        crop_mask = cv2.resize(crop_mask, (crop_w,crop_h), interpolation=cv2.INTER_NEAREST)

        crop_list.append([crop_img, crop_mask, coord, ori_crop_img, ori_crop_mask])
    return crop_list



def crop_resize(img, mask, size=1024, scale=8, padding_rate=1.5):
    info = cv2.findContours(1 - mask[:,:,0].astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(info) == 3:
        info = info[1:]
    H, W, _ = img.shape
    # cv2.imwrite('mask.png', mask[:,:,0]*255)
    crop_list = []
    for cnt in info[0]:
        
        x,y,w,h = cv2.boundingRect(cnt)
        pad = max(int(max(w,h)*padding_rate), size)
        # print('padding size: ', pad)

        c_x, c_y = x + w//2, y + h//2
        left, right = c_x - pad//2, c_x + pad//2
        top, bottom = c_y - pad//2, c_y + pad//2
        # print('裁剪到的图像大小: ', (bottom, top, right, left))
        if left < 0 and right > W:
            left, right = 0, W
        elif left < 0:
            right += abs(left)
            left = 0
            if right > W:
                right = W
        elif right > W:
            left -= abs(right-W)
            right = W
            if left < 0:
                left = 0

        if top < 0 and bottom > H:
            top, bottom = 0, H
        elif top < 0:
            bottom += abs(top)
            top = 0
            if bottom > H:
                bottom = H
        elif bottom > H:
            top -= abs(bottom-H)
            bottom = H
            if top < 0:
                top = 0
        
        coord = [top, left, bottom, right]
        print('裁剪到的图像大小: ', (bottom-top, right-left))
        crop_img = img[top:bottom, left:right, :]
        crop_mask = mask[top:bottom, left:right, :]

        ori_crop_img = crop_img.copy()
        ori_crop_mask = crop_mask.copy()

        h, w, c = crop_img.shape
        if pad > size:
            if w > h:
                crop_img = imutils.resize(crop_img, width=size, inter=cv2.INTER_AREA)
                crop_mask = imutils.resize(crop_mask, width=size, inter=cv2.INTER_NEAREST)
            else:
                crop_img = imutils.resize(crop_img, height=size, inter=cv2.INTER_AREA)
                crop_mask = imutils.resize(crop_mask, height=size, inter=cv2.INTER_NEAREST)
        
        # if pad > size:
        #     crop_img = cv2.resize(crop_img, (size,size), interpolation=cv2.INTER_CUBIC)
        #     crop_mask = cv2.resize(crop_mask, (size,size), interpolation=cv2.INTER_NEAREST)

        crop_h, crop_w, _ = crop_img.shape
        print('裁剪+resize的图像大小: ', (crop_h, crop_w))
        crop_h, crop_w = crop_h // scale * scale, crop_w // scale * scale
        print('裁剪+resize+scale的图像大小: ', (crop_h, crop_w))
        crop_img = cv2.resize(crop_img, (crop_w,crop_h), interpolation=cv2.INTER_AREA)
        crop_mask = cv2.resize(crop_mask, (crop_w,crop_h), interpolation=cv2.INTER_NEAREST)
        # print('ori_crop_img shape: ', ori_crop_img.shape, 'crop_img shape: ', crop_img.shape)
        crop_list.append([crop_img, crop_mask, coord, ori_crop_img, ori_crop_mask])
    return crop_list
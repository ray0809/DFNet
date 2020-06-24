import os
import sys
from pathlib import Path

import cv2
import imutils
import numpy as np
import torch
import tqdm

from core.utils import load_state_dict, resize, crop_resize
from model.dfnet import DFNet
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')


size = 512
ckpt = sys.argv[1]
img_path = sys.argv[2]


print('press \'esc\' to exit, \'i\' to remask, \'Enter to inpainting\'!')
drawing = False # true if mouse is pressed
ix,iy = -1,-1
radius = 20






ori = cv2.imread(img_path, 1)
h, w, c = ori.shape
mask = np.ones((h,w,3), np.uint8)


# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if drawing == True:
            # radius = np.random.randint(20,45)
            cv2.circle(mask,(x,y),radius,(0,0,0),-1)
    else:
        pass



def preprocess(img, mask):
    img = np.ascontiguousarray(img.transpose(2, 0, 1)).astype('float32')
    img /= 255.0
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).to(device)

    mask = np.expand_dims(np.expand_dims(mask, 0), 0)
    mask = mask.astype('float32')
    mask = torch.from_numpy(mask).to(device)

    return img, mask

def postprocess(results, coord):
    results = results.mul(255).byte().data.cpu().numpy()
    results = np.transpose(results, [1,2,0])
    results = results[:,:,::-1] # rgb2bgr
    results = cv2.resize(results, (coord[3]-coord[1], coord[2]-coord[0]), interpolation=cv2.INTER_CUBIC)
    return results

if __name__ == '__main__':
    
    model = DFNet(blend_layers=[0])
    model.load_state_dict(load_state_dict(ckpt))
    model = model.to(device)
    model.eval()

    

    
    cv2.namedWindow('imshow', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('results', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.createTrackbar('radius','imshow',10,100,lambda x: None)
    cv2.setMouseCallback('imshow',draw_circle)

    img = ori.copy()
    while(1):
        cv2.imshow('imshow',img*mask)
        radius = cv2.getTrackbarPos('radius', 'imshow')
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # Esc
            break
        elif k == ord('i'):
            img = ori.copy()
            mask = np.ones((h,w,3), np.uint8)
        elif k == 13:  # Enter
            
            crop_list = crop_resize(img, mask, size=512, scale=128, padding_rate=2)
            for img_, mask_, coord, ori_img_, ori_mask_ in crop_list:
                img_ = img_[:,:,::-1]  # bgr2rgb
                tensor_img, tensor_mask = preprocess(img_*mask_, mask_[:,:,0])
                result = model(tensor_img, tensor_mask)[0][0][0]

                result = postprocess(result, coord)

                ori_img_[ori_mask_ == 0] = result[ori_mask_ == 0]
                img[coord[0]:coord[2], coord[1]:coord[3]] = ori_img_

       

            cv2.imshow('results', img)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            else:
                # img = results.copy()
                mask = np.ones((h,w,3), np.uint8)

        else:
            pass
    
    cv2.destroyAllWindows()

   
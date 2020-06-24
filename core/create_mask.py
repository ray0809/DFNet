import os
import sys
import cv2
import math
import random
from random import randint
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
# from utils import load_img_paths





# free-form 
def random_irregular_mask_free_form(height, width, min_ratio=0.1, max_ratio=0.3):
    '''
    height, width：生成的mask的宽高
    area：容忍的涂抹区域的最小面积比
    min,max：涂抹的条状物的宽度大小范围
    '''
    min_num_vertex = 4
    max_num_vertex = 8
    mean_angle = 2*math.pi/5
    angle_range = 2*math.pi/15
    min_width = max(height, width) * min_ratio
    max_width = max(height, width) * max_ratio
    # min_width = 12
    # max_width = 40
    def generate_mask(H, W):
        average_radius = math.sqrt(H*H+W*W) /8
        mask = Image.new('L',(W,H),0)

        for _ in range(np.random.randint(1,4)):
            num_vertex = np.random.randint(min_num_vertex,max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0,angle_range)
            angle_max = mean_angle + np.random.uniform(0,angle_range)

            angles = []
            vertex = []
            for i in range(num_vertex):
                if i%2 == 0:
                    angles.append(2*math.pi - np.random.uniform(angle_min,angle_max))
                else:
                    angles.append(np.random.uniform(angle_min,angle_max))

            w,h = mask.size
            # 对于人脸，感觉还是在中心区域开始为好
            # 对于自然场景，则需要任意位置
            begin, end = 0.0, 1.0
            w1, w2 = int(begin*w), int(end*w)
            h1, h2 = int(begin*h), int(end*h)
            vertex.append((int(np.random.randint(w1,w2)),int(np.random.randint(h1,h2))))
            # vertex = [(np.random.randint(w1,w2), np.random.randint(h1,h2)) for _ in range(num_vertex)]
            for i in range(num_vertex):
                r = np.clip(np.random.normal(loc=average_radius,scale=average_radius//2),0,2*average_radius)
                new_x = np.clip(vertex[-1][0]+r*math.cos(angles[i]),0,w)
                new_y = np.clip(vertex[-1][1]+r*math.sin(angles[i]),0,h)
                vertex.append((int(new_x),int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(min_width,max_width))
            draw.line(vertex,fill=255,width=width)
            for v in vertex:
                draw.ellipse((v[0]-width//2,v[1]-width//2,v[0]+width//2,v[1]+width//2),fill=255)

        #mask.show()
        mask = np.asarray(mask, np.uint8)
        if np.random.normal()>0:   # left right
            mask = mask[:, ::-1]
        if np.random.normal()>0:   # up down
            mask = mask[::-1, :]
        # mask = np.reshape(mask,(1,1,H,W))
        return mask

    mask = generate_mask(height, width)
    while True:
        area = (mask > 0).mean()
        if area > min_ratio and area < max_ratio:
            break
        mask = generate_mask(height, width)
    return mask


def random_irregular_mask_partial_conv(height, width):
    """Generates a random irregular mask with lines, circles and elipses"""
    size = (height, width)
    img = np.zeros((size[0], size[1]), np.uint8)

    # Set size scale
    max_width = 20
    if size[0] < 64 or size[1] < 64:
        raise Exception("Width and Height of mask must be at least 64!")
    
    low = np.sqrt(height * width) // 256 * 12
    high = low * 3
    # print(low, high)
    number = random.randint(low, high)
    for _ in range(number):
        model = random.random()
        if model < 0.6:
            # Draw random lines
            x1, x2 = randint(1, size[0]), randint(1, size[0])
            y1, y2 = randint(1, size[1]), randint(1, size[1])
            thickness = randint(4, max_width)
            cv2.line(img, (x1, y1), (x2, y2), 255, thickness)

        elif model > 0.6 and model < 0.8:
            # Draw random circles
            x1, y1 = randint(1, size[0]), randint(1, size[1])
            radius = randint(4, max_width)
            cv2.circle(img, (x1, y1), radius, 255, -1)

        elif model > 0.8:
            # Draw random ellipses
            x1, y1 = randint(1, size[0]), randint(1, size[1])
            s1, s2 = randint(1, size[0]), randint(1, size[1])
            a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
            thickness = randint(4, max_width)
            cv2.ellipse(img, (x1, y1), (s1, s2), a1, a2, a3, 255, thickness)

    img = img.astype('uint8')
    

    return img


# https://github.com/zhaoyuzhi/deepfillv2
def random_irregular_mask_github(height, width, min_ratio=0.2, max_ratio=0.3):
    """Generate a random free form mask with configuration.
    Args:
        config: Config should have configuration including IMG_SHAPES,
            VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
    Returns:
        tuple: (top, left, height, width)
    """
    max_angle, max_len, times = 10, 30, 15
    MAX = max(height, width) / 2

    def generate_mask(height, width):
        mask = np.zeros((height, width), np.float32)
        timess = np.random.randint(times)
        for i in range(timess):
            start_x = np.random.randint(width)
            start_y = np.random.randint(height)
            for j in range(1 + np.random.randint(5)):
                angle = 0.01 + np.random.randint(max_angle)
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 30 + np.random.randint(max_len)
                # brush_w = 5 + np.random.randint(max_width)
                brush_w = np.random.randint(MAX*min_ratio, MAX*max_ratio)
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)
                cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                start_x, start_y = end_x, end_y
        return mask
    
    mask = generate_mask(height, width)
    while True:
        area = (mask > 0).mean()
        if area > min_ratio and area < max_ratio:
            break
        mask = generate_mask(height, width)
    
    
    return (mask * 255).astype('uint8')


mask_zoo = {'pconv': random_irregular_mask_partial_conv,
            'free-from': random_irregular_mask_free_form,
            'simpler': random_irregular_mask_github}



if __name__ == '__main__':
    size = 256
    min_rate, max_rate = 0.2, 0.3
    # for i in range(8):
    #     mask = 255 - random_irregular_mask_free_form(size, size)
    #     # mask = cv2.resize(mask, (256,256), interpolation=cv2.INTER_NEAREST)
    #     # print((mask < 10).mean())
    #     name = str(960 + i).zfill(5)
    #     cv2.imwrite('./samples/celebahq/mask/{}.png'.format(name), mask)

    # mask = random_irregular_mask_free_form(512, 512, 0.1, 0.2)
    # mask = random_irregular_mask_free_form(size, size, min_rate, max_rate)
    mask = random_irregular_mask_github(size, size, min_rate, max_rate)

    # cv2.imwrite('../../../test_attention/pic/mask.png', cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
    
    while(1):
        # mask = cv2.resize(mask, (32,32), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('test', mask)
        
        # print(mask.max())
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k == 13:  # Enter
            # mask = random_irregular_mask_free_form(size, size, min_rate, max_rate)
            mask = random_irregular_mask_github(size, size, min_rate, max_rate)
            print((mask > 0).mean())
        else:
            pass

    # size = 512
    # img_root = sys.argv[1]
    # save_mask_root = sys.argv[2]
    # if not os.path.isdir(save_mask_root):
    #     os.makedirs(save_mask_root)

    # paths = load_img_paths(img_root)
    # for path in tqdm(paths):
    #     save_path = path.replace(img_root, save_mask_root).split('.')[0] + '.png'
    #     mask = random_irregular_mask_free_form(size, size, 0.1, 0.2)
    #     cv2.imwrite(save_path, mask)
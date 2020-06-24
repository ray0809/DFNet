import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter



class VisWriter():
    def __init__(self, logdir='./logs', ):
        self.writer = SummaryWriter(logdir=logdir)
        

    def summary_scalar(self, dic, step, title='loss/'):
        for k, v in dic.items():
            self.writer.add_scalar(title+k, v, step)

    def summary_img(self, imgs_cat, step, title, lower=0):
        new_imgs_cat = []
        for img in imgs_cat:
            b, c, h, w = img.shape
            if c == 1:
                img = img.expand(b, 3, h, w).data.cpu()
            if lower == -1:
                img = (img.data.cpu() + 1) / 2
            else:
                img = img.data.cpu()
            new_imgs_cat.append(img)
        
        new_imgs_cat = torch.cat(new_imgs_cat, dim=0)
        new_imgs_cat = vutils.make_grid(new_imgs_cat, nrow=b)
        self.writer.add_image(title, new_imgs_cat, step)


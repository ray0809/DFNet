import torch
from torch import nn
from torchvision import models

from model.modules import resize_like


class ReconstructionLoss(nn.L1Loss):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, results, targets):
        loss = 0.
        for i, (res, target) in enumerate(zip(results, targets)):
            loss += self.l1(res, target)
        return loss / len(results)

# RGB with ./255 input
class VGGFeature(nn.Module):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    def __init__(self, norm=False):
        super().__init__()

        vgg16 = models.vgg16(pretrained=True)
        self.normalization = Normalization(self.MEAN, self.STD)
        for para in vgg16.parameters():
            para.requires_grad = False
        self.norm = norm
        self.vgg16_pool_1 = nn.Sequential(*vgg16.features[0:5])
        self.vgg16_pool_2 = nn.Sequential(*vgg16.features[5:10])
        self.vgg16_pool_3 = nn.Sequential(*vgg16.features[10:17])

    def forward(self, x):
        if self.norm:
            x = self.normalization(x)
        pool_1 = self.vgg16_pool_1(x)
        pool_2 = self.vgg16_pool_2(pool_1)
        pool_3 = self.vgg16_pool_3(pool_2)

        return [pool_1, pool_2, pool_3]

# Normalization Layer for VGG
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, input):
        # normalize img
        if self.mean.type() != input.type():
            self.mean = self.mean.to(input)
            self.std = self.std.to(input)
        return (input - self.mean) / self.std


class PerceptualLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1loss = nn.L1Loss()

    def forward(self, vgg_results, vgg_targets):
        loss = 0.
        for i, (vgg_res, vgg_target) in enumerate(
                zip(vgg_results, vgg_targets)):
            for feat_res, feat_target in zip(vgg_res, vgg_target):
                loss += self.l1loss(feat_res, feat_target)
        return loss / len(vgg_results)


class StyleLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1loss = nn.L1Loss()

    def gram(self, feature):
        n, c, h, w = feature.shape
        feature = feature.view(n, c, h * w)
        gram_mat = torch.bmm(feature, torch.transpose(feature, 1, 2))
        return gram_mat / (c * h * w)

    def forward(self, vgg_results, vgg_targets):
        loss = 0.
        for i, (vgg_res, vgg_target) in enumerate(
                zip(vgg_results, vgg_targets)):
            for feat_res, feat_target in zip(vgg_res, vgg_target):
                loss += self.l1loss(
                    self.gram(feat_res), self.gram(feat_target))
        return loss / len(vgg_results)


class TotalVariationLoss(nn.Module):
    
    def __init__(self, c_img=3):
        super().__init__()
        self.c_img = c_img

        kernel = torch.FloatTensor([
            [0, 1, 0],
            [1, -2, 0],
            [0, 0, 0]]).view(1, 1, 3, 3)
        kernel = torch.cat([kernel] * c_img, dim=0)
        self.register_buffer('kernel', kernel)

    def gradient(self, x):
        return nn.functional.conv2d(
            x, self.kernel, stride=1, padding=1, groups=self.c_img)

    def forward(self, results, mask):
        loss = 0.
        for i, res in enumerate(results):
            grad = self.gradient(res) * resize_like(mask, res)
            loss += torch.mean(torch.abs(grad))
        return loss / len(results)


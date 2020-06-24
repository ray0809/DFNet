import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsnooper


def resize_like(x, target, mode='bilinear'):
    if mode == 'bilinear':
        return F.interpolate(x, target.shape[-2:], mode=mode, align_corners=True)
    elif mode == 'nearest':
        return F.interpolate(x, target.shape[-2:], mode=mode)
    else:
        raise('Upsample type error !!')

class GateConv2d(nn.Module):
    def __init__(self, c_in, c_out, k, s, p):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.k = k
        self.s = s

        self.conv = nn.Conv2d(c_in, c_out, k, s, p)
        

class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False  

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
            
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2], input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in
                        
                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize/(self.update_mask + 1e-6)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)


        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)


        if self.return_mask:
            return output, self.update_mask
        else:
            return output

class Self_Attn(nn.Module):
    """ Self attention Layer
    from: https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py#L8
    """
    def __init__(self, in_dim, activation=None):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.reshape(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out
        

class Auto_Attn(nn.Module):
    """ Short+Long attention Layer
    from: https://github.com/lyndonzheng/Pluralistic-Inpainting/blob/master/model/base_function.py#L310
    """

    def __init__(self, input_nc, norm_layer=nn.BatchNorm2d):
        super(Auto_Attn, self).__init__()
        self.input_nc = input_nc

        self.query_conv = nn.Conv2d(input_nc, input_nc // 4, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    # @torchsnooper.snoop()
    def forward(self, x, pre=None, mask=None):
        """
        inputs :
            x : input feature maps( B X C X W X H)
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is Width*Height)
        """
        # print('attention shape', x.size())
        B, C, W, H = x.size()
        proj_query = self.query_conv(x).view(B, -1, W * H)  # B X (N)X C
        proj_key = proj_query  # B X C x (N)

        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = x.view(B, -1, W * H)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, W, H)

        out = self.gamma * out + x

        if type(pre) != type(None):
            # using long distance attention layer to copy information from valid regions
            context_flow = torch.bmm(pre.view(B, -1, W*H), attention.permute(0, 2, 1)).view(B, -1, W, H)
            context_flow = self.alpha * (1-mask) * context_flow + (mask) * pre
            out = torch.cat([out, context_flow], dim=1)

        return out



class AttentionModule(nn.Module):
    
    def __init__(self, patch_size = 3, propagate_size = 3, stride = 1):
        super(AttentionModule, self).__init__()
        self.patch_size = patch_size
        self.propagate_size = propagate_size
        self.stride = stride
        self.prop_kernels = None
        
    def forward(self, foreground, masks):
        ###assume the masked area has value 1
        bz, nc, w, h = foreground.size()
        if masks.size()[2:] != foreground.size()[2:]:
            masks = F.interpolate(masks, foreground.size()[2:])
        background = foreground.clone()
        background = background * masks
        background = F.pad(background, [self.patch_size//2, self.patch_size//2, self.patch_size//2, self.patch_size//2])
        conv_kernels_all = background.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size, self.stride).contiguous().view(bz, nc, -1, self.patch_size, self.patch_size)
        conv_kernels_all = conv_kernels_all.transpose(2, 1)
        output_tensor = []
        for i in range(bz):
            mask = masks[i:i+1]
            feature_map = foreground[i:i+1]
            #form convolutional kernels
            conv_kernels = conv_kernels_all[i] + 0.0000001
            norm_factor = torch.sum(conv_kernels**2, [1, 2, 3], keepdim = True)**0.5
            conv_kernels = conv_kernels/norm_factor
            
            conv_result = F.conv2d(feature_map, conv_kernels, padding = self.patch_size//2)
            if self.propagate_size != 1:
                if self.prop_kernels is None:
                    self.prop_kernels = torch.ones([conv_result.size(1), 1, self.propagate_size, self.propagate_size])
                    self.prop_kernels.requires_grad = False
                    self.prop_kernels = self.prop_kernels.to(foreground)
                conv_result = F.conv2d(conv_result, self.prop_kernels, stride = 1, padding = 1, groups = conv_result.size(1))
            attention_scores = F.softmax(conv_result, dim = 1)
            ##propagate the scores
            recovered_foreground = F.conv_transpose2d(attention_scores, conv_kernels, stride = 1, padding = self.patch_size//2)
            #average the recovered value, at the same time make non-masked area 0
            recovered_foreground = (recovered_foreground * (1 - mask))/(self.patch_size ** 2)
            #recover the image
            final_output = recovered_foreground + feature_map * mask
            output_tensor.append(final_output)
        return torch.cat(output_tensor, dim = 0)        
        
if __name__ == '__main__':
    
    model = Auto_Attn(128)
    model.eval()
    a = torch.randn(4,128,32,32)
    b = torch.randn(4,256,32,32)
    c = torch.randn(4,1,32,32)
    out = model(a,b,c)
    print(out.shape)
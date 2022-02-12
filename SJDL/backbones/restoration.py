import torch
from torch import nn
import torch.nn.functional as F
from functools import reduce

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':  # default: reflect
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Enhance(nn.Module):
    def __init__(self):
        super(Enhance, self).__init__()

        self.relu=nn.LeakyReLU(0.2, inplace=True)

        self.tanh=nn.Tanh()

        self.refine1= nn.Conv2d(6, 20, kernel_size=3,stride=1,padding=1)
        self.refine2= nn.Conv2d(20, 20, kernel_size=3,stride=1,padding=1)

        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)

        self.upsample = F.upsample_nearest

        self.batch1 = nn.InstanceNorm2d(100, affine=True)

    def forward(self, x):
        dehaze = self.relu((self.refine1(x)))
        dehaze = self.relu((self.refine2(dehaze)))
        shape_out = dehaze.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(dehaze, 32)

        x102 = F.avg_pool2d(dehaze, 16)

        x103 = F.avg_pool2d(dehaze, 8)

        x104 = F.avg_pool2d(dehaze, 4)
        x1010 = self.upsample(self.relu(self.conv1010(x101)),size=shape_out)
        x1020 = self.upsample(self.relu(self.conv1020(x102)),size=shape_out)
        x1030 = self.upsample(self.relu(self.conv1030(x103)),size=shape_out)
        x1040 = self.upsample(self.relu(self.conv1040(x104)),size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, dehaze), 1)
        dehaze= self.tanh(self.refine3(dehaze))

        return dehaze
    
class Restoration(nn.Module):
    def __init__(self, norm_layer=nn.InstanceNorm2d, padding_type='reflect'):
        super(Restoration, self).__init__()
        self.inplaces = [1024, 512, 256, 64]

        ### residual blocks
        model_upsample3 = [ResnetBlock(self.inplaces[2], padding_type=padding_type, norm_layer=norm_layer), 
                           nn.Conv2d(self.inplaces[2], self.inplaces[3], kernel_size=3, padding=1), 
                           norm_layer(self.inplaces[3]), nn.ReLU(True)]
        
        model_upsample4 = [ResnetBlock(self.inplaces[3]*2, padding_type=padding_type, norm_layer=norm_layer), 
                           nn.ConvTranspose2d(self.inplaces[3]*2, self.inplaces[3], kernel_size=3, stride=2, padding=1, output_padding=1), 
                           norm_layer(self.inplaces[3]), nn.ReLU(True)]

        model_upsample5 = [ResnetBlock(self.inplaces[3], padding_type=padding_type, norm_layer=norm_layer), 
                           nn.ConvTranspose2d(self.inplaces[3], 32, kernel_size=3, stride=2, padding=1, output_padding=1), 
                           norm_layer(32), nn.ReLU(True)]

        model_dehaze1 = [nn.ReflectionPad2d(3), nn.Conv2d(32, 3, kernel_size=7, padding=0), nn.Tanh()]                  

        self.up_layer3 = nn.Sequential(*model_upsample3)
        self.up_layer4 = nn.Sequential(*model_upsample4)
        self.up_layer5 = nn.Sequential(*model_upsample5)
        self.dehaze = nn.Sequential(*model_dehaze1)
        self.dehaze2 = Enhance()

    def forward(self, x, embedding_features): 
        up3 = self.up_layer3(embedding_features[-1])
        concat_f3 = torch.cat((up3, embedding_features[-2]), 1)
        up4 = self.up_layer4(concat_f3)
        up5 = self.up_layer5(up4)
        
        dehaze_fake = self.dehaze(up5)

        concat_dehaze = torch.cat((x, dehaze_fake), 1)
        dehaze_ehn = self.dehaze2(concat_dehaze)

        return [dehaze_fake, dehaze_ehn], [x, dehaze_fake, dehaze_ehn, up5, up4, up3]

    def load_param(self, model_path):
        param = torch.load(model_path)
        for i in param:
            if 'fc' in i: continue
            if i not in self.state_dict().keys(): continue
            if param[i].shape != self.state_dict()[i].shape: continue
            self.state_dict()[i].copy_(param[i])
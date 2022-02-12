import torch
from torch import nn

from .backbones.resnet import resnet50_ibn_a, resnet101_ibn_a, resnet152_ibn_a
from .backbones.restoration import Restoration
from .backbones.dsc import ChannelAttention, DCNblock

###########################################################################
## weights init
###########################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

###########################################################################
## SJDL c
###########################################################################
class SJDL(nn.Module):
    dcn_planes = 512
    in_planes = 2048 + dcn_planes
    def __init__(self, model_name, num_classes_syn, num_classes_real, last_stride, model_reid_path="", model_rest_path="", fixed=False):
        super(SJDL, self).__init__()
        ########################## ReID ###################################
        # 1. backbone
        if model_name == 'resnet50':
            self.base = resnet50_ibn_a(last_stride, pretrained=True)
        elif model_name == 'resnet101':
            self.base = resnet101_ibn_a(last_stride, pretrained=True) 
        elif model_name == 'resnet152':
            self.base = resnet152_ibn_a(last_stride, pretrained=True)
        # 2. dsc srbnet
        self.DCN = DCNblock(2048, 512, stride=1)
        # 3. embedding, score
        self.id_attention = ChannelAttention(self.in_planes)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes_syn = num_classes_syn
        self.num_classes_real = num_classes_real

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier_syn = nn.Linear(self.in_planes, self.num_classes_syn, bias=False)
        self.classifier_real = nn.Linear(self.in_planes, self.num_classes_real, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier_syn.apply(weights_init_classifier)
        self.classifier_real.apply(weights_init_classifier)

        if model_reid_path != "":
            self.load_param(model_reid_path)
            print("####### Load reid model finish: model=%s"%(model_reid_path))

        ########################## Resotration ###################################
        self.Rest = Restoration(norm_layer=nn.InstanceNorm2d, padding_type='reflect')

        if model_rest_path != "":
            self.load_param(model_rest_path)
            print("####### Load rest model finish: model=%s"%(model_rest_path))
            if fixed:
                for layer in self.Rest.children():
                    print(layer)
                    print("######### layer type=",type(layer),' ####################')
                    for parameter in layer.parameters():
                        parameter.requires_grad = False
                        print(r'% parameter grad = false')
        else:
            self.Rest.apply(weights_init_kaiming) 

    def forward(self, x):
        ####################### reid #################################
        # 1. backbone
        backbone_f, hidden_feature = self.base(x)    # r4, [f1, r1, r2, r3, r4]    [b,2048,8,8]
        # 2. dsc subnet
        dcn_f = self.DCN(hidden_feature[-1])
        # 3. id attention 
        embedding = torch.cat((backbone_f, dcn_f), 1) 
        embedding = self.id_attention(embedding) * embedding
        global_feat = self.gap(embedding)                         # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 4096)
        feat = self.bottleneck(global_feat)                       # normalize for angular softmax
        
        ####################### rest #################################
        generated, hidden_generate = self.Rest(x, hidden_feature[:-3]) # [dehaze_fake, dehaze_ehn],  [x, dehaze_fake, dehaze_ehn, up5, up4, up3]
        
        if self.training:
            cls_score_syn = self.classifier_syn(feat)
            cls_score_real = self.classifier_real(feat)
            return cls_score_syn, cls_score_real, feat, generated                   # global feature for triplet loss
        else:
            return global_feat, generated

    def load_param(self, model_path):
        param = torch.load(model_path)
        for i in param:
            if 'fc' in i: continue
            if i not in self.state_dict().keys(): continue
            if param[i].shape != self.state_dict()[i].shape: continue
            self.state_dict()[i].copy_(param[i])   

def build_SJDL(cfg, num_classes_syn, num_classes_real):
    ################################################################
    # input: foggy_imgs                                        
    # train:  forward: cls_score_syn, cls_score_real, feat, [defoggy_fake, defoggy_ehn] 
    # eval: forward: global_feat, [defoggy_fake, defoggy_ehn]
    ################################################################
    model = SJDL(cfg.MODEL.NAME, num_classes_syn, num_classes_real, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH_reid, cfg.MODEL.PRETRAIN_PATH_res, cfg.MODEL.FIXED_REST)
    print(cfg.MODEL.NAME)
    return model
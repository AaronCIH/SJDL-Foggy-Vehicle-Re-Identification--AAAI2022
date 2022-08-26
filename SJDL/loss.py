import torch
from torch import nn
import torch.nn.functional as F
import numpy as np 

########################################################
## Define loss function forwar
########################################################
def make_loss_for_syn(cfg):
    sampler = cfg.DATALOADER.SAMPLER
    # for ReID
    triplet = TripletLoss(cfg.SOLVER.MARGIN)
    xent = F.cross_entropy
    # for GAN
    mse_loss = MSEloss()
    # forward
    if sampler == 'SJDL':
        print("############### Loss define SYN ##############")
        print("baseline_loss_w = {%f}, rest_loss_w = {%f}"%(cfg.MODEL.SJDL_BASE_W, cfg.MODEL.SJDL_REST_W))
        print("##########################################")
        def loss_func(score, feat, target, gx_fake, gx_enh, gt):          
            # reid
            Lbaseline = xent(score, target) + triplet(feat, target)[0]
            # gan
            fake = gx_fake[0]
            enh = gx_enh[0]
            clear = gt[0]
            Lgan_fake = mse_loss(fake, clear) 
            Lgan_enh = mse_loss(enh, clear) 
            # total loss
            con_loss = (cfg.MODEL.SJDL_BASE_W * Lbaseline) + (cfg.MODEL.SJDL_REST_W * (Lgan_fake+Lgan_enh)/2)
            return con_loss, [Lbaseline, Lgan_fake, Lgan_enh]
        return loss_func   
    else:
        print('expected sampler should be SJDL, ' 'but got {}'.format(cfg.DATALOADER.SAMPLER))

def make_loss_for_real(cfg):
    sampler = cfg.DATALOADER.SAMPLER
    # reid loss
    triplet = TripletLoss(cfg.SOLVER.MARGIN)
    xent = F.cross_entropy
    # unsupervised gan loss
    CE = color_entropy_loss()
    DC = dark_channel_loss()
    TV = total_variation_loss()
    SC = self_constraint_loss()
    # loss weights
    Wce, Wdc, Wtv, Wsc = 10e-1, 10e-6, 10e-6, 10e-3
    # forward
    if sampler == 'SJDL':
        print("############### Loss define REAL ##############")
        print("baseline_loss_w = {%f}, rest_loss_w = {%f}"%(cfg.MODEL.SJDL_BASE_W, cfg.MODEL.SJDL_REST_W))
        print("##########################################")
        def loss_func(score, feat, target, hazy, gx_ehn): 
            # reid 
            Lbaseline = xent(score, target) + triplet(feat, target)[0]
            # unsupervised gan
            haz = hazy[0].unsqueeze(0)
            ehn = gx_ehn[0].unsqueeze(0)
            Lce = CE(ehn)
            Ldc = DC((ehn+1)/2)
            Ltv = TV((ehn+1)/2)
            Lsc = SC(haz, ehn)    
            Lgan = Wce*Lce + Wdc*Ldc + Wtv*Ltv + Wsc*Lsc
            # total loss
            total_loss = (cfg.MODEL.SJDL_BASE_W * Lbaseline) + (cfg.MODEL.SJDL_REST_W * Lgan)
            return total_loss, [Lbaseline, Lce, Ldc, Ltv, Lsc, Lgan]
    else:
        print('expected sampler should be SJDL, ''but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func    

########################################################
## Basic loss function
########################################################
# ReID
class TripletLoss(object):
    """
    Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'.
    """
    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = self._normalize(global_feat, axis=-1)
        dist_mat = self._euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = self._hard_example_mining(dist_mat, labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an

    def _normalize(self, x, axis=-1):
        x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
        return x

    def _euclidean_dist(self, x, y):
        m, n = x.size(0), y.size(0)
        x2 = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        y2 = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = x2 + y2
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

    def _hard_example_mining(self, dist_mat, labels, return_inds=False):
        assert len(dist_mat.size()) == 2
        assert dist_mat.size(0) == dist_mat.size(1)
        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
        # `dist_ap` means distance(anchor, positive)
        dist_ap, relative_p_inds = torch.max(dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
        # `dist_an` means distance(anchor, negative)
        dist_an, relative_n_inds = torch.min(dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
        # shape [N]
        dist_ap = dist_ap.squeeze(1)
        dist_an = dist_an.squeeze(1)
        if return_inds:
            # shape [N, N]
            ind = (labels.new().resize_as_(labels).copy_(torch.arange(0, N).long()).unsqueeze(0).expand(N, N))
            # shape [N, 1]
            p_inds = torch.gather(ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
            n_inds = torch.gather(ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
            # shape [N]
            p_inds = p_inds.squeeze(1)
            n_inds = n_inds.squeeze(1)
            return dist_ap, dist_an, p_inds, n_inds
        return dist_ap, dist_an

# Syn gan loss
class MSEloss(nn.Module):
    def __init__(self, use_gpu=True):
        super(MSEloss, self).__init__()
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.mseloss = nn.MSELoss().cuda()
        else:
            self.mseloss = nn.MSELoss()
    
    def forward(self, gx, gt):
        loss = self.mseloss(gx, gt)
        return loss

# Unsupervised gan loss
class color_entropy_loss(nn.Module):
    def __init__(self):
        super(color_entropy_loss, self).__init__()
    
    def forward(self, x):
        num_pixel = x.size()[2]*x.size()[3]+256
        bins = []
        temp = x*255
        temp = temp.mean(dim=1)
        temp = torch.unsqueeze(temp, -1)
        zeros = torch.zeros_like(temp)
        ones = torch.ones_like(temp)
        for i in range(256):
            a = torch.where(temp<i+1, temp, zeros)
            a = torch.where(a<i, zeros, a)
            a = torch.where(a>=i, ones, a)
            bins.append(a.sum(dim=[1,2])+1)
        hist = torch.cat(bins,-1)
        prob = hist/num_pixel
        loss = (prob * torch.log(prob)).sum(dim=-1)
        return loss.mean() # doesn't need negative

class dark_channel_loss(nn.Module):
    def __init__(self):
        super(dark_channel_loss, self).__init__()
        self.patch_size = 35
    
    def forward(self, img):
        maxpool = nn.MaxPool3d((3, self.patch_size, self.patch_size), stride=1, padding=(0, self.patch_size//2, self.patch_size//2))
        dc = maxpool(1-img[:, None, :, :, :])
        target = torch.FloatTensor(dc.shape).zero_().cuda()
        loss = nn.L1Loss(reduction='sum')(dc, target) / len(img)
        return -loss

class total_variation_loss(nn.Module):
    def __init__(self):
        super(total_variation_loss, self).__init__()

    def forward(self, img):
        hor = self.grad_conv_hor()(img)
        vet = self.grad_conv_vet()(img)
        target = torch.autograd.Variable(torch.FloatTensor(img.shape).zero_().cuda())
        loss_hor = nn.L1Loss(reduction='sum')(hor, target)
        loss_vet = nn.L1Loss(reduction='sum')(vet, target)
        loss = (loss_hor+loss_vet) / len(img)
        return loss

    # horizontal gradient, the input_channel is default to 3
    def grad_conv_hor(self, ):
        grad = nn.Conv2d(3, 3, (1, 3), stride=1, padding=(0, 1))
        weight = np.zeros((3, 3, 1, 3))
        for i in range(3):
            weight[i, i, :, :] = np.array([[-1, 1, 0]])
        weight = torch.FloatTensor(weight).cuda()
        weight = nn.Parameter(weight, requires_grad=False)
        bias = np.array([0, 0, 0])
        bias = torch.FloatTensor(bias).cuda()
        bias = nn.Parameter(bias, requires_grad=False)
        grad.weight = weight
        grad.bias = bias
        return  grad

    # vertical gradient, the input_channel is default to 3
    def grad_conv_vet(self, ):
        grad = nn.Conv2d(3, 3, (3, 1), stride=1, padding=(1, 0))
        weight = np.zeros((3, 3, 3, 1))
        for i in range(3):
            weight[i, i, :, :] = np.array([[-1, 1, 0]]).T
        weight = torch.FloatTensor(weight).cuda()
        weight = nn.Parameter(weight, requires_grad=False)
        bias = np.array([0, 0, 0])
        bias = torch.FloatTensor(bias).cuda()
        bias = nn.Parameter(bias, requires_grad=False)
        grad.weight = weight
        grad.bias = bias
        return  grad

class self_constraint_loss(nn.Module):
    def __init__(self, use_gpu=True):
        super(self_constraint_loss, self).__init__()
        self.use_gpu = use_gpu
    
    def forward(self, img1s, img2s):
        loss_mean = []
        for im1, im2 in zip(img1s, img2s):
            im1 = F.interpolate(im1, size=255)
            im2 = F.interpolate(im2, size=255)
            fft1 = torch.fft.fft2(im1, dim=(-2, -1), norm=None) # fft tranform, with complex
            fft1_2dim = torch.stack((fft1.real, fft1.imag), -1)
            fft2 = torch.fft.fft2(im2, dim=(-2, -1), norm=None) # fft tranform, with complex
            fft2_2dim = torch.stack((fft2.real, fft2.imag), -1)
            inner_product = (fft1_2dim * fft2_2dim).sum(dim=-1)
            norm1 = (fft1_2dim.pow(2).sum(dim=-1)+1e-20).pow(0.5)
            norm2 = (fft2_2dim.pow(2).sum(dim=-1)+1e-20).pow(0.5)
            cos = inner_product / (norm1*norm2 + 1e-20)
            loss_mean.append(-1.0*cos.mean())
        loss_mean = torch.tensor(loss_mean)
        loss = torch.mean(loss_mean)
        return loss


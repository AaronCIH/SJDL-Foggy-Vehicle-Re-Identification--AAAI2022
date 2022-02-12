##########################################################################
## Thirty-Sixth AAAI Conference on Artificial Intelligence (AAAI-22)
## Creator: Cihsiang
## Email: f09921058@ntu.edu.tw 
##########################################################################


import os
import os.path as osp

# PyTorch as the main lib for neural network
import torch
torch.backends.cudnn.benchmark = True
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torchvision as tv
import numpy as np

# Use visdom for moniting the training process
import visdom
from utils import dehaze_plot

from utils import rank_list_to_im

# Use yacs for training config management
# argparse for overwrite



# import losses and model
from losses import make_loss_for_dhvreid, make_loss_for_real, make_loss_for_sup
from losses import MSEloss
from model_c2_dc import build_dhverid_dc, convert_model
from dhvreid_c2dc_trainer import DhvreidTrainer

# dataset
from dataset_dhvreid import make_dhvreid_dataloader

# for traing
from optim import make_optimizer, WarmupMultiStepLR
from evaluate import eval_func, euclidean_dist, re_rank, eval_dh
from tqdm import tqdm
import shutil




   

def test(args):

    num_gpus = torch.cuda.device_count()
    logger = setup_logger('dhvreid_model', output_dir, 0)
    logger.info('Using {} GPUS'.format(num_gpus))
    logger.info(args)
    logger.info('Running with config:\n{}'.format(cfg))

    logger.info("##############################################################")
    logger.info("# % TRAIN .....  ")
    logger.info("# Backbone Model is {%s}" %(cfg.MODEL.NAME))
    logger.info("# Dataset is {%s}" %(cfg.DATASETS.NAMES))
    logger.info("# pretrain_model = {%s} #" %(cfg.MODEL.PRETRAIN_PATH))
    logger.info("# pretrain_baseline = {%s} #" %(cfg.MODEL.PRETRAIN_PATH_reid))
    logger.info("# pretrain_restoration = {%s} #" %(cfg.MODEL.PRETRAIN_PATH_res))
    logger.info("# data_path = {%s} #" %(cfg.DATASETS.DATA_PATH))
    logger.info("# output_dir = {%s} #" %(cfg.OUTPUT_DIR))
    logger.info("# dhvreid_baseline_W = {%s} #" %(cfg.MODEL.DHVREID_BASE_W))
    logger.info("# dhvreid_restoration_W = {%s} #" %(cfg.MODEL.DHVREID_REST_W))
    logger.info("##############################################################")
    
    print("#In[0]: make_dataloader--------------------------------")
    train_dl, real_dl, val_dl, num_query, num_classes_syn, num_classes_real = make_dhvreid_dataloader(cfg, num_gpus)

    print("#In[1]: build_model--------------------------------")
    model = build_dhverid_dc(cfg, num_classes_syn, num_classes_real)

    if cfg.TEST.MULTI_GPU:
        model = nn.DataParallel(model)
        model = convert_model(model)
        logger.info('Use multi gpu to inference')

    para_dict = torch.load(cfg.TEST.WEIGHT)
    model.load_state_dict(para_dict)
    model.cuda()
    model.eval()

    feats, pids, camids = [], [], []
    ssim_score_fake, psnr_score_fake, mse_fake = [], [], []
    ssim_score_ehn, psnr_score_ehn, mse_ehn = [], [], []

    if cfg.TEST.VIS:
        sample_result_path = os.path.join(cfg.OUTPUT_DIR, "sample_result/")
        if not os.path.isdir(sample_result_path):
            os.makedirs(sample_result_path)

    mse_loss = MSEloss()
    with torch.no_grad():
        for ct, batch in enumerate(tqdm(val_dl, total=len(val_dl), leave=False)):
            hazy_img, gt_img, pid, camid = batch['label'], batch['image'], batch['pids'], batch['camids']
            hazy_paths = batch['hazy_path']
            hazy_img, gt_img = hazy_img.cuda(), gt_img.cuda()
            feat, generated, hidden_f, hidden_g = model(hazy_img)

            ### % reid.....
            feat = feat.detach().cpu()
            feats.append(feat)
            pids.append(pid)
            camids.append(camid)

            ### % restoration.....
            gx_fake = generated[0].cuda()
            gx_ehn = generated[1].cuda()
            if cfg.DATALOADER.NORMALZIE:
                gx_fake = gx_fake*0.5 + 0.5
                gx_ehn = gx_ehn*0.5 + 0.5
                gt_img = gt_img*0.5 + 0.5
            
            if int(ct % 100) == 0:
               if cfg.TEST.VIS:
                   sample_idx = 0
                   sample_result_path = os.path.join(cfg.OUTPUT_DIR, "sample_result/")
                   name = hazy_paths[sample_idx].split("/")[-1]
                   save_full_path = sample_result_path + name
                   hazy_ = hazy_img.cpu().numpy()[sample_idx].transpose((1, 2, 0))
                   pred_fake = gx_fake.cpu().numpy()[sample_idx].transpose((1,2,0))
                   pred_ehn = gx_ehn.cpu().numpy()[sample_idx].transpose((1,2,0))
                   gt_ = gt_img.cpu().numpy()[sample_idx].transpose((1, 2, 0))
                   dehaze_plot(hazy_, pred_fake, pred_ehn, gt_, save_full_path)

            loss_fake = mse_loss(gx_fake, gt_img)
            loss_ehn = mse_loss(gx_ehn, gt_img)

            ssim_bts_fake, psnr_bts_fake = eval_dh(gx_fake, gt_img)
            ssim_bts_ehn, psnr_bts_ehn = eval_dh(gx_ehn, gt_img)

            ssim_score_fake.append(float(ssim_bts_fake)), psnr_score_fake.append(float(psnr_bts_fake)), mse_fake.append(float(loss_fake.cpu().item()))
            ssim_score_ehn.append(float(ssim_bts_ehn)), psnr_score_ehn.append(float(psnr_bts_ehn)), mse_ehn.append(float(loss_ehn.cpu().item()))

        ### % reid.....
        feats = torch.cat(feats, dim=0)
        pids = torch.cat(pids, dim=0)
        camids = torch.cat(camids, dim=0)

        query_feat = feats[:num_query]
        query_pid = pids[:num_query]
        query_camid = camids[:num_query]

        gallery_feat = feats[num_query:]
        gallery_pid = pids[num_query:]
        gallery_camid = camids[num_query:]

        distmat = euclidean_dist(query_feat, gallery_feat)

        cmc, mAP, _ = eval_func(distmat.numpy(), query_pid.numpy(), gallery_pid.numpy(), 
                                query_camid.numpy(), gallery_camid.numpy(),
                                use_cython=cfg.SOLVER.CYTHON)

        ### % dehaze.....
        ssim_score_fake, psnr_score_fake, mse_fake = np.array(ssim_score_fake).mean(),  np.array(psnr_score_fake).mean(),  np.array(mse_fake).mean()
        ssim_score_ehn, psnr_score_ehn, mse_ehn = np.array(ssim_score_ehn).mean(),  np.array(psnr_score_ehn).mean(),  np.array(mse_ehn).mean()

        logger.info('Validation Result:')
        for r in cfg.TEST.CMC:
            logger.info('CMC Rank-{}: {:.2%}'.format(r, cmc[r-1]))
        logger.info('mAP: {:.2%}'.format(mAP))
        logger.info('SSIM_fake: {:.3f}, SSIM_ehn: {:.3f}'.format(ssim_score_fake, ssim_score_ehn))
        logger.info('PSNR_fake: {:.3f}, PSNR_ehn: {:.3f}'.format(psnr_score_fake, psnr_score_ehn))
        logger.info('MSE_fake: {:.4f}, MSE_ehn: {:.4f}'.format(mse_fake, mse_ehn))
        logger.info('-' * 20)

        distmat = re_rank(query_feat, gallery_feat)
        cmc, mAP, all_AP = eval_func(distmat, query_pid.numpy(), gallery_pid.numpy(),
                            query_camid.numpy(), gallery_camid.numpy(),
                            use_cython=True)
        logger.info('Rernak Validation Result:')
        for r in cfg.TEST.CMC:
            logger.info('CMC Rank-{}: {:.2%}'.format(r, cmc[r-1]))
        logger.info('mAP: {:.2%}'.format(mAP))
        logger.info('-' * 20)


if __name__ == '__main__':
    main()
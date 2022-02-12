##########################################################################
## Thirty-Sixth AAAI Conference on Artificial Intelligence (AAAI-22)
## Creator: Cihsiang
## Email: f09921058@ntu.edu.tw 
## SJDL inference
##########################################################################
import os
import os.path as osp
import argparse
import shutil
from config import cfg
import logging 
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from utils import setup_logger, Defoggy_plot
from dataset import make_FVRID_dl
from model import build_SJDL

from collections import defaultdict
from sync_bn import convert_model
from evaluate import eval_defoggy, euclidean_dist, eval_func

def test():
    # Step1. config setting
    parser = argparse.ArgumentParser(description="ReID testing")
    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    # Step2. output setting
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not osp.exists(output_dir):
        os.makedirs(output_dir)
    shutil.copy(args.config_file, cfg.OUTPUT_DIR)  # copy config file
    if not os.path.exists(os.path.join(output_dir, "checkpoint/")):
        os.makedirs(os.path.join(output_dir, "checkpoint/"))
    if cfg.TEST.VIS:    # save folder
        if not os.path.exists(os.path.join(output_dir, "sample_result/")):
            os.makedirs(os.path.join(output_dir, "sample_result/"))

    # Step3. logging 
    num_gpus = torch.cuda.device_count()
    logger = setup_logger('SJDL', output_dir, 0)
    logger.info('Using {} GPUS'.format(num_gpus))
    logger.info(args)
    logger.info('Running with config:\n{}'.format(cfg))
    logger.info("##############################################################")
    logger.info("# % TEST .....  ")
    logger.info("# Backbone Model is {%s}" %(cfg.MODEL.NAME))
    logger.info("# Dataset is {%s}" %(cfg.DATASETS.NAMES))
    logger.info("# TEST_WEIGHT = {%s} #" %(cfg.TEST.WEIGHT))
    logger.info("# DATA_PATH = {%s} #" %(cfg.DATASETS.DATA_PATH))
    logger.info("# TRAIN_PATH = {%s} #" %(cfg.DATASETS.TRAIN_PATH))
    logger.info("# TRAIN_GT_PATH = {%s} #" %(cfg.DATASETS.TRAIN_GT_PATH))
    logger.info("# QUERY_PATH = {%s} #" %(cfg.DATASETS.QUERY_PATH))
    logger.info("# QUERY_GT_PATH = {%s} #" %(cfg.DATASETS.QUERY_GT_PATH))
    logger.info("# GALLERY_PATH = {%s} #" %(cfg.DATASETS.GALLERY_PATH))
    logger.info("# GALLERY_GT_PATH = {%s} #" %(cfg.DATASETS.GALLERY_GT_PATH))
    logger.info("# REAL_FOGGY_PATH = {%s} #" %(cfg.DATASETS.REAL_FOGGY_PATH))
    logger.info("# OUTPUT_DIR = {%s} #" %(cfg.OUTPUT_DIR))
    logger.info("##############################################################")

    # Step4. Create FVRID dataloader
    print("#In[0]: make_dataloader--------------------------------")
    train_dl, real_dl, val_dl, num_query, num_classes_syn, num_classes_real = make_FVRID_dl(cfg, num_gpus)

    # Step5. Build SJDL model
    print("#In[1]: build_model--------------------------------")
    print("num_classes_syn = ", num_classes_syn)
    print("num_classes_real = ", num_classes_real)
    SJDL = build_SJDL(cfg, num_classes_syn, num_classes_real)  
    if cfg.TEST.MULTI_GPU:
        SJDL = nn.DataParallel(SJDL)
        SJDL = convert_model(SJDL)
        logger.info('Use multi gpu to inference')

    para_dict = torch.load(cfg.TEST.WEIGHT)
    SJDL.load_state_dict(para_dict)
    SJDL.cuda()
    SJDL.eval()

    # Step6. Logger
    feats, pids, camids = [], [], []
    SSIM_fake, PSNR_fake = [], []
    SSIM_enh, PSNR_enh = [], []
      
    # % Val forward..
    ####################################################################################################
    ## val_loader: getitem[Index]: {"foggy", "gt", "foggy_paths", "gt_paths", "pids", "camids"}
    ## -eval: forward: global_feat, [defoggy_fake, defoggy_ehn]
    ####################################################################################################
    with torch.no_grad():
        for ct, batch in enumerate(tqdm(val_dl, total=len(val_dl), leave=False)):
            foggy_img, gt_img, pid, camid, foggy_paths = batch['foggy'], batch['gt'], batch['pids'], batch['camids'], batch['foggy_paths']
            foggy_img, gt_img = foggy_img.cuda(), gt_img.cuda()
            feat, generated = SJDL(foggy_img)  # global_feat, [defoggy_fake, defoggy_ehn]

            ### % reid.....
            feat = feat.detach().cpu()
            feats.append(feat)
            pids.append(pid)
            camids.append(camid)

            ### % restoration.....
            gx_fake = generated[0].cuda()
            gx_enh = generated[1].cuda()
            if cfg.DATALOADER.NORMALZIE:
                gx_fake = gx_fake*0.5 + 0.5
                gx_enh = gx_enh*0.5 + 0.5
                gt_img = gt_img*0.5 + 0.5

            if int(ct % 100) == 0:
                if cfg.TEST.VIS:
                    sample_idx = 0
                    sample_result_path = os.path.join(output_dir, "sample_result_for_val/")
                    name = foggy_paths[sample_idx].split("/")[-1]
                    save_full_path = sample_result_path + name
                    foggy_ = foggy_img.cpu().numpy()[sample_idx].transpose((1, 2, 0))
                    pred_fake = gx_fake.cpu().numpy()[sample_idx].transpose((1,2,0))
                    pred_ehn = gx_enh.cpu().numpy()[sample_idx].transpose((1,2,0))
                    gt_ = gt_img.cpu().numpy()[sample_idx].transpose((1, 2, 0))
                    Defoggy_plot(foggy_, pred_fake, pred_ehn, gt_, save_full_path)

            ssim_bts_fake, psnr_bts_fake = eval_defoggy(gx_fake, gt_img)
            ssim_bts_ehn, psnr_bts_ehn = eval_defoggy(gx_enh, gt_img)

            SSIM_fake.append(float(ssim_bts_fake)), PSNR_fake.append(float(psnr_bts_fake))
            SSIM_enh.append(float(ssim_bts_ehn)), PSNR_enh.append(float(psnr_bts_ehn))

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
                                query_camid.numpy(), gallery_camid.numpy())

        ### % dehaze.....
        SSIM_fake, PSNR_fake = np.array(SSIM_fake).mean(),  np.array(PSNR_fake).mean()
        SSIM_enh, PSNR_enh = np.array(SSIM_enh).mean(),  np.array(PSNR_enh).mean()

        logger.info('Validation Result:')
        for r in cfg.TEST.CMC:
            logger.info('CMC Rank-{}: {:.4%}'.format(r, cmc[r-1]))
        logger.info('mAP: {:.4%}'.format(mAP))
        logger.info('SSIM_fake: {:.3f}, SSIM_ehn: {:.3f}'.format(SSIM_fake, SSIM_enh))
        logger.info('PSNR_fake: {:.3f}, PSNR_ehn: {:.3f}'.format(PSNR_fake, PSNR_enh))
        logger.info('-' * 20)
   
if __name__ == '__main__':
    test()
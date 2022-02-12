##########################################################################
## Thirty-Sixth AAAI Conference on Artificial Intelligence (AAAI-22)
## Creator: Cihsiang
## Email: f09921058@ntu.edu.tw 
## SJDL inference
## cmd: python inference.py 
##########################################################################
import os, sys
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
    parser.add_argument('-c', '--config_file', type=str,
                        help='the path to the training config')
    parser.add_argument('-p', '--pre_train', action='store_true',
                        default=False, help='Model test')
    parser.add_argument('-t', '--test', action='store_true',
                        default=False, help='Model test')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('opts', help='overwriting the training config' 
                        'from commandline', default=None,
                        nargs=argparse.REMAINDER)
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
    _, _, val_dl, num_query, _, _ = make_FVRID_dl(cfg, num_gpus)

    # Step5. Build SJDL model
    print("#In[1]: build_model--------------------------------")
    SJDL = build_SJDL(cfg, _, _, mode='Test')  
    if cfg.TEST.MULTI_GPU:
        SJDL = nn.DataParallel(SJDL)
        SJDL = convert_model(SJDL)
        logger.info('Use multi gpu to inference')

    model_dict = SJDL.state_dict()
    para_dict = torch.load(cfg.TEST.WEIGHT)
    para_dict = {k: v for k, v in para_dict.items() if k in model_dict}
    model_dict.update(para_dict)
    SJDL.load_state_dict(model_dict)
    SJDL.cuda()
    SJDL.eval()

    # Step6. Logger
    feats, pids, camids = [], [], []
    # % Val forward..
    ####################################################################################################
    ## val_loader: getitem[Index]: {"foggy", "gt", "foggy_paths", "gt_paths", "pids", "camids"}
    ## -eval: forward: global_feat, [defoggy_fake, defoggy_ehn]
    ####################################################################################################
    with torch.no_grad():
        for ct, batch in enumerate(tqdm(val_dl, total=len(val_dl), leave=False)):
            foggy_img, gt_img, pid, camid = batch['foggy'], batch['gt'], batch['pids'], batch['camids']
            foggy_img, gt_img = foggy_img.cuda(), gt_img.cuda()
            feat = SJDL(foggy_img)  # global_feat

            ### % reid.....
            feat = feat.detach().cpu()
            feats.append(feat)
            pids.append(pid)
            camids.append(camid)

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

 
        logger.info('Validation Result:')
        for r in cfg.TEST.CMC:
            logger.info('CMC Rank-{}: {:.4%}'.format(r, cmc[r-1]))
        logger.info('mAP: {:.4%}'.format(mAP))
        logger.info('-' * 20)
   
if __name__ == '__main__':
    test()
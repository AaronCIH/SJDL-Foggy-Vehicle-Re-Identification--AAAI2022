##########################################################################
## Thirty-Sixth AAAI Conference on Artificial Intelligence (AAAI-22)
## Creator: Cihsiang
## Email: f09921058@ntu.edu.tw 
## SJDL trainer
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
from utils import setup_logger, AvgerageMeter, Defoggy_plot
from dataset import make_FVRID_dl
from model import build_SJDL
from loss import make_loss_for_syn, make_loss_for_real

from collections import defaultdict
from torch.cuda import amp
from sync_bn import convert_model
from optim import make_optimizer, WarmupMultiStepLR
from evaluate import eval_defoggy, euclidean_dist, eval_func

def train():
    # Step1. config setting
    parser = argparse.ArgumentParser(description="ReID training")
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
    if cfg.TEST.VIS:    # save folder
        if not os.path.exists(os.path.join(output_dir, "sample_result_for_train/")):
            os.makedirs(os.path.join(output_dir, "sample_result_for_train/"))
        if not os.path.exists(os.path.join(output_dir, "sample_result_for_real/")):
            os.makedirs(os.path.join(output_dir, "sample_result_for_real/"))
        if not os.path.exists(os.path.join(output_dir, "sample_result_for_val/")):
            os.makedirs(os.path.join(output_dir, "sample_result_for_val/"))

    # Step3. logging 
    num_gpus = torch.cuda.device_count()
    logger = setup_logger('SJDL', output_dir, 0)
    logger.info('Using {} GPUS'.format(num_gpus))
    logger.info(args)
    logger.info('Running with config:\n{}'.format(cfg))
    logger.info("##############################################################")
    logger.info("# % TRAIN .....  ")
    logger.info("# Backbone Model is {%s}" %(cfg.MODEL.NAME))
    logger.info("# SAMPLER is {%s}" %(cfg.DATALOADER.SAMPLER))
    logger.info("# Dataset is {%s}" %(cfg.DATASETS.NAMES))
    logger.info("# pretrain_model = {%s} #" %(cfg.MODEL.PRETRAIN_PATH))
    logger.info("# pretrain_baseline = {%s} #" %(cfg.MODEL.PRETRAIN_PATH_reid))
    logger.info("# pretrain_restoration = {%s} #" %(cfg.MODEL.PRETRAIN_PATH_res))
    logger.info("# data_path = {%s} #" %(cfg.DATASETS.DATA_PATH))
    logger.info("# TRAIN_PATH = {%s} #" %(cfg.DATASETS.TRAIN_PATH))
    logger.info("# TRAIN_GT_PATH = {%s} #" %(cfg.DATASETS.TRAIN_GT_PATH))
    logger.info("# QUERY_PATH = {%s} #" %(cfg.DATASETS.QUERY_PATH))
    logger.info("# QUERY_GT_PATH = {%s} #" %(cfg.DATASETS.QUERY_GT_PATH))
    logger.info("# GALLERY_PATH = {%s} #" %(cfg.DATASETS.GALLERY_PATH))
    logger.info("# GALLERY_GT_PATH = {%s} #" %(cfg.DATASETS.GALLERY_GT_PATH))
    logger.info("# REAL_FOGGY_PATH = {%s} #" %(cfg.DATASETS.REAL_FOGGY_PATH))
    logger.info("# OUTPUT_DIR = {%s} #" %(cfg.OUTPUT_DIR))
    logger.info("# SJDL_BASE_W = {%s} #" %(cfg.MODEL.SJDL_BASE_W))
    logger.info("# SJDL_REST_W = {%s} #" %(cfg.MODEL.SJDL_REST_W))
    logger.info("##############################################################")

    # Step4. Create FVRID dataloader
    #####################################################################
    ## train_loader: getitem[Index]: {"foggy", "gt", "foggy_paths", "gt_paths", "pids", "camids"}
    ## real_loader: getitem[Index]: {"images", "paths", "pids", "camids"}
    ## val_loader: getitem[Index]: {"foggy", "gt", "foggy_paths", "gt_paths", "pids", "camids"}
    #####################################################################
    print("#In[0]: make_dataloader--------------------------------")
    train_dl, real_dl, val_dl, num_query, num_classes_syn, num_classes_real = make_FVRID_dl(cfg, num_gpus)

    # Step5. Build SJDL model
    ################################################################
    # input: foggy_imgs                                        
    # -train: forward: cls_score_syn, cls_score_real, feat, [defoggy_fake, defoggy_ehn] 
    # -eval: forward: global_feat, [defoggy_fake, defoggy_ehn]
    ################################################################
    print("#In[1]: build_model--------------------------------")
    print("num_classes_syn = ", num_classes_syn)
    print("num_classes_real = ", num_classes_real)
    SJDL = build_SJDL(cfg, num_classes_syn, num_classes_real, mode='Train')  

    # Step6. Define Loss function
    ####################################################################
    ## sampler == 'SJDL',       
    ## SYN : input: (score, feat, target, gx_fake, gx_enh, gt) ,  output: con_loss, [Lbaseline, Lgan_fake, Lgan_enh]
    ## REAL : input: (score, feat, target, hazy, gx_ehn) ,  output: total_loss, [Lbaseline, Lce, Ldc, Ltv, Lsc, Lgan]
    ####################################################################
    print("#In[2]: make_loss--------------------------------")
    loss_for_syn = make_loss_for_syn(cfg)
    loss_for_real = make_loss_for_real(cfg)

    # Step7. Training
    print("#In[3]: Start training--------------------------------")
    trainer = SJDLTrainer(cfg, SJDL, train_dl, val_dl, real_dl,
                             loss_for_syn, loss_for_real, num_query, num_gpus)

    trainer._make_new_real_batch() 
    for epoch in range(trainer.epochs):
        print("----------- Epoch [%d/%d] ----------- "%(epoch,trainer.epochs))
        for ct, train_batch in enumerate(trainer.train_dl):
            print('step: %d'%(ct), end='\r')
            trainer.step(train_batch)
            trainer.handle_new_batch()
        trainer.handle_new_epoch()    

class SJDLTrainer(object):
    def __init__(self, cfg, model, train_dl, val_dl, real_dl,
                 loss_for_syn, loss_for_real, num_query, num_gpus):
        self.cfg = cfg
        self.std = self.cfg.INPUT.PIXEL_STD
        self.mean = self.cfg.INPUT.PIXEL_MEAN
        self.model = model
        self.train_dl = train_dl
        self.real_dl = real_dl
        self.val_dl = val_dl
        self.len_batchs = len(train_dl)

        self.loss_for_syn = loss_for_syn
        self.loss_for_real = loss_for_real
        self.num_query = num_query
        
        self.best_mAP = 0
        self.best_SSIM = 0
        self.best_epoch = []
        self.loss_avg = AvgerageMeter()
        self.loss_syn_avg = AvgerageMeter()
        self.loss_real_avg = AvgerageMeter()
        self.real_acc_avg = AvgerageMeter()
        self.acc_avg = AvgerageMeter()
        self.Epoch_cnt = 1
        self.Batch_cnt = 0

        self.logger = logging.getLogger('SJDL.train')
        self.log_period = cfg.SOLVER.LOG_PERIOD
        self.eval_period = cfg.SOLVER.EVAL_PERIOD
        self.output_dir = cfg.OUTPUT_DIR
        self.device = cfg.MODEL.DEVICE
        self.epochs = cfg.SOLVER.MAX_EPOCHS

        self.real_batch_index = []
        self.real_data_dict = defaultdict(list)
        self.real_pids = []
        self._make_real_dict_init()
        self.real_step = 0
        self.real_acc_avg = AvgerageMeter()

        if self.cfg.MODEL.TENSORBOARDX:
            print("############## create tensorboardx ##################")
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'Log_SJDL'))
            self.writer_graph = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'Log_SJDL/Model'))
            # self.writer_graph.add_graph(self.model, torch.FloatTensor(np.random.rand(8, 3, 256, 256)))

        ######################################################
        ## 根據GPU 數量 設定pipeline                         ##
        ######################################################
        if num_gpus > 1:
            # convert to use sync_bn
            self.logger.info('More than one gpu used, convert model to use SyncBN.')
            if cfg.SOLVER.FP16:
                self.logger.info('Using apex to perform SyncBN and FP16 training')
                torch.distributed.init_process_group(backend='nccl', 
                                                     init_method='env://')
                self.model = apex.parallel.convert_syncbn_model(self.model)
            else:
                # Multi-GPU model without FP16
                self.model = nn.DataParallel(self.model)
                self.model = convert_model(self.model)
                self.model.cuda()
                self.logger.info('Using pytorch SyncBN implementation')

                self.optim = make_optimizer(cfg, self.model, num_gpus)
                self.scheduler = WarmupMultiStepLR(self.optim, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                                    cfg.SOLVER.WARMUP_FACTOR, cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
                self.scheduler.step()
                self.mix_precision = False
                self.logger.info('Trainer Built')
                return
        else:
            # Single GPU model
            self.model.cuda()
            self.optim = make_optimizer(cfg, self.model, num_gpus)
            self.scheduler = WarmupMultiStepLR(self.optim, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                                cfg.SOLVER.WARMUP_FACTOR, cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
            self.scheduler.step()
            self.mix_precision = False
            if cfg.SOLVER.FP16:
                # Single model using FP16
                self.model, self.optim = amp.initialize(self.model, self.optim,
                                                        opt_level='O1')
                self.mix_precision = True
                self.logger.info('Using fp16 training')
            self.logger.info('Trainer Built')
            return

    def step(self, train_batch):
        self.model.train()
        self.optim.zero_grad()

        # % Supervised ...
        #################################################################################
        ## train_loader: getitem[Index]: {"foggy", "gt", "foggy_paths", "gt_paths", "pids", "camids"}
        ## model: getitem[foggy_imgs]: [cls_score_syn, cls_score_real, feat, [defoggy_fake, defoggy_ehn] ]        
        #################################################################################
        syn_foggy, syn_gt, syn_pid, syn_foggy_pths = train_batch['foggy'], train_batch['gt'], train_batch['pids'], train_batch['foggy_paths']
        syn_foggy, syn_gt, syn_pid = syn_foggy.cuda(), syn_gt.cuda(), syn_pid.cuda()

        syn_score, _, syn_feat, syn_generated = self.model(syn_foggy)  # cls_score_syn, cls_score_real, feat, [defoggy_fake, defoggy_ehn] 
        syn_gx_fake = syn_generated[0].cuda()
        syn_gx_enh = syn_generated[1].cuda()

        if self.cfg.DATALOADER.NORMALZIE:
            syn_gx_fake = syn_gx_fake*0.5 + 0.5
            syn_gx_enh = syn_gx_enh*0.5 + 0.5
            syn_gt = syn_gt*0.5 + 0.5
            syn_foggy = syn_foggy*0.5 + 0.5

        # SYN : input: (score, feat, target, gx_fake, gx_enh, gt) ,  output: con_loss, [Lbaseline, Lgan_fake, Lgan_enh]
        Lsyn, Lsyn_list = self.loss_for_syn(syn_score, syn_feat, syn_pid, syn_gx_fake, syn_gx_enh, syn_gt) 

        # % Unsupervised ...
        #################################################################################
        ## real_loader: getitem[Index]: {"images", "paths", "pids", "camids"}
        ## model: getitem[foggy_imgs]: [cls_score_syn, cls_score_real, feat, [defoggy_fake, defoggy_ehn]]        
        #################################################################################       
        Wsyn, Wreal = 1, 1
        if self.Batch_cnt % 5 == 0:
            real_batch = self.real_dl.__getitem__(self.real_batch_index[self.real_step])
            real_foggy, real_pid, real_pths = real_batch['images'], real_batch['pids'], real_batch['paths']
            real_foggy, real_pid = real_foggy.cuda(), real_pid.cuda()
            _, real_score, real_feat, real_generated = self.model(real_foggy) # cls_score_syn, cls_score_real, feat, [defoggy_fake, defoggy_ehn] 
            real_gx_fake = real_generated[0].cuda()
            real_gx_enh = real_generated[1].cuda()

            if self.cfg.DATALOADER.NORMALZIE:
                real_gx_fake = real_gx_fake*0.5 + 0.5
                real_gx_enh = real_gx_enh*0.5 + 0.5
                real_foggy = real_foggy*0.5 + 0.5
        
            # REAL : input: (score, feat, target, hazy, gx_ehn) ,  output: total_loss, [Lbaseline, Lce, Ldc, Ltv, Lsc, Lgan]
            Lreal, Lreal_list = self.loss_for_real(real_score, real_feat, real_pid, real_foggy, real_gx_enh) 
            Ltotal = Wsyn*Lsyn + Wreal*Lreal
        else:
            Ltotal = Lsyn

        # % Optimization & Backward()...
        if self.mix_precision:
            with amp.scale_loss(Ltotal, self.optim) as scaled_loss:
                scaled_loss.backward()
        else:
            Ltotal.backward()
        self.optim.step()

        # % Evaluate...
        #################################################################################
        ## acc = mean(pid==predict)
        ## eval_defoggy: input: (gxs, gts, maxi=1.0), output: ssim_score, psnr_score
        #################################################################################       
        syn_acc = (syn_score.max(1)[1] == syn_pid).float().mean() # reid
        syn_ssim_fake, syn_psnr_fake = eval_defoggy(syn_gx_fake, syn_gt)
        syn_ssim_enh, syn_psnr_enh = eval_defoggy(syn_gx_enh, syn_gt)

        self.loss_avg.update(Ltotal.cpu().item())
        self.loss_syn_avg.update(Lsyn.cpu().item())
        self.acc_avg.update(syn_acc)

        # % eval for real .... (acc is fault)
        if self.Batch_cnt % 5 == 0:
            real_acc = (real_score.max(1)[1] == real_pid).float().mean() # reid
            self.real_acc_avg.update(real_acc)
            self.loss_real_avg.update(Lreal.cpu().item())

        # % record ....
        if self.Batch_cnt % self.cfg.SOLVER.LOG_PERIOD == 0:
            self.logger.info('% Record... Epoch[{}] Iteration[{}/{}] : Base Lr: {:.2e}, Ltotal(n/a):{:.4e}/{:.4e}' 
                             .format(self.Epoch_cnt, self.Batch_cnt, len(self.train_dl), self.scheduler.get_lr()[0], Ltotal.cpu().item(), self.loss_avg.avg))
            self.logger.info('--SYN: Lsyn(n/a):{:.4f}/{:.4f}, Lbaseline:{:.4f}, Lgan_fake:{:.4f}, Lgan_enh:{:.4f},'
                             'SSIM_fake: {:.3f}, PSNR_fake: {:.3f}, SSIM_ehn: {:.3f}, PSNR_ehn: {:.3f}, Acc(n/a): {:.3f}'
                             .format(Lsyn.cpu().item(), self.loss_avg.avg, Lsyn_list[0].cpu().item(), Lsyn_list[1].cpu().item(), Lsyn_list[2].cpu().item(),
                                     syn_ssim_fake, syn_psnr_fake, syn_ssim_enh, syn_psnr_enh, syn_acc, self.acc_avg.avg))
            if self.Batch_cnt % 5 == 0:
                self.logger.info('--REAL: Lreal(n/a):{:.4f}/{:.4f}, Lbaseline:{:.4f}, Lce:{:.4f}, Ldc:{:.4f}, Ltv:{:.4f}, Lsc:{:.4f}, Acc(n/a):{:.3f}/{:.3f}'
                                .format(Lreal.cpu().item(), self.loss_real_avg.avg, Lreal_list[0].cpu().item(), Lreal_list[1].cpu().item(), Lreal_list[2].cpu().item(), 
                                        Lreal_list[3].cpu().item(), Lreal_list[4].cpu().item(), real_acc, self.real_acc_avg.avg))

            if self.cfg.MODEL.TENSORBOARDX:
                self.writer.add_scalars("TRAIN/Loss",{"Lbaseline":Lsyn_list[0], "Lgan_fake":Lsyn_list[1], "Lreal":Lsyn_list[2]}, (self.Epoch_cnt* self.len_batchs) + self.Batch_cnt)
                self.writer.add_scalars("TRAIN/SSIM",{"SSIM_fake":syn_ssim_fake, "SSIM_enh":syn_ssim_enh,}, (self.Epoch_cnt* self.len_batchs) + self.Batch_cnt)
                self.writer.add_scalars("TRAIN/PSNR",{"PSNR_fake":syn_psnr_fake, "PSNR_enh":syn_psnr_enh}, (self.Epoch_cnt* self.len_batchs) + self.Batch_cnt)
                self.writer.add_scalars("TRAIN/ACC",{"ACC":syn_acc}, (self.Epoch_cnt* self.len_batchs) + self.Batch_cnt)
                self.writer.add_scalars("TRAIN/LR",{"LR":self.scheduler.get_lr()[0]}, (self.Epoch_cnt* self.len_batchs) + self.Batch_cnt)

                if self.Batch_cnt % 5 == 0:
                    self.writer.add_scalars("TRAIN_REAL/Loss",{"Lbaseline":Lreal_list[0], "Lce":Lreal_list[1], "Ldc":Lreal_list[2], "Ltv":Lreal_list[3], "Lsc":Lreal_list[4]}, (self.Epoch_cnt* self.len_batchs) + self.Batch_cnt)
                    self.writer.add_scalars("TRAIN_REAL/ACC",{"ACC":real_acc}, (self.Epoch_cnt* self.len_batchs) + self.Batch_cnt)

        # % vis defogging result : Defoggy_plot(input_img, Gan_fake, Gan_enh, gt_img, save_full_path)
        if self.cfg.TEST.VIS and (self.Batch_cnt == self.Epoch_cnt):
            sample_idx = 0
            sample_result_path = os.path.join(self.output_dir, "sample_result_for_train/")
            name = "Epoch_" + str(self.Epoch_cnt) + "_" + syn_foggy_pths[sample_idx].split("/")[-1]
            save_full_path = sample_result_path + name
            foggy_ = syn_foggy.cpu().numpy()[sample_idx].transpose((1, 2, 0))
            pred_fake = syn_gx_fake.detach().cpu().numpy()[sample_idx].transpose((1,2,0))
            pred_enh = syn_gx_enh.detach().cpu().numpy()[sample_idx].transpose((1,2,0))
            gt_ = syn_gt.detach().cpu().numpy()[sample_idx].transpose((1,2,0))
            Defoggy_plot(foggy_, pred_fake, pred_enh, gt_, save_full_path)

            if self.Batch_cnt % 5 == 0:
                real_result_path = os.path.join(self.output_dir, "sample_result_for_real/")
                name = "Epoch_" + str(self.Epoch_cnt) + "_" + real_pths[sample_idx].split("/")[-1]
                real_full_path = real_result_path + name
                foggy_ = real_foggy.cpu().numpy()[sample_idx].transpose((1, 2, 0))
                pred_fake = real_gx_fake.detach().cpu().numpy()[sample_idx].transpose((1,2,0))
                pred_enh = real_gx_enh.detach().cpu().numpy()[sample_idx].transpose((1,2,0))
                Defoggy_plot(foggy_, pred_fake, pred_enh, foggy_, real_full_path)
        return 0

    def handle_new_batch(self,):
        self.Batch_cnt += 1
        self.real_step += 1

    def handle_new_epoch(self):
        self.Batch_cnt = 1
        self.scheduler.step()
        self.logger.info('Epoch {} done'.format(self.Epoch_cnt))
        self.logger.info('-' * 20)

        if self.Epoch_cnt % self.eval_period == 0:
            mAP, defoggy = self.evaluate()   # mAP, [SSIM_fake, SSIM_enh]
            if mAP > self.best_mAP:
                self.save()
                if len(self.best_epoch) > 5:
                    self.remove(self.best_epoch[0])
                    self.best_epoch.remove(self.best_epoch[0])
                self.best_epoch.append(self.Epoch_cnt)
                self.best_mAP = mAP
                self.best_SSIM = defoggy[1]

        if self.Epoch_cnt % 50 == 0:
            self.save()

        self.Epoch_cnt += 1
        self.logger.info('Best_epoch {}, best_mAP {}'.format(self.best_epoch[-1], self.best_mAP))
        self._make_new_real_batch()

    def evaluate(self):
        num_query = self.num_query
        feats, pids, camids = [], [], []
        SSIM_fake, PSNR_fake = [], []
        SSIM_enh, PSNR_enh = [], []
        vis_idx = self.Epoch_cnt
        if vis_idx > len(self.val_dl):
            vis_idx = vis_idx % len(self.val_dl)
      
        # % Val forward..
        ####################################################################################################
        ## val_loader: getitem[Index]: {"foggy", "gt", "foggy_paths", "gt_paths", "pids", "camids"}
        ## -eval: forward: global_feat, [defoggy_fake, defoggy_ehn]
        ####################################################################################################
        self.model.eval()
        with torch.no_grad():
            for ct, batch in enumerate(tqdm(self.val_dl, total=len(self.val_dl), leave=False)):
                foggy_img, gt_img, pid, camid, foggy_paths = batch['foggy'], batch['gt'], batch['pids'], batch['camids'], batch['foggy_paths']
                foggy_img, gt_img = foggy_img.cuda(), gt_img.cuda()
                feat, generated = self.model(foggy_img)  # global_feat, [defoggy_fake, defoggy_ehn]

                ### % reid.....
                feat = feat.detach().cpu()
                feats.append(feat)
                pids.append(pid)
                camids.append(camid)

                ### % restoration.....
                gx_fake = generated[0].cuda()
                gx_enh = generated[1].cuda()
                if self.cfg.DATALOADER.NORMALZIE:
                    gx_fake = gx_fake*0.5 + 0.5
                    gx_enh = gx_enh*0.5 + 0.5
                    gt_img = gt_img*0.5 + 0.5

                ssim_bts_fake, psnr_bts_fake = eval_defoggy(gx_fake, gt_img)
                ssim_bts_ehn, psnr_bts_ehn = eval_defoggy(gx_enh, gt_img)

                SSIM_fake.append(float(ssim_bts_fake)), PSNR_fake.append(float(psnr_bts_fake))
                SSIM_enh.append(float(ssim_bts_ehn)), PSNR_enh.append(float(psnr_bts_ehn))

                if ct == vis_idx:
                    if self.cfg.TEST.VIS:
                        sample_idx = 0
                        sample_result_path = os.path.join(self.output_dir, "sample_result_for_val/")
                        name = "Epoch_" + str(self.Epoch_cnt) + "_" + foggy_paths[sample_idx].split("/")[-1]
                        save_full_path = sample_result_path + name
                        foggy_ = foggy_img.cpu().numpy()[sample_idx].transpose((1, 2, 0))
                        pred_fake = gx_fake.cpu().numpy()[sample_idx].transpose((1,2,0))
                        pred_ehn = gx_enh.cpu().numpy()[sample_idx].transpose((1,2,0))
                        gt_ = gt_img.cpu().numpy()[sample_idx].transpose((1, 2, 0))
                        Defoggy_plot(foggy_, pred_fake, pred_ehn, gt_, save_full_path)

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

            self.logger.info('Validation Result:')
            for r in self.cfg.TEST.CMC:
                self.logger.info('CMC Rank-{}: {:.4%}'.format(r, cmc[r-1]))
            self.logger.info('mAP: {:.4%}'.format(mAP))
            self.logger.info('SSIM_fake: {:.3f}, SSIM_ehn: {:.3f}'.format(SSIM_fake, SSIM_enh))
            self.logger.info('PSNR_fake: {:.3f}, PSNR_ehn: {:.3f}'.format(PSNR_fake, PSNR_enh))
            self.logger.info('-' * 20)
            
            if self.cfg.MODEL.TENSORBOARDX:
                self.writer.add_scalars("TEST/SSIM",{"SSIM_fake":SSIM_fake, "SSIM_ehn":SSIM_enh}, (self.Epoch_cnt* self.len_batchs) + self.Batch_cnt)
                self.writer.add_scalars("TEST/PSNR",{"PSNR_fake":PSNR_fake, "PSNR_ehn":PSNR_enh}, (self.Epoch_cnt* self.len_batchs) + self.Batch_cnt)
                self.writer.add_scalars("TEST/mAP",{"mAP":mAP}, (self.Epoch_cnt* self.len_batchs) + self.Batch_cnt)
                self.writer.add_scalars("TEST/CMC",{"CMC1":cmc[0], "CMC5":cmc[4], "CMC10":cmc[9]}, (self.Epoch_cnt* self.len_batchs) + self.Batch_cnt)
                      
            return mAP, [SSIM_fake, SSIM_enh]
    
    def _make_real_dict_init(self,):
        for index in range(len(self.real_dl)):
            print(index, end='\r')
            real_data = self.real_dl.__getitem__([index])
            pid = real_data['pids'][0].tolist()
            self.real_data_dict[pid].append(index)
        self.real_pids = list(self.real_data_dict.keys())

    def _make_new_real_batch(self,):
        num_batch_id = int(self.cfg.SOLVER.IMS_PER_BATCH / self.cfg.DATALOADER.NUM_INSTANCE)   # 每個batch 需要幾個id
        new_epochs_batchs = []
        for tt in range(self.len_batchs):
            new_iter_batch = []
            cho_pids = np.random.choice(self.real_pids, num_batch_id, replace=False)
            for pid in cho_pids:
                real_idxs = self.real_data_dict[pid]
                if len(real_idxs) < self.cfg.DATALOADER.NUM_INSTANCE:
                    cho_idx = np.random.choice(real_idxs, self.cfg.DATALOADER.NUM_INSTANCE, replace=True)
                else:
                    cho_idx = np.random.choice(real_idxs, self.cfg.DATALOADER.NUM_INSTANCE, replace=False)
                for idx in cho_idx:
                    new_iter_batch.append(idx)
            new_epochs_batchs.append(new_iter_batch)
        self.real_batch_index = new_epochs_batchs
        self.real_step = 0  

    def save(self):
        torch.save(self.model.state_dict(), osp.join(self.output_dir, "checkpoint/", 
                self.cfg.MODEL.NAME + '_epoch' + str(self.Epoch_cnt) + '.pth'))
        if self.Epoch_cnt > 20:
            torch.save(self.optim.state_dict(), osp.join(self.output_dir, "checkpoint/", 
                    self.cfg.MODEL.NAME + '_epoch'+ str(self.Epoch_cnt) + '_optim.pth'))

    def remove(self, epoch):
        pre_checkpoint = osp.join(self.output_dir, "checkpoint/", self.cfg.MODEL.NAME + '_epoch' + str(epoch) + '.pth')
        if os.path.isfile(pre_checkpoint):
            os.remove(pre_checkpoint)

        pre_checkpointopt = osp.join(self.output_dir, "checkpoint/", self.cfg.MODEL.NAME + '_epoch'+ str(epoch) + '_optim.pth')
        if os.path.isfile(pre_checkpointopt):
            os.remove(pre_checkpointopt)  

if __name__ == '__main__':
    train()
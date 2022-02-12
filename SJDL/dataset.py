import torchvision.transforms as T
from torch.utils.data.dataloader import DataLoader
import os.path as osp
import glob
import re
import numpy as np
import torch
import copy
import random
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from collections import defaultdict
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

######################################################################
## Transformer
######################################################################
def get_trm(cfg, is_train=True):
    trans_list = []
    if is_train:
        trans_list.append(T.Resize(cfg.INPUT.SIZE_TRAIN))
        # trans_list.append(T.RandomHorizontalFlip(p=cfg.INPUT.PROB))
        trans_list.append(T.Pad(cfg.INPUT.PADDING))
        trans_list.append(T.RandomCrop(cfg.INPUT.SIZE_TRAIN))
    else:
        trans_list.append(T.Resize(cfg.INPUT.SIZE_TEST))

    trans_list.append(T.ToTensor())
    if cfg.DATALOADER.NORMALZIE:
        normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        trans_list.append(normalize_transform)
    transform = T.Compose(trans_list)
    return transform

def read_image(img_path):
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img_type = 'RGB'
            img = Image.open(img_path).convert(img_type)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. " "Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

######################################################################
## Dataset init
######################################################################
class FVRIDDataset:
    def __init__(self, root='/data/reid',
                    train_dir='', train_gt_dir='',
                    query_dir='', query_gt_dir='',
                    gallery_dir='', gallery_gt_dir='',
                    real_foggy_dir='', verbose=True, **kwargs):
        # Data Paths
        self.dataset_dir = root
        self.train_dir = osp.join(self.dataset_dir, train_dir)
        print("train_dir:",self.train_dir)
        self.train_gt_dir = osp.join(self.dataset_dir, train_gt_dir)
        print("train_gt_dir:",self.train_gt_dir)
        self.query_dir = osp.join(self.dataset_dir, query_dir)
        print("query_dir:",self.query_dir)
        self.query_gt_dir = osp.join(self.dataset_dir, query_gt_dir)
        print("query_gt_dir:",self.query_gt_dir)
        self.gallery_dir = osp.join(self.dataset_dir, gallery_dir)
        print("gallery_dir:",self.gallery_dir)
        self.gallery_gt_dir = osp.join(self.dataset_dir, gallery_gt_dir) 
        print("gallery_gt_dir:",self.gallery_gt_dir)
        self.real_foggy_dir = osp.join(self.dataset_dir, real_foggy_dir)
        print("real_foggy_dir:",self.real_foggy_dir)
        # 確認路徑存在 否則報錯
        self._check_before_run()
        # Data Source Process
        ###############################################
        # [foggy_img_path, gt_img_path, pid, camid]
        ###############################################
        self.train_set = self._process_dir_dhreid(self.train_dir, self.train_gt_dir, relabel=True) 
        self.real_foggy = self._process_dir_dhreid(self.real_foggy_dir, self.real_foggy_dir, relabel=True)
        self.query = self._process_dir_dhreid(self.query_dir, self.query_gt_dir, relabel=False)    
        self.gallery = self._process_dir_dhreid(self.gallery_dir, self.gallery_gt_dir, relabel=False)
        # Get Data info
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info_fvrid(self.train_set)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info_fvrid(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info_fvrid(self.gallery)
        self.num_real_pids, self.num_real_imgs, self.num_real_cams = self.get_imagedata_info_fvrid(self.real_foggy)
        if verbose:
            print("=> Data loaded")
            print("Dataset statistics:")
            print("  ----------------------------------------")
            print("  subset   | # ids | # images | # cameras")
            print("  ----------------------------------------")
            print("  train    | {:5d} | {:8d} | {:9d}".format(self.num_train_pids, self.num_train_imgs, self.num_train_cams))
            print("  query    | {:5d} | {:8d} | {:9d}".format(self.num_query_pids, self.num_query_imgs, self.num_query_cams))
            print("  gallery  | {:5d} | {:8d} | {:9d}".format(self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams))
            print("  real_foggy  | {:5d} | {:8d} | {:9d}".format(self.num_real_pids, self.num_real_imgs, self.num_real_cams))
            print("  ----------------------------------------")

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
        if not osp.exists(self.train_gt_dir):
            raise RuntimeError("'{}' is not available".format(self.train_gt_dir))
        if not osp.exists(self.query_gt_dir):
            raise RuntimeError("'{}' is not available".format(self.query_gt_dir))
        if not osp.exists(self.gallery_gt_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_gt_dir))
        if not osp.exists(self.real_foggy_dir):
            raise RuntimeError("'{}' is not available".format(self.real_foggy_dir))

    def _process_dir_fvrid(self, dir_path, gt_path, relabel=False, num=None):
        if num == None:
            foggy_img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        else:
            foggy_img_paths = glob.glob(osp.join(dir_path, '*.jpg'))[:num]
        pattern = re.compile(r'([-\d]+)_c([\d]+)')
        # make relabel library for train classification 
        pid_container = set()
        for img_path in foggy_img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        # make basedataset
        dataset = []
        for img_path in foggy_img_paths:  
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            gt_img_path = osp.join(gt_path, img_path.split("/")[-1])
            dataset.append((img_path, gt_img_path, pid, camid))    # [foggy_img_path, gt_img_path,  pid, camid]
        # check pair data
        error_ct = 0
        for foggy_img, gt_img, id, cid in dataset:
            foggy_name = foggy_img.split("\\")[-1]
            gt_name = gt_img.split("\\")[-1]
            if foggy_name != gt_name:
                error_ct += 1
        if error_ct != 0:
            print("!!! Data Error: %d/%d" %(error_ct, len(dataset)))
        return dataset

    def get_imagedata_info_fvrid(self, data):
        pids, cams = [], []
        for _, _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

class ImageDataset_for_Pairs(Dataset):
    """ Image Vehicle ReID Dataset for pairs data (foggy/gt) """
    """ Index: [foggy_img, gt_img, pid, camid, foggy_path, gt_path] """
    def __init__(self, dataset, cfg, transform=None):
        self.dataset = dataset
        self.cfg = cfg
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        foggy_path, gt_path, pid, camid = self.dataset[index]
        foggy_img = read_image(foggy_path)
        gt_img = read_image(gt_path)
        if self.transform is not None:
            foggy_img = self.transform(foggy_img)
            gt_img = self.transform(gt_img)
        return foggy_img, gt_img, pid, camid, foggy_path, gt_path

class ImageDataset_for_real(Dataset):
    """ Image Vehicle ReID Dataset for real foggy data """
    """ Index: {"images", "paths", "pids", "camids"} """
    def __init__(self, dataset, cfg, transform=None):
        self.dataset = dataset
        self.cfg = cfg
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indexs):
        paths, _, pids, camids, imgs = [], [], [], [], [], []
        for idx in indexs:
            path, _, pid, camid = self.dataset[idx]
            img = read_image(path)
            if self.transform is not None:
                img = self.transform(img)
            paths.append(path)
            pids.append(pid)
            camids.append(camid)
            imgs.append(img)

        imgs = torch.stack(imgs, dim=0)
        pids = torch.tensor(pids, dtype=torch.int64)
        camids = torch.tensor(camids, dtype=torch.int64)

        return {"images":imgs, "paths":paths, 'pids':pids, 'camids':camids}

######################################################################
## Dataloader function - ID Sampler
## fn = ("foggy", "gt", "foggy_paths", "gt_paths", "pids", "camids")
######################################################################
class RandomIdentitySampler(Sampler):
    """
    Only use for training to suitable for triplet 
    """
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, _, pid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []
        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)
        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length

def collate_FVRID_fn(batch):
    foggy_imgs, gt_imgs, pids, camids, foggy_paths, gt_paths, = zip(*batch)
    foggy_imgs =  torch.stack(foggy_imgs, dim=0)
    gt_imgs =  torch.stack(gt_imgs, dim=0)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return {"foggy":foggy_imgs, "gt":gt_imgs, 
            "foggy_paths":foggy_paths, "gt_paths":gt_paths, 
            "pids":pids, "camids":camids}

#####################################################################
## Create FVRID dataloader
## train_loader: getitem[Index]: {"foggy", "gt", "foggy_paths", "gt_paths", "pids", "camids"}
## real_loader: getitem[Index]: {"images", "paths", "pids", "camids"}
## val_loader: getitem[Index]: {"foggy", "gt", "foggy_paths", "gt_paths", "pids", "camids"}
#####################################################################
def make_FVRID_dl(cfg, num_gpus=1):
    print("###########################################################")
    print("### Make dataloader for FVRID !!!                       ###")
    print("###########################################################")
    train_trm = get_trm(cfg, is_train=True)
    val_trm = get_trm(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS * num_gpus

    # {train_set, real_foggy, query, gallery}
    dataset = FVRIDDataset(root=cfg.DATASETS.DATA_PATH,
                           train_dir=cfg.DATASETS.TRAIN_PATH, train_gt_dir=cfg.DATASETS.TRAIN_GT_PATH,
                           query_dir=cfg.DATASETS.QUERY_PATH, query_gt_dir=cfg.DATASETS.QUERY_GT_PATH,
                           gallery_dir=cfg.DATASETS.GALLERY_PATH, gallery_gt_dir=cfg.DATASETS.GALLERY_GT_PATH, 
                           real_foggy_dir=cfg.DATASETS.REAL_FOGGY_PATH)  

    num_classes_syn = dataset.num_train_pids
    num_classes_real = dataset.num_real_pids

    # getitem[Index]: {"foggy", "gt", "foggy_paths", "gt_paths", "pids", "camids"}
    train_set = ImageDataset_for_Pairs(dataset.train_set, cfg, train_trm)   # Index: [foggy_img, gt_img, pid, camid, foggy_path, gt_path]
    train_loader = DataLoader(
        train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH * num_gpus,
        sampler=RandomIdentitySampler(dataset.train_set,
            cfg.SOLVER.IMS_PER_BATCH * num_gpus,
            cfg.DATALOADER.NUM_INSTANCE * num_gpus),
        num_workers=num_workers, collate_fn=collate_FVRID_fn)   

    # getitem[Index]: {"images", "paths", "pids", "camids"}
    real_loader = ImageDataset_for_real(dataset.real_foggy, cfg, train_trm)   

    # getitem[Index]: {"foggy", "gt", "foggy_paths", "gt_paths", "pids", "camids"}
    val_set = ImageDataset_for_Pairs(dataset.query + dataset.gallery, cfg, val_trm)  # Index: [foggy_img, gt_img, pid, camid, foggy_path, gt_path]
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH * num_gpus, shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_FVRID_fn)

    return train_loader, real_loader, val_loader, len(dataset.query), num_classes_syn, num_classes_real

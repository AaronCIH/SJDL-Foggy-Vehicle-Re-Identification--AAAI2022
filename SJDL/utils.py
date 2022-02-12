import os, sys
import os.path as osp
from os.path import dirname as ospdn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import logging

################################################
## Prediction Visualization:
## 1. Defoggy result plot
## 2. Rank10 result plot
################################################
# Defoggy plot
def Defoggy_plot(input_img, Gan_fake, Gan_enh, gt_img, save_full_path):
    """
    input image must to be the numpy format
    arg:
        input_img : foggy input image
        Gan_fake : restoration image 1
        Gan_enh : resotration image 2
        gt_img : Gt image (clear)
        save_full_path : save path with root/file.png
    """
    img_w, img_h, img_dims = input_img.shape
    plt.figure(figsize=(20, 20))
    ####### processing img #######
    full_img = np.hstack((input_img, Gan_fake, Gan_enh, gt_img))
    ####### plot result #########
    plt.axis('off')
    fig = plt.imshow(full_img)
    ####### labeling ########
    encoder = ["Input", "Gan_fake", "Gan_enh", "Gt"]
    for ct, code in enumerate(encoder):
       plt.text(ct*img_w+ct*14, -5, "%s"%code, color="blue", fontsize=24, fontweight='bold')
    plt.savefig(save_full_path)
    plt.show()

# Rank10 plot
class Rank10_engine:
    def __init__(self, n_row=-1, n_col=-1):
        self.n_row = n_row
        self.n_col = n_col
    
    def rank_list_to_im(self, rank_list, same_id, q_im_path, g_im_paths, save_path):
        """Save a query and its rank list as an image.
        Args:
            rank_list: a list, the indices of gallery images to show
            same_id: a list, len(same_id) = rank_list, whether each ranked image is
            with same id as query
            q_im_path: query image path
            g_im_paths: ALL gallery image paths
            save_path: save png root path.
        """
        ims = [self._read_im(q_im_path)]
        for idx, (ind, sid) in enumerate(zip(rank_list, same_id)):
            im = self._read_im(g_im_paths[ind])
            # Add green boundary to true positive, red to false positive
            color = np.array([0, 255, 0]) if sid else np.array([255, 0, 0])
            im = self._add_border(im, 3, color)
            ims.append(im)

        if self.n_row == -1 and self.n_col == -1:
            n_row, n_col = 1, len(rank_list) + 1
        assert n_row * n_col == len(rank_list) + 1
        
        im = self._make_im_grid(ims, n_row, n_col, 8, 255)
        self._save_im(im, save_path) 
        return Image.fromarray(im.transpose(1,2,0))

    def _read_im(self, im_path):
        im = np.asarray(Image.open(im_path))    # shape [H, W, 3]
        resize_h_w = (384, 292)
        if (im.shape[0], im.shape[1]) != resize_h_w:
            im = cv2.resize(im, resize_h_w[::-1], interpolation=cv2.INTER_LINEAR)
        im = im.transpose(2, 0, 1)    # shape [3, H, W]
        return im

    def _add_border(self, im, border_width, value):
        """Add color border around an image. The resulting image size is not changed.
        Args:
            im: numpy array with shape [3, im_h, im_w]
            border_width: scalar, measured in pixel
            value: scalar, or numpy array with shape [3]; the color of the border
        Returns:
            im: numpy array with shape [3, im_h, im_w]
        """
        assert (im.ndim == 3) and (im.shape[0] == 3)
        im = np.copy(im)
        if isinstance(value, np.ndarray):
            value = value.flatten()[:, np.newaxis, np.newaxis]
        im[:, :border_width, :] = value
        im[:, -border_width:, :] = value
        im[:, :, :border_width] = value
        im[:, :, -border_width:] = value
        return im

    def _make_im_grid(ims, n_rows, n_cols, space, pad_val):
        """Make a grid of images with space in between.
        Args:
            ims: a list of [3, im_h, im_w] images
            n_rows: num of rows
            n_cols: num of columns
            space: the num of pixels between two images
            pad_val: scalar, or numpy array with shape [3]; the color of the space
        Returns:
            ret_im: a numpy array with shape [3, H, W]
        """
        assert (ims[0].ndim == 3) and (ims[0].shape[0] == 3)
        assert len(ims) <= n_rows * n_cols
        h, w = ims[0].shape[1:]
        H = h * n_rows + space * (n_rows - 1)
        W = w * n_cols + space * (n_cols - 1)
        if isinstance(pad_val, np.ndarray):
            pad_val = pad_val.flatten()[:, np.newaxis, np.newaxis]  # reshape to [3, 1, 1]
        ret_im = (np.ones([3, H, W]) * pad_val).astype(ims[0].dtype)
        for n, im in enumerate(ims):
            r = n // n_cols
            c = n % n_cols
            h1 = r * (h + space)
            h2 = r * (h + space) + h
            w1 = c * (w + space)
            w2 = c * (w + space) + w
            ret_im[:, h1:h2, w1:w2] = im
        return ret_im

    def _save_im(self, im, save_path):
        """im: shape [3, H, W]"""
        if save_path in [None, '']:
            print("Error: save_path not available! ")
            return
        if not osp.exists(save_path):
            os.makedirs(save_path)
        im = im.transpose(1, 2, 0)
        Image.fromarray(im).save(osp.join(save_path, 'Rank10.png'))

################################################
## Record Logger:
## 1. setup_logger, writer recorder.
## 2. AvgerageMeter, record Meter.
################################################
# Logger file
def setup_logger(name, save_dir, distributed_rank, train=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "log.txt" if train else 'log_eval.txt'), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

## Model logger
class AvgerageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += float(val) * n
        self.cnt += n
        self.avg = self.sum / self.cnt

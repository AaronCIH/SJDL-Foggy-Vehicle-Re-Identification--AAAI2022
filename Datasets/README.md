# FVRID dataset for training and testing

## Examples of the dataset
![image](https://github.com/Cihsaing/SJDL-Foggy-Vehicle-Re-Identification--AAAI2022/blob/master/Datasets/Dataset.png)

## Pipeline of generation
### 1. Download Public Dataset
#### * VeriWild
Page: https://github.com/PKU-IMRE/VERI-Wild
```bash
@inproceedings{lou2019large,
title={VERI-Wild: A Large Dataset and a New Method for Vehicle Re-Identification in the Wild},
author={Lou, Yihang and Bai, Yan and Liu, Jun and Wang, Shiqi and Duan, Ling-Yu},
booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
pages = {3235--3243},
year={2019}
}
```
```bash
@inproceedings{lou2019large,
 title={Disentangled Feature Learning Network and a Comprehensive Benchmark for Vehicle Re-Identification},
 author={Bai, Yan and Liu, Jun and Lou, Yihang and Wang, Ce and Duan, Ling-Yu},
 booktitle={In IEEE Transactions on Pattern Analysis and Machine Intelligence},
 year={2021}
}
```

#### * Vehicle-1M
Page: http://www.nlpr.ia.ac.cn/iva/homepage/jqwang/Vehicle1M.htm <br> 
```bash
Haiyun Guo, Chaoyang Zhao, Zhiwei Liu, Jinqiao Wang, Hanqing Lu: Learning coarse-to-fine structured feature embedding for vehicle re-identification. AAAI 2018.
```
### 2. Place the datasets to the corrseponding folders.
### 3. Create FVRID and enter into preposcessing.

## Structure of the dataset
* Datasets
```bash
{
  "FVRID_real":[
        { 
            "train_foggy": Real world training data, eg. "20000_c014_007925_0.jpg"  ...
            "query_foggy": Real world query data, eg. "20156_c094_068364_0.jpg"  ...
            "gallery_foggy": Real world gallery data, eg. "21247_c001_242288_0.jpg"  ...
        }
   ],
   "FVRID_syn":[
        {
            "train_clear": GT training data, eg. "00000_c013_298403_0.jpg" ...
            "train_foggy": synethetic foggy training data, eg. "00000_c013_298403_0.jpg" ...
            "query_clear": GT query data, eg. "05000_c170_365655_0.jpg" ...
            "query_foggy": synethetic foggy query data, eg. "05000_c170_365655_0.jpg" ...
            "gallery_clear": GT gallery data, eg. "15000_c001_616718_0.jpg" ...
            "gallery_foggy": synethetic foggy gallery data, eg. "15000_c001_616718_0.jpg" ...
        }
    ]
    "Preprocessing": use to construct dataset,
    "Vehicle1M": source dataset of Vehicle1M,
    "VeriWild": source dataset of VeriWild,
}
```

# FVRID 資料集創建
透過人工label，創建第一個包含Foggy的訓練集。
## 資料集資訊
![image](https://github.com/Cihsaing/SJDL-Foggy-Vehicle-Re-Identification--AAAI2022/blob/master/Datasets/Dataset.png)

## 創建流程
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
### 2. 將資料集放入各自資料夾
### 3. 進入Preprocessing開始創建FVRID

## 資料集結構
* Datasets
```bash
{
  "FVRID_real":[
        { 
            "train_foggy":
            "query_foggy":
            "gallery_foggy":
        }
   ],
   "FVRID_syn":[
        {
            "train_clear":
            "train_foggy":
            "query_clear":
            "query_foggy":
            "gallery_clear":
            "gallery_foggy":
        }
    ]
    "Preprocessing":,
    "
            
  
}
```

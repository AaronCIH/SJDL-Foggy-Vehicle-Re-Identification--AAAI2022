# FVRID Pre-Processing
## 1_Split_Dataset
* 切割資料集，
'''bash
python Split.py
'''
## 2_Depth_Estimation
* Depth from Single Monocular Images
This is the test code for the paper: "Learning Depth from Single Monocular Images Using Deep Convolutional Neural Fields"
Github: https://github.com/comeonyang/Depth-Estimation-DCNF

1. Install according to the instructions.
2. Move FVRID_SYN_***_prediction.m files to ./Depth-Estimation-DCNF/demo/
3. run FVRID_SYN_***_prediction.m files

## 3_Synthesis_foggy
% 透過depth合成hazy, 公式 hazy = clear*t(x) + A(1-t(x)), A大氣光, t(x)透射值
% 需要資料: clear_image_folder, depth_image_folder.  # 注意事項: depth folder 需要對應每張clear image 創建folder 並放入predict_depth_gray.png
% 可調整參數 function beta, function A

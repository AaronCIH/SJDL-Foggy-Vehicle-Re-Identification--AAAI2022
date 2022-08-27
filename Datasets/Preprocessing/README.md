# FVRID Pre-Processing
> There are many processes have to check your data path, please kindly check it with the annotations: `please check your path !!!!`

## 1_Split_Dataset
```bash
python Split.py
```

## 2_Depth_Estimation
* Depth from Single Monocular Images
This is the test code for the paper: "Learning Depth from Single Monocular Images Using Deep Convolutional Neural Fields"
Github: https://github.com/comeonyang/Depth-Estimation-DCNF
```bash
1. Install according to the instructions.
2. Download the pre-trained model and move to created folder './model_trained'
2. Move FVRID_SYN_***_prediction.m files to ./Depth-Estimation-DCNF-master/demo/
3. run FVRID_SYN_***_prediction.m files (on the matlab)
```
Due to the original trained weight of depth estimation has been deleted, I provieds the record link:https://drive.google.com/drive/folders/1cJtLa-3fSv4xSd-njEp3R4Ebz0uWjlrX?usp=sharing

In addition, I have also sorted out additional depth annotations, you can download the data directly, and put the folder to ```"../Datasets/FVRID_syn/Depth/*"```: https://drive.google.com/drive/u/1/folders/1H2bhzxQGaQIB4wYhsB8yOX7-id9WibFl

## 3_Synthesis_foggy
> With the estimated depth map. We can render fog by the following physical model.
> `I(x) = J (x)* t(x) + A(1-t(x))`, where A is the atmoshperic light and t(x) is the transmission value. <br>

> Required data: clear_image_folder, depth_image_folder.<br>  

Note: the depth folder needs to be created with the same name and put the corresponding depth map (i.e., predict_depth_gray.png) into it. 
<br> Controllable parameters: beta and A. <br>
```bash
run Dense_Foggy_*_OTS.m
```

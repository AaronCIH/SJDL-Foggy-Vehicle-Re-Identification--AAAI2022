# FVRID Pre-Processing
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
2. Move FVRID_SYN_***_prediction.m files to ./Depth-Estimation-DCNF/demo/
3. run FVRID_SYN_***_prediction.m files
```

## 3_Synthesis_foggy
% With the estimated depth map. We can render fog by the following physical model. $I(x) = J (x)\times t(x) + A(1-t(x))$, where $A$ is the atmoshperic light and $t(x)$ is the transmission value.
% Required data: clear_image_folder, depth_image_folder.  # Note: the depth folder needs to be created with the same name and put the corresponding depth map (i.e., predict_depth_gray.png) into it.
% Controllable parameters: beta and A.
```bash
run Dense_Foggy_*_OTS.m
```

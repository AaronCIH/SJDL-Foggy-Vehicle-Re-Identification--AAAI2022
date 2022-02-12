- Instruction
% 透過depth合成hazy, 公式 hazy = clear*t(x) + A(1-t(x)), A大氣光, t(x)透射值
% 需要資料: clear_image_folder, depth_image_folder.  # 注意事項: depth folder 需要對應每張clear image 創建folder 並放入predict_depth_gray.png
% 可調整參數 function beta, function A

- 資料集範例 (*代表資料夾, -代表影像檔)
	* clear_image
	-	clear_0000.png
	-	clear_0001.png
	-	clear_0002.png
	-	...
	
	* sample_depth
		* clear_0000
		- 	predict_depth_gray.png  # 對應 clear_0000.png 的 gray depth 
		* clear_0001
		- 	predict_depth_gray.png
		* clear_0002
		- 	predict_depth_gray.png
		* ...

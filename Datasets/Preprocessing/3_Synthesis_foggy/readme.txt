- Instruction

% With the estimated depth map. We can render fog by the following physical model. I(x) = J (x)* t(x) + A(1-t(x)), where A is the atmoshperic light and t(x) is the transmission value.
% Required data: clear_image_folder, depth_image_folder.
% Controllable parameters: beta and A

- Example of the dataset (*represetns the folder, -represents the image)
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

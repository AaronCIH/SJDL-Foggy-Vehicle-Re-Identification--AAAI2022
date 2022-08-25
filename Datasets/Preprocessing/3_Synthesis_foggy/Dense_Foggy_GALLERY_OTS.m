% 透過depth合成hazy, hazy = clear*t(x) + A(1-t(x)) 
% 需要資料: clear_image_folder, depth_image_folder.  # 注意事項: depth folder 需要對應每張clear image 創建folder 並放入predict_depth_gray.png
% example: clear_image_folder 內有一張 clear_x.png, 要在depth_image_foler/clear_x/predict_depth_gray.png
% 可調整參數 function beta, function A
clear all
CLEAR_PATH = ('../../FVRID_syn/gallery_clear/');   % please check your path !!!!
DEPTH_PATH = ('../../FVRID_syn/Depth/gallery_clear/'); % please check your path !!!!
SAVE_PATH = ('../../FVRID_syn/gallery_foggy/');   % please check your path !!!!
mkdir(SAVE_PATH);

clear_image_list = dir(CLEAR_PATH);
%length(clear_image_list)

%start index
start_index = 1;  % !!!!!!!!!!
end_index = length(clear_image_list); % !!!!!!!!!! need to setting process max number

for index = start_index : end_index
    image_name = clear_image_list(index).name;
    if (image_name ~= ".") && (image_name ~= "..")
        depth_folder = image_name(1:19);   
        fprintf('processing image (%d of %d): %s\n', index, end_index, image_name(1:19));
        
        image = imread([CLEAR_PATH,image_name]);
        image = im2double(image);
        [x1,y1,z1] = size(image);
%         fprintf("%f ",image(:,:,1));
        S = imread([DEPTH_PATH,depth_folder,'/predict_depth_gray.png']);
        depth = im2double(S);
        depth = imresize(depth,[x1,y1],'bicubic');
%         fprintf("%f ", depth(1,1));
%         resize = imresize(image,[224,224],'bicubic');
        A = A_Rnad();  % !!!!!!!!
        beta = betaRand();   % !!!!!!!!!
        t = [];
        t_temp = [];
        t_temp = exp(-beta*depth);
        t(:,:,1) = t_temp;
        t(:,:,2) = t_temp;
        t(:,:,3) = t_temp;
        newG = image.*t+A*(1-t);
        newG2 = im2uint8(newG);

        new_filename = [SAVE_PATH,image_name];
        imwrite(newG2, new_filename);  % save
        fprintf("saving img ---");
%         figure
%         imshow(newG2)   % show
        close all
    end
end

fprintf("Finish !! \n");


function beta = betaRand()
    xmin=1.0;
    xmax=5.0;
    n = 1;
    beta = xmin+rand(1,1)*(xmax-xmin);
end

function A = A_Rnad()
    xmin=0.4;
    xmax=1;
    n = 1;
    A = xmin+rand(1,1)*(xmax-xmin);
end
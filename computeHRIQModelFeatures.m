%-------------------------------------------------------------------------
%
%  computeCNNfeatures.m
%  
%  Use this function to compute sequence of features vectors 
%  for input image. Note that C++/OpenCV implementation was used
%  instead of this script to produce the results in the paper,
%  and due to different downscaling in Matlab and OpenCV, the 
%  results using this script will not be identical to the 
%  results in the paper.
%
%
%  Input: 
%           test_image:    Path to the test video file (e.g. png, jpg)
%           cnn:           Convolutional neural network for spatial
%                          feature extraction
%           cpugpu:        For using CPU, set to 'cpu', and for using 
%                          GPU, set to 'gpu'
%
%  Output:
%           features:      Resulting sequence of feature vectors 
%

function features = computeHRIQModelFeatures(test_image, cnn, vit, cpugpu)

    % Open image
    img = imread(test_image);
    img = cast(img,'double');

    % Make sure the image is large enough for at least one patch
    patch_size = [224 224];
    [height,width,~] = size(img);
    if height<patch_size(1) || width<patch_size(2)
        img = imresize(img, patch_size);
    end
    img_small = imresize(img,0.5,'method','box');
    [height,width,~] = size(img_small);
    if height<patch_size(1) || width<patch_size(2)
        img_small = imresize(img_small, patch_size);
    end
    
    % Initializations
    features = [];
    
    % Extrat patches and spatial activity vector
    [patches,pos] = extract_patches(img_small);
    for i1=1:size(pos,1)
        features(i1,:) = [1 pos(i1,:) predict(cnn,patches(:,:,:,i1),...
                                      'ExecutionEnvironment',cpugpu),...
                                      activations(vit,patches(:,:,:,i1),...
                                      'cls_index',...
                                      'ExecutionEnvironment',cpugpu)'];
    end 
    [patches,pos] = extract_patches(img);
    for i2=1:size(pos,1)
        features(i1+i2,:) = [2 pos(i2,:) predict(cnn,patches(:,:,:,i2),...
                                      'ExecutionEnvironment',cpugpu),...
                                      activations(vit,patches(:,:,:,i2),...
                                      'cls_index',...
                                      'ExecutionEnvironment',cpugpu)'];
    end 
    % csv_file = test_image;
    % csv_file(end-2:end) = 'csv';
    % writematrix(features, csv_file);
end
    
function [im_patches,pos] = extract_patches(img)

    % Make Sobel filter -based spatial activity map
    [height,width,~] = size(img);
    patch_size = [224 224];
                
    % Split image in patches
    x_numb = ceil(width/patch_size(2));
    y_numb = ceil(height/patch_size(1));
    x_step = 1;
    y_step = 1;
    if x_numb>1 && y_numb>1
        x_step = floor((width-patch_size(1))/(x_numb-1));
        y_step = floor((height-patch_size(2))/(y_numb-1));
    end
    
    im_patches = [];
    pos = [];
    num_patches = 0;
    
    % Loop through all patches  
    for i=1:x_step:width-patch_size(2)+1
        for j=1:y_step:height-patch_size(2)+1      
            y_range = j:j+patch_size(2)-1;
            x_range = i:i+patch_size(1)-1;
            pos = [pos; 2*(i+111)/width-1 2*(j+111)/height-1];
            num_patches = num_patches + 1;
            im_patches(:,:,:,num_patches) = img(y_range, x_range,:);          
        end
    end 
end

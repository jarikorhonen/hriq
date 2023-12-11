%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  This script runs scripts to process LIVE Challenge database,
%  train CNN feature extractor, extract CNN and ViT features from 
%  HRIQ datasets, and finally train and test RNN model to predict MOS in 
%  HRIQ dataset using ten different random splits.
%
%  inputs: 
%          livec_path: path to the LIVE Challenge image quality  
%          database (e.g. 'd:\\live_challenge')
%
%          hriq_path: path to the HRIQ image quality  
%          database (e.g. 'd:\\hriq')
%
%          cpugpu: defined if CPU or GPU is used for training
%          and testing the CNN model, use either 'cpu' or 'gpu'
%
%  outputs: 
%          Displays SCC, PCC and RMSE results for cross-dataset
%          test on the RNN model (KoNIQ-10k / SPAQ)
%
function out = masterScript_hriq(livec_path, ...
                                 hriq_path, ...
                                 cpugpu)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Phase 1: setup 
%
cnn_model_file = 'cnn_model_3.mat';

% Make CLIVE patches
fprintf('Generating patches from LIVE Challenge database...\n');
processLiveChallenge(livec_path, livec_patches_path);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Phase 2: train the CNN feature extractor
fprintf('Training CNN feature extractor...\n');
trainCNNmodelV3(livec_patches_path, cnn_model_file, cpugpu);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Phase 3: computing feature sequences for HRIQ image dataset

% Read saved model
load(cnn_model_file,'model');

% Read and initialise pre-trained vanilla ViT model
vit = visionTransformer("base-16-imagenet-384");
lgraph = layerGraph(vit);
orig_mean = vit.Layers(1).Mean;
orig_std = vit.Layers(1).StandardDeviation;
lgraph = replaceLayer(lgraph,'imageinput', imageInputLayer([224 224 3], ...
    'Normalization', 'zscore', 'Name', 'imageinput', ...
    'Mean', orig_mean, 'StandardDeviation', orig_std));
lgraph = replaceLayer(lgraph,'softmax',regressionLayer('Name','regr'));
lgraph = replaceLayer(lgraph,'re-flatten',functionLayer(@(X) reflattenHack(X),'Name','re-flatten')); 
vit = assembleNetwork(lgraph);

% Read HRIQ metadata and compute features image by image
fprintf('Extracting features for HRIQ...\n');
metadata = readtable([hriq_path '/hriq_mos_file.csv']);
seqlen = size(metadata,1);
for i=1:seqlen
    filename = sprintf('%s/%s',hriq_path,char(metadata{i,1}));
    X{i} = computeHRIQModelFeatures(filename, model, vit, 'gpu')';
    size(X{i})
    Y(i,:) = (metadata{i,8}-1)./4;
    if mod(i,100)==0
        fprintf('Extracted features for %d/%d images\n',i,seqlen);
    end
end

% You may want to save the features here for future use
save("hriq_features.mat","-v7.3","X","Y");

load('hriq_features.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Phase 4: training and testing the RNN quality model

% Use these values to get exactly the same test splits as in the paper
vals = [0,3407,114514,199800,926,2022,3,9,90,900];
result = [];

for i=1:10
    seqlen = length(X); 
    fprintf('Training and testing the model...\n');
    rng(vals(i));      
    rand_seq = randperm(seqlen);

    % Split data to training and test sets    
    XTrain = X(rand_seq(1:ceil(0.8*seqlen)));
    YTrain = Y(rand_seq(1:ceil(0.8*seqlen)),:);
    XTest = X(rand_seq(ceil(0.8*seqlen)+1:seqlen));
    YTest = Y(rand_seq(ceil(0.8*seqlen)+1:seqlen),:);
    
    % Train and test to get new results
    new_result = trainAndTestHRIQmodel(XTrain, YTrain, XTest, YTest);

    fprintf('Result for test split %d:\n', i);
    fprintf('SCC: %1.3f PCC: %1.3f RMSE: %1.3f\n', new_result(1), new_result(2), new_result(3)*4);
    result = [result; new_result]
end
mean_result = mean(result)
fprintf('Average results with 10 splits on HRIQ dataset:\n');
fprintf('SCC: %1.3f PCC: %1.3f RMSE: %1.3f\n', mean_result(1), mean_result(2), mean_result(3)*4);
 
out = 0;
% end

% End of file %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

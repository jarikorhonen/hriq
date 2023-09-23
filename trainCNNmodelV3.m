% --------------------------------------------------------------
%   trainCNNmodel.m
%
%   This function trains the CNN model to be used as local
%   feature extractor.
%
%   Changes: weight initialization defined explicitely to
%   ensure backwards compatibility with other Matlab versions.
%   Tested with R2018b and R2020a.
%
%   Inputs: 
%       path:       Path to the training patches
%       model_file: Name of the file where to save the obtained
%                   CNN model
%       cpugpu:     For using CPU, set to 'cpu', and for using 
%                   GPU, set to 'gpu'
%
%   Output: dummy
%   
function res = trainCNNmodelV3(path, model_file)

    % Load probabilistic representations for quality scores
    %path = 'c:\\train_images';
    load(sprintf('%s\\LiveC_prob2.mat',path),'LiveC_prob');

    % Loop through all the test images to obtain source paths
    % for test images and the respective ground truth outputs
    filenames1 = {};
    outputs1 = [];
    sqlen = floor(length(LiveC_prob(:,1))/52);
    rng(123); % Seed for initialization
    sq = randperm(sqlen);
    for i=sq(1:ceil(0.95*sqlen))
        for j=1:52
            filenames1 = [filenames1; sprintf('%s\\%04d_%02d.png',path,i,j)];
            outputs1 = [outputs1; LiveC_prob((i-1)*52+j,:)];
        end
    end

    % % Additional 
    filenames2 = {};
    outputs2 = [];
    % files = dir(sprintf('%s\\lq_images\\*.png',path));
    % for i=1:length(files)
    %     filenames2 = [filenames2; [path '\\lq_images\\' files(i).name]];
    %     outp1 = 0.5+0.5*rand(1);
    %     outp2 = 1-outp1;
    %     outputs2 = [outputs2; outp1 outp2 0 0 0];
    % end
    % files = dir(sprintf('%s\\hq_images\\*.png',path));
    % for i=1:length(files)
    %     filenames2 = [filenames2; [path '\\hq_images\\' files(i).name]];
    %     outp1 = 0.5+0.25*rand(1);
    %     outp2 = 1-outp1;
    %     outputs2 = [outputs2; 0 0 0 outp2 outp1];
    % end

    filenames = [filenames1; filenames2];
    outputs = [outputs1; outputs2];

    T = table(filenames, outputs);

    filenames = {};
    outputs = [];
    mos = [];
    for i=sq(ceil(0.95*sqlen)+1:sqlen)
        for j=1:52
            filenames = [filenames; sprintf('%s\\%04d_%02d.png',path,i,j)];
            outputs = [outputs; LiveC_prob((i-1)*52+j,:)];
        end
    end
    Tval = table(filenames, outputs);   
    
    %-----------------------------------------------------------------
    % modified ResNet50 defined here
    %
    net = resnet50;
    lgraph = layerGraph(net);   

    lgraph = removeLayers(lgraph,'fc1000');
    lgraph = removeLayers(lgraph,'fc1000_softmax');
    lgraph = removeLayers(lgraph,'ClassificationLayer_fc1000');  
    
    lgraph = addLayers(lgraph, convolution2dLayer([3 3],128,'Stride',[1 1],'WeightLearnRateFactor',2,'BiasLearnRateFactor',2,'Name','conv_extra1','WeightsInitializer','narrow-normal')); 
    lgraph = addLayers(lgraph, batchNormalizationLayer('name','batchnorm_extra'));
    lgraph = addLayers(lgraph, reluLayer('Name','relu_extra'));
    lgraph = addLayers(lgraph, globalAveragePooling2dLayer('name','avg1'));
    lgraph = addLayers(lgraph, concatenationLayer(3,2,'name','concat')); 

    lgraph = connectLayers(lgraph, 'conv1', 'conv_extra1');
    lgraph = connectLayers(lgraph, 'conv_extra1', 'batchnorm_extra');
    lgraph = connectLayers(lgraph, 'batchnorm_extra', 'relu_extra');
    lgraph = connectLayers(lgraph, 'relu_extra', 'avg1');
    lgraph = connectLayers(lgraph, 'avg_pool', 'concat/in1');
    lgraph = connectLayers(lgraph, 'avg1', 'concat/in2');

    headLayers = layerGraph([...
       batchNormalizationLayer('Name','bn_red')
       dropoutLayer(0.25,'Name','do1')
       fullyConnectedLayer(512,'WeightLearnRateFactor',2,'BiasLearnRateFactor',2,'Name','fc_red','WeightsInitializer','glorot') 
       dropoutLayer(0.25,'Name','do2')
       fullyConnectedLayer(5,'WeightLearnRateFactor',2,'BiasLearnRateFactor',2,'Name','fc_output','WeightsInitializer','narrow-normal')       
       mySoftmaxLayer('softmax')
       myCrossentropyRegressionLayer('output')]);

    layers = [lgraph.Layers
              headLayers.Layers];
    connections = [lgraph.Connections
                   headLayers.Connections];
    layers(1:5) = setLayerWeights(layers(1:5),0.25);    
    layers(6:36) = setLayerWeights(layers(6:36),0.5); 
    layers(37:140) = setLayerWeights(layers(37:140),0.75); 
    
    lgraph = createLgraphUsingConnections(layers,connections);
    lgraph = connectLayers(lgraph, 'concat', 'bn_red');

    %------------------------------------------------------------

    % Define training options
    options = trainingOptions('sgdm', ...
        'MiniBatchSize',16, ...        
        'MaxEpochs',2, ...
        'L2Regularization',0.01, ...
        'InitialLearnRate',0.001, ...
        'LearnRateDropPeriod',1, ...
        'LearnRateDropFactor',0.5, ...
        'Shuffle','every-epoch', ...
        'ExecutionEnvironment','gpu', ...
        'ValidationData',Tval, ...
        'ValidationFrequency',200, ...
        'ResetInputNormalization',false, ...
        'Verbose',false, ...
        'Plots','training-progress');

    % Train the model
    model = trainNetwork(T,'outputs',lgraph,options);
   
% Optional: print prediction results
    % mospred = predict(model, Tval);   
    % mospr = sum(mospred'.*[1 2 3 4 5]');
    % mosgt = sum((Tval.outputs)'.*[1 2 3 4 5]');
    % res = [corr(mosgt', mospr','type','Spearman') ...
    %        corr(mosgt', mospr','type','Pearson') ...
    %        sqrt(mse(mosgt', mospr'))]

    lgraph = layerGraph(model);   
    lgraph = removeLayers(lgraph,{'do1','do2','fc_output','softmax'});
    lgraph = replaceLayer(lgraph,'output',regressionLayer('Name','output'));
    lgraph = connectLayers(lgraph, 'bn_red', 'fc_red');
    lgraph = connectLayers(lgraph, 'fc_red', 'output');
    lgraph = removeLayers(lgraph,{'fc_red'});
    lgraph = connectLayers(lgraph, 'bn_red', 'output');
    model = assembleNetwork(lgraph);
    save(model_file,'model');  
%end


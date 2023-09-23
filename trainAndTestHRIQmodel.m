% ----------------------------------------------------------------------
%  trainAndTestRNNmodelv2.m
%
%  Written by Jari Korhonen, Shenzhen University
%
%  This function trains the RNN model version 2 to predict quality
%  of large resolution test images, using a sequence of low and
%  high resultion patch features as input.
% 
%  Usage: result = trainAndTestRNNmodelv2(XTrain,YTrain,XTest,YTest)
%  Inputs:
%      XTrain:    Training feature vector sequences
%      YTrain:    Training ground truth MOS vector
%      XTest:     Testing feature vector sequences
%      YTest:     Testing ground truth MOS vector
%
%  Output:
%      [SROCC PCC RMSE]

function result = trainAndTestHRIQModel(XTrain,YTrain,XTest,YTest)

% Preprocess input data
XTrain = XTrain';
XTest = XTest';
XTest1 = {};
XTest2 = {};
for i=1:length(XTest)
    XTest1{i} = XTest{i}(:,XTest{i}(1,:)==1);
    XTest2{i} = XTest{i}(:,XTest{i}(1,:)==2);
end
%size(XTrain{1})
X1 = padsequences(XTest1,2,'Length','longest','Direction','left','PaddingValue',0);
X1 = permute(X1,[1 3 2]);
X2 = padsequences(XTest2,2,'Length','longest','Direction','left','PaddingValue',0);
X2 = permute(X2,[1 3 2]);
dlXt1 = dlarray(X1,'CBT');
dlXt2 = dlarray(X2,'CBT');
dlYt = dlarray(YTest,'BC');
numFeatures = size(XTrain{1},1)-1;
numFeatures2 = numFeatures;

% Initialize training options
miniBatchSize = 16;
numEpochs = 6;
learnRate = 0.0002;
learnRateDropFactor = 1;
gradientDecayFactor = 0.9;
squaredGradientDecayFactor = 0.999;

% Initialize datastore used for training the model
xDs = arrayDatastore(XTrain,'OutputType','same');
yDs = arrayDatastore(YTrain,'OutputType','same');
trainDs = combine(xDs,yDs);
clear("XTrain","YTrain");
numMiniBatchOutputs = 3;
mbq = minibatchqueue(trainDs,numMiniBatchOutputs,...
    'MiniBatchSize',miniBatchSize,...
    'MiniBatchFormat',{'CBT','CBT','BC'},...
    'MiniBatchFcn',@(x,t) preprocessMiniBatch(x,t));

% Initialize training visualization
figure
lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);
lineLossTest = animatedline('Color',[0 0 0]);
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on
iteration = 0;
start = tic;

rng(555); % seed for random initialization

% Initialize hyperparameters for regularization etc.
state.do1 = 1;
state.do2 = 0.5;
state.transpoint = 0.2;
state.mu1 = []; 
state.mu2 = []; 
state.sigmaSq1 = []; 
state.sigmaSq2 = [];

% Initialize batch normalization
parameters.bn.sf1 = dlarray(ones(numFeatures-2,1));
parameters.bn.offset1 = dlarray(zeros(numFeatures-2,1));
parameters.bn.sf2 = dlarray(ones(numFeatures-2,1));
parameters.bn.offset2 = dlarray(zeros(numFeatures-2,1));
% size(parameters.bn.offset1)

sz = [1 numFeatures];
numOut = 1;
numIn = numFeatures;
parameters.fcw.Weights = initializeGlorot(sz,numOut,numIn);
parameters.fcw.Bias = initializeZeros([numOut 1]);

% Initialize pre-stage FC layer (fcpre)
sz = [numFeatures2 numFeatures];
numOut = numFeatures2;
numIn = numFeatures;
parameters.fcpre.Weights = initializeGlorot(sz,numOut,numIn);
parameters.fcpre.Bias = initializeZeros([numOut 1]);

parameters.fcpre1.Weights = parameters.fcpre.Weights;
parameters.fcpre2.Weights = parameters.fcpre.Weights;
parameters.fcpre1.Bias = parameters.fcpre.Bias;
parameters.fcpre2.Bias = parameters.fcpre.Bias;

% Initialize 1st GRU layer for low resolution stream (gru_l1)
numHiddenUnits1 = 256;
sz = [3*numHiddenUnits1 numFeatures2];
parameters.gru_l1.Weights = initializeGlorot(sz,3*numHiddenUnits1,numFeatures2);
parameters.gru_l1.RecurrentWeights = initializeOrthogonal([3*numHiddenUnits1 numHiddenUnits1]);
parameters.gru_l1.Bias = initializeZerosGru(3*numHiddenUnits1);

% Initialize 2nd GRU layer for low resolution stream (gru_l2)
numHiddenUnits2 = 128;
sz = [3*numHiddenUnits2 numHiddenUnits1];
parameters.gru_l2.Weights = initializeGlorot(sz,3*numHiddenUnits2,numHiddenUnits1);
parameters.gru_l2.RecurrentWeights = initializeOrthogonal([3*numHiddenUnits2 numHiddenUnits2]);
parameters.gru_l2.Bias = initializeZerosGru(3*numHiddenUnits2);

% Initialize 3rd GRU layer for low resolution stream (gru_l3)
numHiddenUnits3 = 64;
sz = [3*numHiddenUnits3 numHiddenUnits2];
parameters.gru_l3.Weights = initializeGlorot(sz,3*numHiddenUnits3,numHiddenUnits2);
parameters.gru_l3.RecurrentWeights = initializeOrthogonal([3*numHiddenUnits3 numHiddenUnits3]);
parameters.gru_l3.Bias = initializeZerosGru(3*numHiddenUnits3);

parameters.mhsa_l.wq = dlarray(rand(64,64));
parameters.mhsa_l.wk = dlarray(rand(64,64));
parameters.mhsa_l.wv = dlarray(rand(64,64));
parameters.mhsa_l.wo = initializeHe([64 64],64);
 
% Initialize 1st GRU layer for high resolution stream (gru_h1)
numHiddenUnits1 = 256;
sz = [3*numHiddenUnits1 numFeatures2];
parameters.gru_h1.Weights = initializeGlorot(sz,3*numHiddenUnits1,numFeatures2);
parameters.gru_h1.RecurrentWeights = initializeOrthogonal([3*numHiddenUnits1 numHiddenUnits1]);
parameters.gru_h1.Bias = initializeZerosGru(3*numHiddenUnits1);

% Initialize 2nd GRU layer for high resolution stream (gru_h2)
numHiddenUnits2 = 128;
sz = [3*numHiddenUnits2 numHiddenUnits1];
parameters.gru_h2.Weights = initializeGlorot(sz,3*numHiddenUnits2,numHiddenUnits1);
parameters.gru_h2.RecurrentWeights = initializeOrthogonal([3*numHiddenUnits2 numHiddenUnits2]);
parameters.gru_h2.Bias = initializeZerosGru(3*numHiddenUnits2);

% Initialize 3rd GRU layer for high resolution stream (gru_h3)
numHiddenUnits3 = 64;
sz = [3*numHiddenUnits3 numHiddenUnits2];
parameters.gru_h3.Weights = initializeGlorot(sz,3*numHiddenUnits3,numHiddenUnits2);
parameters.gru_h3.RecurrentWeights = initializeOrthogonal([3*numHiddenUnits3 numHiddenUnits3]);
parameters.gru_h3.Bias = initializeZerosGru(3*numHiddenUnits3);

% Initialize the GRU layer for head (gruhead)
numHiddenUnits4 = 1;
sz = [3*numHiddenUnits4 128];
parameters.gruhead.Weights = initializeGlorot(sz,3*numHiddenUnits4, 128);
parameters.gruhead.RecurrentWeights = initializeOrthogonal([3*numHiddenUnits4 numHiddenUnits4]);
parameters.gruhead.Bias = initializeZerosGru(3*numHiddenUnits4);

trailingAvg = [];
trailingAvgSq = [];

% Loop over epochs
for epoch = 1:numEpochs
    
    shuffle(mbq);
    reset(mbq);
        
    % Loop over mini-batches.
    while hasdata(mbq)
    
        iteration = iteration + 1;
        
        [dlX1,dlX2,T] = next(mbq);
        
        % Compute loss and gradients.
        [gradients,loss,newstate] = dlfeval(@modelGradients,parameters,state,dlX1,dlX2,T);
        state = newstate;

        % Update parameters using adamupdate.
        [parameters,trailingAvg,trailingAvgSq] = adamupdate(parameters,gradients,trailingAvg,trailingAvgSq,...
            iteration,learnRate,gradientDecayFactor,squaredGradientDecayFactor,10^(-9));
        
        % Display the training progress.
        D = duration(0,0,toc(start),'Format','hh:mm:ss');
        addpoints(lineLossTrain,iteration,sqrt(double(extractdata(loss))))
        title("Epoch: " + epoch + ", Elapsed: " + string(D))
        if size(dlXt1,2)>2048
            if mod(iteration,50)==0 || iteration==1 
                for i=1:2048:size(dlXt1,2)
                    dlYp(i:min(size(dlXt1,2),i+2047)) = model(parameters,state, ...
                                                             dlXt1(:,i:min(size(dlXt1,2),i+2047),:),...
                                                             dlXt2(:,i:min(size(dlXt1,2),i+2047),:),1);
                end
                YPred = double(extractdata(dlYp))';      
                YTest = double(extractdata(dlYt))'; 
                addpoints(lineLossTest,iteration,sqrt(mse(YPred,YTest)));
                res = [corr(YTest,YPred,'type','Spearman') ...
                       corr(YTest,YPred,'type','Pearson') ...
                       sqrt(mse(YTest,YPred))]     
            end
        else
             if mod(iteration,50)==0 || iteration==1 
                dlYp = model(parameters,state,dlXt1,dlXt2,1);
                YPred = double(extractdata(dlYp))';      
                YTest = double(extractdata(dlYt))'; 
                addpoints(lineLossTest,iteration,sqrt(mse(YPred,YTest)));
                res = [corr(YTest,YPred,'type','Spearman') ...
                       corr(YTest,YPred,'type','Pearson') ...
                       sqrt(mse(YTest,YPred))]  
            end           
        end
        drawnow
    end

    % Update learning rate
    learnRate = learnRate * learnRateDropFactor;
    state.do1 = state.do1 * 0.5;
    state.do2 = state.do2 * 0.5;
    learnRateDropFactor = learnRateDropFactor * 0.7;
    
end % end of training loop

mbq = [];

if size(dlXt1,2)<2048
    dlYp = model(parameters,state,dlXt1,dlXt2,0);
else
    for i=1:2048:size(dlXt1,2)
        dlYp(i:min(size(dlXt1,2),i+2047)) = model(parameters,state, ...
                                                 dlXt1(:,i:min(size(dlXt1,2),i+2047),:),...
                                                 dlXt2(:,i:min(size(dlXt1,2),i+2047),:),0);
    end
end
YPred = double(extractdata(dlYp))';      
YTest = double(extractdata(dlYt))'; 
size(YPred)
size(YTest)
result = [corr(YTest,YPred,'type','Spearman') ...
          corr(YTest,YPred,'type','Pearson') ...
          sqrt(mse(YTest,YPred))]
% 
% plot(YTest,YPred);

end


function [X1,X2,T] = preprocessMiniBatch(src,trg)
    X1 = {};
    X2 = {};
    for i=1:length(src)
        X1{i} = src{i}(:,src{i}(1,:)==1);
        X2{i} = src{i}(:,src{i}(1,:)==2);
    end
    X1 = padsequences(X1,2,'Length','longest','Direction','left','PaddingValue',0);
    X1 = permute(X1,[1 3 2]);
    X2 = padsequences(X2,2,'Length','longest','Direction','left','PaddingValue',0);
    X2 = permute(X2,[1 3 2]);
    T = cat(1,trg{:});
end

function [gradients,loss,newstate] = modelGradients(parameters,state,dlX1,dlX2,T)
    [dlY,newstate] = model(parameters, state, dlX1, dlX2, 1);
    loss = huber(dlY,T,'TransitionPoint',state.transpoint); 
    
    % Update gradients.
    gradients = dlgradient(loss,parameters);
end

% ------------------------------------------------------------------------------------------
function [dlY,newstate] = model(parameters,state,dlX1,dlX2,training)

    % Regularization
    newstate = state;
    if training
        mult = dlarray(rand(size(dlX1(4:end,:,:))),'CBT');
        dlX1(4:end,:,:) = dlX1(4:end,:,:).*(mult.*state.do1+1-state.do1/2);
        mult = dlarray(rand(size(dlX2(4:end,:,:))),'CBT');
        dlX2(4:end,:,:) = dlX2(4:end,:,:).*(mult.*state.do2+1-state.do2/2);
    end           

    % Prescaling by FC layer
    dlY1 = (dlX1(2:end,:,:));
    dlY2 = (dlX2(2:end,:,:));
    dlY1 = fullyconnect(dlY1,parameters.fcpre.Weights,parameters.fcpre.Bias);
    dlY1 = relu((dlY1));
    dlY2 = fullyconnect(dlY2,parameters.fcpre.Weights,parameters.fcpre.Bias);
    dlY2 = relu((dlY2));
    % 
    if training
        mult = dlarray(rand(size(dlY1)),'CBT');
        dlY1 = dlY1.*(mult.*state.do1+1-state.do1/2);
        mult = dlarray(rand(size(dlY2)),'CBT');
        dlY2 = dlY2.*(mult.*state.do2+1-state.do2/2);
    end

    % Low resolution stream
    H0 = dlarray(zeros(size(parameters.gru_l1.Bias,1)/3,size(dlY1,2)),'CB');
    dlY1 = gru(dlY1,H0,parameters.gru_l1.Weights,parameters.gru_l1.RecurrentWeights,parameters.gru_l1.Bias);
    H0 = dlarray(zeros(size(parameters.gru_l2.Bias,1)/3,size(dlY1,2)),'CB');
    dlY1 = gru(dlY1,H0,parameters.gru_l2.Weights,parameters.gru_l2.RecurrentWeights,parameters.gru_l2.Bias);
    H0 = dlarray(zeros(size(parameters.gru_l3.Bias,1)/3,size(dlY1,2)),'CB');
    dlY1 = gru(dlY1,H0,parameters.gru_l3.Weights,parameters.gru_l3.RecurrentWeights,parameters.gru_l3.Bias);   

    % High resolution stream
    H0 = dlarray(zeros(size(parameters.gru_h1.Bias,1)/3,size(dlY1,2)),'CB');
    dlY2 = gru(dlY2,H0,parameters.gru_h1.Weights,parameters.gru_h1.RecurrentWeights,parameters.gru_h1.Bias);
    H0 = dlarray(zeros(size(parameters.gru_h2.Bias,1)/3,size(dlY2,2)),'CB');
    dlY2 = gru(dlY2,H0,parameters.gru_h2.Weights,parameters.gru_h2.RecurrentWeights,parameters.gru_h2.Bias);
    H0 = dlarray(zeros(size(parameters.gru_h3.Bias,1)/3,size(dlY2,2)),'CB');
    dlY2 = gru(dlY2,H0,parameters.gru_h3.Weights,parameters.gru_h3.RecurrentWeights,parameters.gru_h3.Bias);

    dlZ1 = dlarray(ones(1,1,size(dlY1,3)));
    mult = (dlarray((1:size(dlY1,3))./size(dlY1,3)).^2);
    dlZ1(1,1,:) = mult./sum(mult);
    dlY1 = sum(dlY1.*dlZ1,3); %dlY1(:,:,end);

    dlZ2 = dlarray(ones(1,1,size(dlY2,3)));
    mult = (dlarray((1:size(dlY2,3))./size(dlY2,3)).^2);
    dlZ2(1,1,:) = mult./sum(mult);
    dlY2 = sum(dlY2.*dlZ2,3); 

    % Concatenate streams
    dlY = cat(1,dlY1,dlY2);
    dlY = relu(dlY);
    
    % Head
    H0 = dlarray(zeros(size(parameters.gruhead.Bias,1)/3,size(dlY,2)),'CB');
    dlY = gru(dlY,H0,parameters.gruhead.Weights,parameters.gruhead.RecurrentWeights,parameters.gruhead.Bias);
    dlY = dlarray(dlY,'CB');
end


% Weight initializers
function weights = initializeGlorot(sz,numOut,numIn)

Z = 2*rand(sz,'single') - 1;
bound = sqrt(6 / (numIn + numOut));

weights = bound * Z;
weights = dlarray(weights);

end

function weights = initializeHe(sz,numIn)

weights = randn(sz,'single') * sqrt(2/numIn);
weights = dlarray(weights);

end

function parameter = initializeZeros(sz)

parameter = zeros(sz,'single');
parameter = dlarray(parameter);

end

function parameter = initializeOrthogonal(sz)

Z = randn(sz,'single');
[Q,R] = qr(Z,0);

D = diag(R);
Q = Q * diag(D ./ abs(D));

parameter = dlarray(Q);

end

function parameter = initializeZerosGru(sz)

parameter = zeros(sz,1,'single');
parameter = dlarray(parameter);

end


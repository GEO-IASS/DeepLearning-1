%% CS294A/CS294W Stacked Autoencoder Exercise

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  sstacked autoencoder exercise. You will need to complete code in
%  stackedAECost.m
%  You will also need to have implemented sparseAutoencoderCost.m and 
%  softmaxCost.m from previous exercises. You will need the initializeParameters.m
%  loadMNISTImages.m, and loadMNISTLabels.m files from previous exercises.
%  
%  For the purpose of completing the assignment, you do not need to
%  change the code in this file. 
%
%%======================================================================
%% STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.

DEBUG = true;   % Set DEBUG to true when debugging.
%DEBUG = false;

inputSize = 28 * 28;
numClasses = 10;
hiddenSizeL1 = 200;    % Layer 1 Hidden Size
hiddenSizeL2 = 200;    % Layer 2 Hidden Size
sparsityParam = 0.1;   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		               %  in the lecture notes). 
lambda = 3e-3;         % weight decay parameter       
beta = 3;              % weight of sparsity penalty term       

%%======================================================================
%% STEP 1: Load data from the MNIST database
%
%  This loads our training data from the MNIST database files.

% Load MNIST database files
trainData = loadMNISTImages('mnist/train-images-idx3-ubyte');
trainLabels = loadMNISTLabels('mnist/train-labels-idx1-ubyte');

trainLabels(trainLabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1

%%======================================================================
%% STEP 2: Train the first sparse autoencoder
%  This trains the first sparse autoencoder on the unlabelled STL training
%  images.
%  If you've correctly implemented sparseAutoencoderCost.m, you don't need
%  to change anything here.

%  Check if file 'sae1OptTheta.mat' exist
if (exist ("sae1OptTheta.mat", "file"))
  load sae1OptTheta.mat;
else
  disp ("File 'sae1OptTheta.mat' not found.");
endif

%  Check if var 'sae1OptTheta' exist
if (exist ("sae1OptTheta", "var"))
  disp ("'sae1OptTheta' parameters loaded from file 'sae1OptTheta.mat'.");
else
  disp ("Training the 1st layer of Stacked AutoEncoder ...");

  %  Randomly initialize the parameters
  sae1Theta = initializeParameters(hiddenSizeL1, inputSize);

  %% ---------------------- YOUR CODE HERE  ---------------------------------
  %  Instructions: Train the first layer sparse autoencoder, this layer has
  %                an hidden size of "hiddenSizeL1"
  %                You should store the optimal parameters in sae1OptTheta

  addpath minFunc/;               % Use minFunc to minimize the cost function of sparse autoencoder
  options.Method = 'lbfgs';       % Use L-BFGS to optimize sparse autoencoder cost
  options.maxIter = 400;          % Maximum number of iterations of L-BFGS to run 
  options.display = 'on';
  options.useMex = 0;             % Bugfix: error: 'lbfgsC' undefined near line 559 column 25

  [sae1OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
				    inputSize, hiddenSizeL1, ...
				    lambda, sparsityParam, ...
				    beta, trainData), ...
				sae1Theta, options);
endif

%  Saving 'sae1OptTheta' parameters in file 'sae1OptTheta.mat'
if (! exist ("sae1OptTheta.mat", "file"))
  save sae1OptTheta.mat sae1OptTheta;
  disp ("'sae1OptTheta' parameters saved in file 'sae1OptTheta.mat'.");
endif


% -------------------------------------------------------------------------

%%======================================================================
%% STEP 2: Train the second sparse autoencoder
%  This trains the second sparse autoencoder on the first autoencoder
%  features.
%  If you've correctly implemented sparseAutoencoderCost.m, you don't need
%  to change anything here.

[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, trainData);

%  Check if file 'sae2OptTheta.mat' exist
if (exist ("sae2OptTheta.mat", "file"))
  load sae2OptTheta.mat;
else
  disp ("File 'sae2OptTheta.mat' not found.");
endif

%  Check if var 'sae2OptTheta' exist
if (exist ("sae2OptTheta", "var"))
  disp ("'sae2OptTheta' parameters loaded from file 'sae2OptTheta.mat'.");
else
  disp ("Training the 2nd layer of Stacked AutoEncoder ...");

  %  Randomly initialize the parameters
  sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1);

  %% ---------------------- YOUR CODE HERE  ---------------------------------
  %  Instructions: Train the second layer sparse autoencoder, this layer has
  %                an hidden size of "hiddenSizeL2" and an inputsize of
  %                "hiddenSizeL1"
  %
  %                You should store the optimal parameters in sae2OptTheta

  addpath minFunc/;               % Use minFunc to minimize the cost function of sparse autoencoder
  options.Method = 'lbfgs';       % Use L-BFGS to optimize sparse autoencoder cost
  options.maxIter = 400;          % Maximum number of iterations of L-BFGS to run 
  options.display = 'on';
  options.useMex = 0;             % Bugfix: error: 'lbfgsC' undefined near line 559 column 25

  [sae2OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
				    hiddenSizeL1, hiddenSizeL2, ...
				    lambda, sparsityParam, ...
				    beta, sae1Features), ...
				sae2Theta, options);
endif

%  Saving 'sae2OptTheta' parameters in file 'sae2OptTheta.mat'
if (! exist ("sae2OptTheta.mat", "file"))
  save sae2OptTheta.mat sae2OptTheta;
  disp ("'sae2OptTheta' parameters saved in file 'sae2OptTheta.mat'.");
endif


% -------------------------------------------------------------------------

%%======================================================================
%% STEP 3: Train the softmax classifier
%  This trains the sparse autoencoder on the second autoencoder features.
%  If you've correctly implemented softmaxCost.m, you don't need
%  to change anything here.

[sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
                                        hiddenSizeL1, sae1Features);

%  Check if file 'saeSoftmaxOptTheta.mat' exist
if (exist ("saeSoftmaxOptTheta.mat", "file"))
  load saeSoftmaxOptTheta.mat;
else
  disp ("File 'saeSoftmaxOptTheta.mat' not found.");
endif

%  Check if var 'saeSoftmaxOptTheta' exist
if (exist ("saeSoftmaxOptTheta", "var"))
  disp ("'saeSoftmaxOptTheta' parameters loaded from file 'saeSoftmaxOptTheta.mat'.");
else
  disp ("Training the Softmax Classifier ...");
                                        
  %  Randomly initialize the parameters
  saeSoftmaxTheta = 0.005 * randn(hiddenSizeL2 * numClasses, 1);


  %% ---------------------- YOUR CODE HERE  ---------------------------------
  %  Instructions: Train the softmax classifier, the classifier takes in
  %                input of dimension "hiddenSizeL2" corresponding to the
  %                hidden layer size of the 2nd layer.
  %
  %                You should store the optimal parameters in saeSoftmaxOptTheta 
  %
  %  NOTE: If you used softmaxTrain to complete this part of the exercise,
  %        set saeSoftmaxOptTheta = softmaxModel.optTheta(:);

  addpath minFunc/;               % Use minFunc to minimize the function
  options.maxIter = 400;          % Maximum number of iterations of L-BFGS to run 
  options.Method = 'lbfgs';       % Use L-BFGS to optimize softmax cost
  minFuncOptions.display = 'on';
  options.useMex = 0;             % Bugfix: error: 'lbfgsC' undefined near line 559 column 25

  [saeSoftmaxOptTheta, cost] = minFunc( @(p) softmaxCost(p, ...
				    numClasses, hiddenSizeL2, lambda, ...
				    sae2Features, trainLabels), ...                                   
				saeSoftmaxTheta, options);
endif

%  Saving 'saeSoftmaxOptTheta' parameters in file 'saeSoftmaxOptTheta.mat'
if (! exist ("saeSoftmaxOptTheta.mat", "file"))
  save saeSoftmaxOptTheta.mat saeSoftmaxOptTheta;
  disp ("'saeSoftmaxOptTheta' parameters saved in file 'saeSoftmaxOptTheta.mat'.");
endif

% -------------------------------------------------------------------------

%%======================================================================
%% STEP 5: Finetune softmax model

% Implement the stackedAECost to give the combined cost of the whole model
% then run this cell.

% Initialize the stack using the parameters learned
stack = cell(2,1);
stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...
                     hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);
stack{2}.w = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), ...
                     hiddenSizeL2, hiddenSizeL1);
stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);

% Initialize the parameters for the deep model
[stackparams, netconfig] = stack2params(stack);
stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];

%  Check if file 'stackedAEOptTheta.mat' exist
if (exist ("stackedAEOptTheta.mat", "file"))
  load stackedAEOptTheta.mat;
else
  disp ("File 'stackedAEOptTheta.mat' not found.");
endif

%  Check if var 'stackedAEOptTheta' exist
if (exist ("stackedAEOptTheta", "var"))
  disp ("'stackedAEOptTheta' parameters loaded from file 'stackedAEOptTheta.mat'.");
else
  disp ("Training the deep network ...");

  %% ---------------------- YOUR CODE HERE  ---------------------------------
  %  Instructions: Train the deep network, hidden size here refers to the '
  %                dimension of the input to the classifier, which corresponds 
  %                to "hiddenSizeL2".

  addpath minFunc/;               % Use minFunc to minimize the function
  options.maxIter = 400;          % Maximum number of iterations of L-BFGS to run 
  options.Method = 'lbfgs';       % Use L-BFGS to optimize softmax cost
  minFuncOptions.display = 'on';
  options.useMex = 0;             % Bugfix: error: 'lbfgsC' undefined near line 559 column 25

  [stackedAEOptTheta, cost] = minFunc( @(p) stackedAECost(p, inputSize, ...
					hiddenSizeL2, numClasses, netconfig, ...
					lambda, trainData, trainLabels), ...                                   
				    stackedAETheta, options);
endif

%  Saving 'stackedAEOptTheta' parameters in file 'stackedAEOptTheta.mat'
if (! exist ("stackedAEOptTheta.mat", "file"))
  save stackedAEOptTheta.mat stackedAEOptTheta
  disp ("'stackedAEOptTheta' parameters saved in file 'stackedAEOptTheta.mat'.");
endif

% -------------------------------------------------------------------------

%%======================================================================
%% STEP 6: Test 
%  Instructions: You will need to complete the code in stackedAEPredict.m
%                before running this part of the code
%

% Get labelled test images
% Note that we apply the same kind of preprocessing as the training set
testData = loadMNISTImages('mnist/t10k-images-idx3-ubyte');
testLabels = loadMNISTLabels('mnist/t10k-labels-idx1-ubyte');

testLabels(testLabels == 0) = 10; % Remap 0 to 10

[pred] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

acc = mean(testLabels(:) == pred(:));
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

[pred] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

acc = mean(testLabels(:) == pred(:));
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

% Accuracy is the proportion of correctly classified images
% The results for our implementation were:
%
% Before Finetuning Test Accuracy: 87.7%
% After Finetuning Test Accuracy:  97.6%
%
% If your values are too low (accuracy less than 95%), you should check 
% your code for errors, and make sure you are training on the 
% entire data set of 60000 28x28 training images 
% (unless you modified the loading code, this should be the case)

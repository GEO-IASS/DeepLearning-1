% Load MNIST images and labels
images = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');
 
visibleSize = 28*28;	% number of input units
%hiddenSize = 196;	% number of hidden units
hiddenSize = 54;	% number of hidden units
sparsityParam = 0.1;	% desired average activation of the hidden units
lambda = 3e-3;		% weight decay parameter
beta = 3;		% weight of sparsity penalty term
numimages = 10000;			% number of images from the MNIST dataset to get
patches = images(:,1:numimages);	% get first 'numimages' images

% We are using display_network from the autoencoder code
display_network(images(:,1:numimages));	% Show the first 'numimages' images
%disp(labels(1:numimages));		% Show the first 'numimages' labels
print -djpeg digits.jpg   % save the visualization to a file 

%  Obtain random parameters theta
%theta = initializeParameters(hiddenSize, visibleSize);

% Get cost and gradientss running sparseAutoencoderCost over ssample images
%[cost, grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, ...
%                                     sparsityParam, beta, patches);

% Compute numerical gradients for debug
%numgrad = computeNumericalGradient( @(x) sparseAutoencoderCost(x, visibleSize, ...
%                                                  hiddenSize, lambda, ...
%                                                  sparsityParam, beta, ...
%                                                  patches), theta);

% Use this to visually compare the gradients side by side
%disp([numgrad grad]); 

% Compare numerically computed gradients with the ones obtained from backpropagation
%diff = norm(numgrad-grad)/norm(numgrad+grad);
%disp(diff); % Should be small. In our implementation, these values are
%            % usually less than 1e-9.

%  Randomly initialize the parameters
theta = initializeParameters(hiddenSize, visibleSize);

%  Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

% Bug: error: 'lbfgsC' undefined near line 559 column 25
options.useMex = 0;

[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, patches), ...
                              theta, options);

% Visualization
W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
display_network(W1'); 
print -djpeg weights.jpg   % save the visualization to a file 

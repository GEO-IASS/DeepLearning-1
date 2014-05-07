function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% inputSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));

%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

% Step1: Perform a feedforward pass, computing the activations for layers L2, L3, 
% up to the output layer, using the equations defining the forward propagation steps.

% Creating cells to store input data and activations
depth = numel(stack);
z = cell(depth+1,1);
a = cell(depth+1,1);
delta = cell(depth+1,1);

% Computing activations for each hidden layer (feedForwardAutoencoder.m)
a{1} = data;
for d = 1:depth
  z{d+1} = stack{d}.w * a{d} + repmat(stack{d}.b,1,M);
  a{d+1} = sigmoid(z{d+1});
end

% Computing Softmax cost and gradients (softmaxCost.m)
h = softmaxTheta*a{depth+1};            % Compute prediction matrix for each class 
h = bsxfun(@minus, h, max(h, [], 1));   % Prevent overflow subtracting maximum theta from 
                                        % each of theta terms before computing the exponential
h = exp(h);                             % Precompute the exponential for each term 
h = bsxfun(@rdivide, h, sum(h));        % Compute the propability matrix for each class
cost = -1/M*(sum(sum(groundTruth.*log(h)))) + lambda/2*(sum(sum(softmaxTheta.^2)));
softmaxThetaGrad = -1/M.*((groundTruth-h)*(a{depth+1}')) + lambda.*softmaxTheta;

% Computing errors and gradients (backpropagation)  
delta{depth+1} = -(softmaxTheta'*(groundTruth-h)) .* (a{depth+1}.*(1-a{depth+1}));
for d = depth:-1:1
  delta{d} = (stack{d}.w'*delta{d+1}) .* (a{d}.*(1-a{d}));
  stackgrad{d}.w = ((delta{d+1}*(a{d}'))./M);
  stackgrad{d}.b = sum(delta{d+1},2)./M;
end
  

% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

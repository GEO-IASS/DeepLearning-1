function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

m = size(data, 2);

groundTruth = full(sparse(labels, 1:m, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

% Compute prediction matrix for each class 
h = theta*data;
% Prevent overflow subtracting maximum theta from each of theta terms before computing the exponential
h = bsxfun(@minus, h, max(h, [], 1));
% Precompute the exponential for each term 
h = exp(h);
% Compute the propability matrix for each class
h = bsxfun(@rdivide, h, sum(h));

% Computing cost function with weight decay
cost = -1/m*(sum(sum(groundTruth.*log(h)))) + lambda/2*(sum(sum(theta.^2)));

% Computing gradients
thetagrad = -1/m.*((groundTruth - h)*data') + lambda.*theta;


% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

